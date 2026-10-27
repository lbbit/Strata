#include "kv_pool.hpp"

#include "strata/core/layer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace strata::core;
using strata::kernels::QsaShapes;
using strata::kernels::qsa_real_shapes;

namespace {

int checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (check %d)\n", message, checks);
        std::exit(1);
    }
}

void expect_error(bool result, const std::string& error, const char* message) {
    check(!result && !error.empty(), message);
}

// Subprocess helper for the STRATA_TEST_KV_POOL_FAIL_AFTER gate: the injection counter is a
// static that must start fresh, so it is exercised in a child of this same binary.
std::string self_exe() {
#if defined(__linux__)
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) return std::string(buf, (size_t) n);
#endif
    return std::string();
}

int run_with_env(const std::string& exe, std::initializer_list<const char*> env) {
#if defined(__linux__)
    std::vector<std::string> store;
    for (const char* e : env) store.emplace_back(e);
    std::vector<char*> envp;
    for (auto& s : store) envp.push_back(const_cast<char*>(s.c_str()));
    envp.push_back(nullptr);
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::execle(exe.c_str(), exe.c_str(), (char*) nullptr, envp.data());
        ::_exit(127);
    }
    if (pid < 0) return -1;
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#else
    (void) exe; (void) env; return -1;
#endif
}

struct HostDevice {
    struct Allocation { std::vector<uint8_t> bytes; };
    std::map<void*, Allocation> allocations;
    int fail_zero_after = -1;
    int fail_upload_after = -1;
    int zero_calls = 0;
    int upload_calls = 0;
    std::vector<void*> free_order;

    static void* allocate(void* opaque, uint64_t bytes, std::string& error) {
        auto& self = *static_cast<HostDevice*>(opaque);
        try {
            auto allocation = std::make_unique<Allocation>();
            allocation->bytes.resize((size_t) bytes, 0xa5);
            void* p = allocation->bytes.data();
            self.allocations.emplace(p, std::move(*allocation));
            return p;
        } catch (const std::bad_alloc&) {
            error = "host fixture allocation failed";
            return nullptr;
        }
    }
    static void release(void* opaque, void* p) {
        auto& self = *static_cast<HostDevice*>(opaque);
        self.free_order.push_back(p);
        self.allocations.erase(p);
    }
    static bool zero(void* opaque, void* p, uint64_t bytes, void*, std::string& error) {
        auto& self = *static_cast<HostDevice*>(opaque);
        if (self.fail_zero_after >= 0 && self.zero_calls++ >= self.fail_zero_after) {
            error = "injected zero failure";
            return false;
        }
        std::memset(p, 0, (size_t) bytes);
        return true;
    }
    static bool upload(void* opaque, void* dst, const void* src, uint64_t bytes,
                       void*, std::string& error) {
        auto& self = *static_cast<HostDevice*>(opaque);
        if (self.fail_upload_after >= 0 && self.upload_calls++ >= self.fail_upload_after) {
            error = "injected upload failure";
            return false;
        }
        std::memcpy(dst, src, (size_t) bytes);
        return true;
    }
    KvPoolDeviceOps ops() {
        return {this, allocate, release, zero, upload};
    }
};

struct Fixture {
    QsaShapes shapes = qsa_real_shapes();
    ModelGeometry geometry;
    HostDevice device;
    std::unique_ptr<GlobalKvPool> pool;
    QsaState a, b;
    std::vector<int32_t> table_a, table_b;

    explicit Fixture(int pages = 64) {
        shapes.page_size = 4;
        pool = GlobalKvPool::create(0, geometry, 12, (int64_t) pages * 8192, error, device.ops(),
                                    KvPoolFormat::Fp16);
        check(pool != nullptr, "host-backed pool creation");
        a.max_cells = b.max_cells = 64;
        a.n_pages = b.n_pages = 16;
        table_a.assign(16, -1);
        table_b.assign(16, -1);
        a.page_table = table_a.data();
        b.page_table = table_b.data();
        pool->bind_main_state(a);
        pool->bind_main_state(b);
    }
    std::string error;
};

uint64_t reference_price(const QsaShapes& s, KvPoolFormat format) {
    const uint64_t rows = (uint64_t) s.page_size * (uint64_t) s.n_head_kv;
    const uint64_t values = rows * (uint64_t) s.head_dim;
    const uint64_t q4 = rows * (uint64_t) (s.head_dim / 32) * 18;
    if (format == KvPoolFormat::Fp16) return values * 4;
    if (format == KvPoolFormat::Int8) return values * 2 + rows * (uint64_t) (s.head_dim / 64) * 4;
    if (format == KvPoolFormat::Q4) return q4 * 2;
    return values + rows * (uint64_t) (s.head_dim / 64) * 2 + q4;
}

void test_backend_diagnostic() {
    std::string error;
    check(!GlobalKvPool::create(0, ModelGeometry{}, 12, 8192, error,
                                KvPoolDeviceOps{}, KvPoolFormat::Fp16),
          "a host-only build cannot create a device pool without injected ops");
    check(error.find("built without a GPU backend") != std::string::npos,
          "missing GPU backend diagnostic identifies the build configuration");
}

void test_prices_and_capacity() {
    QsaShapes s = qsa_real_shapes();
    for (KvPoolFormat format : {KvPoolFormat::Fp16, KvPoolFormat::Int8,
                                KvPoolFormat::Q4, KvPoolFormat::Hybrid}) {
        uint64_t price = 0, bytes = 0;
        std::string error;
        check(kv_pool_page_price(s, format, price, error), "real page price is computable");
        check(price == reference_price(s, format), "page price matches layer formula");
        check(kv_pool_storage_bytes(s, 3, format, bytes, error), "pool storage price");
        check(bytes == price * 3 + (format == KvPoolFormat::Fp16 ? 0 : 64),
              "quantized 64-byte allocation tail");
        HostDevice host;
        auto pool = GlobalKvPool::create(0, ModelGeometry{}, 12, (int64_t) (price * 7 + price - 1),
                                         error, host.ops(), format);
        if (!pool) std::fprintf(stderr, "create error (format %d, price %llu): %s\n", (int) format,
                                (unsigned long long) price, error.c_str());
        check(pool && pool->pages() == 7 && pool->free_pages() == 7,
              "capacity floors to complete pages, once for all layers");
        check(pool->qsa_layers() == 12, "qsa_layers does not multiply capacity");
        check(!GlobalKvPool::create(0, ModelGeometry{}, 12, (int64_t) price - 1, error, host.ops(), format),
              "less than one page is refused");
    }
}

void test_accounting_and_exhaustion() {
    Fixture f(40);
    std::string error;
    check(f.pool->ensure(f.a, 0, 35, nullptr, error), "ensure nine logical pages");
    check(f.pool->free_pages() == 31 && f.pool->pages() - f.pool->free_pages() == 9,
          "actual logical pages, not eight-page rounding, are charged");
    check(f.pool->high_water_pages() == 9 && f.pool->owns(f.a), "high-water and ownership");
    check(f.pool->ensure(f.a, 35, 64, nullptr, error), "grow within logical capacity");
    check(f.pool->free_pages() == 24 && f.pool->high_water_pages() == 16, "grow accounting");
    const int64_t free_before = f.pool->free_pages();
    const std::vector<int32_t> before = f.table_a;
    expect_error(f.pool->ensure(f.a, 64, 65, nullptr, error), error, "out-of-range ensure refused");
    check(f.pool->free_pages() == free_before && f.table_a == before, "rejected ensure is read-only");
    expect_error(f.pool->ensure(f.a, 64, 64 + 1, nullptr, error), error, "cell past max_cells refused");
    check(f.pool->release(f.a, nullptr, error), "release live lease");
    check(f.pool->free_pages() == 40 && f.pool->high_water_pages() == 16 && !f.pool->owns(f.a),
          "release restores free and retains high-water");
    check(f.pool->release(f.a, nullptr, error) && f.pool->free_pages() == 40,
          "repeated release is safe");
    expect_error(f.pool->ensure(f.a, -1, 0, nullptr, error), error, "negative commit refused");
    expect_error(f.pool->ensure(f.a, 8, 7, nullptr, error), error, "write before commit refused");
    expect_error(f.pool->ensure(f.a, 8, 7, nullptr, error), error, "decreasing committed prefix refused");
}

void test_batch_rollback() {
    Fixture f(9);
    std::string error;
    check(f.pool->ensure(f.a, 0, 4, nullptr, error), "reserve first page");
    const std::vector<int32_t> before = f.table_a;
    expect_error(f.pool->ensure(f.a, 4, 40, nullptr, error), error, "second batch cannot be half acquired");
    check(f.pool->free_pages() == 8, "failed batch leaks no pages");
    check(f.table_a == before && f.pool->high_water_pages() == 1,
          "failed batch leaves live mapping and watermarks unchanged");
    check(f.pool->ensure(f.a, 4, 36, nullptr, error), "remaining eight pages fit after retry");
    check(f.pool->free_pages() == 0, "all nine pages accounted");
    expect_error(f.pool->ensure(f.b, 0, 4, nullptr, error), error, "another state cannot steal a live page");
    check(f.pool->free_pages() == 0, "failed other-state reservation is balanced");
    check(!f.pool->owns(f.b), "first-time failed ensure leaves no empty lease");
    std::vector<int32_t> empty;
    expect_error(f.pool->snapshot_page_table(f.b, empty, error), error,
                 "failed first-time ensure has no mirror");
}

void test_fail_after_gate() {
    // STRATA_TEST_KV_POOL_FAIL_AFTER=N: the N-th successful reservation fails before publish and
    // rolls back. A subprocess (fresh static counter) proves: reservation rollback, no lease/mapping
    // change, free accounting balanced, and a later ensure still works (nothing latched/broken).
    const std::string exe = self_exe();
    if (exe.empty()) { check(true, "fail-after gate: self exe unknown, skipped"); return; }
    {
        const int rc = run_with_env(exe, {"STRATA_TEST_KV_POOL_FAIL_AFTER=2", "STRATA_KV_POOL_FAIL_AFTER_CHILD=1"});
        check(rc == 0, "fail-after gate: injected 2nd reservation, engine keeps working (child rc 0)");
    }
}

int fail_after_child() {
    // Runs only in the spawned child (STRATA_KV_POOL_FAIL_AFTER=2): 1st ensure succeeds, 2nd must
    // fail with the injected error and leave free/table/ownership unchanged; 3rd succeeds again.
    Fixture f(40);
    std::string error;
    check(f.pool->ensure(f.a, 0, 4, nullptr, error), "fail-after: first reservation succeeds");
    const int64_t free_after_first = f.pool->free_pages();
    const std::vector<int32_t> table_after_first = f.table_a;
    const bool failed = !f.pool->ensure(f.a, 4, 8, nullptr, error);
    check(failed && error.find("injected failure") != std::string::npos, "fail-after: 2nd reservation injected-fails");
    check(f.pool->free_pages() == free_after_first && f.table_a == table_after_first,
          "fail-after: injected failure rolls back free and mapping exactly");
    check(f.pool->ensure(f.a, 4, 8, nullptr, error), "fail-after: 3rd reservation succeeds (nothing latched)");
    check(f.pool->free_pages() < free_after_first, "fail-after: recovery actually acquires pages");
    return 0;
}

void test_abort_and_publish() {
    Fixture f(18);
    std::string error;
    check(f.pool->ensure(f.a, 0, 16, nullptr, error), "old lease four pages");
    check(f.pool->ensure(f.b, 0, 16, nullptr, error), "other live lease");
    const std::vector<int32_t> old_a = f.table_a, old_b = f.table_b;
    const int64_t old_free = f.pool->free_pages();
    KvLeaseTxn txn;
    check(f.pool->prepare_prefix(f.a, 28, txn, error), "prepare replacement seven pages");
    expect_error(f.pool->prepare_prefix(f.a, 12, txn, error), error,
                 "one state cannot be reserved twice in a transaction");
    check(f.pool->prepare_prefix(f.b, 12, txn, error), "prepare a second state in the same transaction");
    check(f.pool->free_pages() == old_free - 10 && txn.entries.size() == 2,
          "transaction owns both reservations");
    f.pool->abort_prepare(txn);
    check(f.pool->free_pages() == old_free && f.table_a == old_a && f.table_b == old_b,
          "multi-state abort releases every reservation");
    check(f.pool->prepare_prefix(f.a, 28, txn, error), "prepare replacement seven pages");
    check(f.table_a == old_a && f.pool->free_pages() == old_free - 7,
          "prepare reserves without publishing");
    f.pool->abort_prepare(txn);
    check(f.pool->free_pages() == old_free && f.table_a == old_a && f.table_b == old_b,
          "abort is byte-for-byte identical to pre-prepare state");
    check(f.pool->prepare_prefix(f.a, 28, txn, error), "prepare after abort");
    expect_error(f.pool->prepare_prefix(f.b, 40, txn, error), error, "all-or-nothing exhaustion in second state");
    check(f.pool->free_pages() == old_free - 7 && f.table_b == old_b,
          "failed later prepare does not publish anything");
    f.pool->abort_prepare(txn);
    check(f.pool->free_pages() == old_free, "multi-state abort releases all reservations");

    check(f.pool->prepare_prefix(f.a, 28, txn, error), "prepare publishable replacement");
    check(f.pool->publish_prepared(f.a, txn, nullptr, error), "publish replacement");
    check(f.pool->free_pages() == old_free - 7 && f.pool->owns(f.b),
          "old pages remain reserved until scatter completion");
    f.pool->finish_prepared(txn);
    check(f.pool->free_pages() == old_free - 3 && txn.retired.empty(),
          "finishing after scatter releases the old four pages");
    check(f.a.kv_pool_page_mirror != nullptr && f.a.kv_pool_committed == 28 && f.a.kv_pool_write_top == 28,
          "state handles name the live host mirror and committed/write tops");
    check(*f.a.kv_pool_page_mirror == f.table_a, "state mirror handle aliases the authoritative table");
    check(f.table_b == old_b, "publish leaves other state untouched");
    for (size_t i = 0; i < 7; ++i) {
        check(f.table_a[i] >= 0 && f.table_a[i] < f.pool->pages(), "published page is in range");
        check(f.pool->physical_page(f.a, (int64_t) i) == f.table_a[i], "mirror agrees with device fixture");
    }
    check(f.pool->release(f.a, nullptr, error) && f.pool->release(f.b, nullptr, error),
          "release both states");
    check(f.pool->free_pages() == f.pool->pages(), "no physical page appears in two live leases");
}

void test_publish_failure() {
    Fixture f(16);
    std::string error;
    check(f.pool->ensure(f.a, 0, 4, nullptr, error), "ensure initial lease");
    const auto old_table = f.table_a;
    const auto old_mirror = [&] { std::vector<int32_t> t; f.pool->snapshot_page_table(f.a, t, error); return t; }();
    const int64_t free = f.pool->free_pages();
    f.device.fail_upload_after = 0;
    expect_error(f.pool->ensure(f.a, 4, 8, nullptr, error), error, "upload failure is propagated");
    check(f.table_a == old_table && f.pool->free_pages() == free, "failed ensure restores accounting");
    std::vector<int32_t> mirror;
    check(f.pool->snapshot_page_table(f.a, mirror, error) && mirror == old_mirror,
          "failed ensure preserves authoritative mirror");
    f.device.fail_upload_after = -1;

    KvLeaseTxn txn;
    check(f.pool->prepare_prefix(f.a, 12, txn, error), "prepare failed-publish fixture");
    f.device.fail_upload_after = 0;
    expect_error(f.pool->publish_prepared(f.a, txn, nullptr, error), error, "publish failure propagated");
    check(f.table_a == old_table && f.pool->free_pages() == free - 3,
          "failed publish leaves old live lease and new reservation intact");
    f.pool->abort_prepare(txn);
    check(f.pool->free_pages() == free && f.table_a == old_table,
          "abort after failed publish restores free count");
}

void test_gather_scatter() {
    constexpr uint64_t page_bytes = 13;
    std::vector<uint8_t> k(page_bytes * 5, 0), v(page_bytes * 5, 0);
    std::vector<void*> parts = {k.data(), v.data()};
    const std::vector<uint64_t> sizes = {page_bytes, page_bytes};
    const std::vector<int32_t> table = {4, 1, 3};
    std::vector<uint8_t> image(37);
    for (size_t i = 0; i < image.size(); ++i) image[i] = (uint8_t) (3 + i * 11);
    std::string error;
    check(GlobalKvPool::scatter_logical_bytes(parts, sizes, table, 0, 0, image.data(), image.size(), error),
          "scatter across permuted pages");
    std::vector<uint8_t> gathered(image.size(), 0xff);
    check(GlobalKvPool::gather_logical_bytes(parts, sizes, table, 0, 0, gathered.data(), gathered.size(), error),
          "gather across permuted pages");
    check(gathered == image, "gather(scatter(image)) is byte-identical");
    check(std::all_of(k.begin() + 2 * page_bytes, k.begin() + 3 * page_bytes,
                      [](uint8_t x) { return x == 0; }),
          "unmapped physical page is untouched");
    const std::vector<uint8_t> old_k = k;
    const std::vector<uint8_t> slice(9, 0x6d);
    check(GlobalKvPool::scatter_logical_bytes(parts, sizes, table, 0, 11, slice.data(), slice.size(), error),
          "non-page-aligned scatter");
    check(GlobalKvPool::gather_logical_bytes(parts, sizes, table, 0, 11, gathered.data(), slice.size(), error),
          "non-page-aligned gather");
    check(std::equal(slice.begin(), slice.end(), gathered.begin()), "unaligned bytes survive");
    check(std::equal(old_k.begin(), old_k.begin() + 11, k.begin()), "scatter has no prefix overwrite");
    expect_error(GlobalKvPool::gather_logical_bytes(parts, sizes, table, 0, 3 * page_bytes,
                                                    gathered.data(), 1, error), error,
                 "gather refuses an unmapped logical page");
    expect_error(GlobalKvPool::scatter_logical_bytes(parts, sizes, table, 1, 3 * page_bytes - 1,
                                                     slice.data(), 2, error), error,
                 "scatter refuses unmapped page across boundary");
}

void test_layer_sizing_and_identity_fixture() {
    ModelGeometry g;
    constexpr int64_t cells = 65536;
    const QsaShapes s = qsa_real_shapes();
    const int64_t pages = (cells + s.page_size - 1) / s.page_size;
    for (KvPoolFormat format : {KvPoolFormat::Fp16, KvPoolFormat::Int8,
                                KvPoolFormat::Q4, KvPoolFormat::Hybrid}) {
        KvStateSizingConfig off{format};
        KvStateSizingConfig on{format, 0, false, true};
        const uint64_t private_bytes = qsa_state_sizing_bytes(g, cells, true, 0, off);
        const uint64_t pooled_bytes = qsa_state_sizing_bytes(g, cells, true, 0, on);
        uint64_t kv_bytes = 0;
        std::string error;
        check(kv_pool_storage_bytes(s, pages, format, kv_bytes, error), "test format storage bytes");
        const uint64_t expected_delta = kv_bytes + 4 * 16;
        std::printf("kv_pool sizing format=%d off=%llu on=%llu delta=%llu non_kv_delta=%llu\n",
                    (int) format, (unsigned long long) private_bytes, (unsigned long long) pooled_bytes,
                    (unsigned long long) (private_bytes - pooled_bytes), 0ull);
        check(private_bytes - pooled_bytes == expected_delta,
              "pool off/on differ by exactly the private K/V allocation");
        // This is the non-KV equality assertion, not a second capacity formula:
        // removing the measured K/V delta must make the two byte counts equal.
        check(private_bytes - expected_delta == pooled_bytes,
              "all non-KV qsa_state_bytes fields are exactly equal");
        check(qsa_state_sizing_bytes(g, cells, false, 0, off) -
              qsa_state_sizing_bytes(g, cells, false, 0, on) == expected_delta,
              "non-KV equality also holds with borrowed RoPE");
        check(qsa_state_sizing_bytes(g, cells, true, 128, off) ==
              qsa_state_sizing_bytes(g, cells, true, 128, on),
              "drafter sizing never changes under a pool binding");
        KvStateSizingConfig streamed{format, 128, false, false};
        KvStateSizingConfig streamed_pool = streamed;
        streamed_pool.pooled = true;
        check(qsa_state_sizing_bytes(g, cells, true, 0, streamed) ==
              qsa_state_sizing_bytes(g, cells, true, 0, streamed_pool),
              "streamed sizing never changes under a pool binding");
        KvStateSizingConfig elastic{format, 0, true, false};
        KvStateSizingConfig elastic_pool = elastic;
        elastic_pool.pooled = true;
        check(qsa_state_sizing_bytes(g, cells, true, 0, elastic) ==
              qsa_state_sizing_bytes(g, cells, true, 0, elastic_pool),
              "elastic sizing never changes under a pool binding");
    }

    // One state exclusively owns this small pool: [0,n_pages) is therefore
    // both the logical and physical range. This is not a multi-session mode.
    HostDevice device;
    std::string error;
    uint64_t page_price = 0;
    check(kv_pool_page_price(s, KvPoolFormat::Fp16, page_price, error), "identity fixture page price");
    auto pool = GlobalKvPool::create(0, g, 1, pages * (int64_t) page_price,
                                     error, device.ops(), KvPoolFormat::Fp16);
    check(pool && pool->pages() == pages, "identity fixture has exactly one page per logical page");
    check(pool != nullptr, "identity fixture pool creation");
    QsaState st;
    std::vector<int32_t> table((size_t) pages, -1);
    st.page_table = table.data();
    st.n_pages = pages;
    st.max_cells = cells;
    pool->bind_main_state(st);
    check(st.kv_pool == pool.get() && st.kv_pool_page_mirror == nullptr && st.k_pool != nullptr,
          "binding has stable shared bases and an empty lease");
    check(pool->bind_identity_for_test(st, nullptr, error), "exclusive identity lease");
    check(std::all_of(table.begin(), table.end(), [i = 0](int32_t p) mutable { return p == i++; }),
          "identity lease publishes table[i]=i for every logical page");
    check(st.kv_pool_page_mirror != nullptr && *st.kv_pool_page_mirror == table,
          "identity host mirror is attached to QsaState");
    check(pool->free_pages() == 0 && st.kv_pool_write_top == cells,
          "identity fixture accounts every page");
    uint8_t* original_pool = reinterpret_cast<uint8_t*>(st.k_pool);
    check(pool->release(st, nullptr, error), "identity fixture release");
    check(reinterpret_cast<uint8_t*>(st.k_pool) == original_pool && st.kv_pool_page_mirror == nullptr &&
          pool->free_pages() == pool->pages(), "release keeps the binding and stable base");
}

void test_randomized_accounting() {
    Fixture f(64);
    std::mt19937 rng(0x5eed);
    std::set<int> live;
    std::string error;
    for (int round = 0; round < 100000; ++round) {
        QsaState& st = (rng() & 1) ? f.a : f.b;
        if (rng() % 3 == 0) {
            check(f.pool->release(st, nullptr, error), "random release");
            std::vector<int32_t> table;
            if (f.pool->snapshot_page_table(st, table, error)) live.clear();
        } else {
            const int64_t cells = 1 + (int64_t) (rng() % 64);
            const bool ok = f.pool->ensure(st, 0, cells, nullptr, error);
            check(ok || !error.empty(), "random ensure has a verdict");
        }
        live.clear();
        for (QsaState* state : {&f.a, &f.b}) {
            std::vector<int32_t> table;
            if (!f.pool->snapshot_page_table(*state, table, error)) continue;
            for (int32_t p : table) if (p >= 0) {
                check(p < f.pool->pages() && live.insert(p).second, "no double allocation");
            }
        }
        check((int64_t) live.size() + f.pool->free_pages() == f.pool->pages(), "free plus used conservation");
        check(f.pool->free_pages() >= 0 && f.pool->high_water_pages() <= f.pool->pages(),
              "watermarks remain bounded");
    }
    check(f.pool->release(f.a, nullptr, error) && f.pool->release(f.b, nullptr, error),
          "final randomized cleanup");
    check(f.pool->free_pages() == f.pool->pages(), "final free-list balance");
}

}  // namespace

int main() {
    if (std::getenv("STRATA_KV_POOL_FAIL_AFTER_CHILD") != nullptr) return fail_after_child();
    test_backend_diagnostic();
    test_prices_and_capacity();
    test_accounting_and_exhaustion();
    test_batch_rollback();
    test_fail_after_gate();
    test_abort_and_publish();
    test_publish_failure();
    test_gather_scatter();
    test_layer_sizing_and_identity_fixture();
    test_randomized_accounting();
    std::printf("kv_pool_test: %d checks passed\n", checks);
    return 0;
}
