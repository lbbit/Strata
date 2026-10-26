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
    test_prices_and_capacity();
    test_accounting_and_exhaustion();
    test_batch_rollback();
    test_abort_and_publish();
    test_publish_failure();
    test_gather_scatter();
    test_randomized_accounting();
    std::printf("kv_pool_test: %d checks passed\n", checks);
    return 0;
}
