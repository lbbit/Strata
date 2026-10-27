#include "kv_pool.hpp"

#include "strata/core/layer.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

#if defined(STRATA_USE_CUDA) || defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
#include <cuda_runtime.h>
#define STRATA_KV_POOL_GPU 1
#else
#define STRATA_KV_POOL_GPU 0
#endif

namespace strata::core {
namespace {

constexpr int64_t kAcquireBatch = 8;

bool checked_mul(uint64_t a, uint64_t b, uint64_t& result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t& result) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    result = a + b;
    return true;
}

bool valid_shapes(const strata::kernels::QsaShapes& s, KvPoolFormat format,
                  std::string& error) {
    if (s.page_size <= 0 || s.n_head_kv <= 0 || s.head_dim <= 0) {
        error = "kv_pool: invalid QSA page geometry";
        return false;
    }
    if (format != KvPoolFormat::Fp16 && s.head_dim % strata::kernels::KV_Q8_GROUP != 0) {
        error = "kv_pool: quantized KV head_dim must be a multiple of 64";
        return false;
    }
    return true;
}

bool runtime_zero(void* context, void* pointer, uint64_t bytes, void* stream,
                  std::string& error) {
    (void) context;
#if STRATA_KV_POOL_GPU
    if (bytes > std::numeric_limits<size_t>::max()) {
        error = "kv_pool: zeroing size exceeds size_t";
        return false;
    }
    if (cudaMemsetAsync(pointer, 0, (size_t) bytes, (cudaStream_t) stream) != cudaSuccess) {
        error = std::string("kv_pool: device zeroing failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
#else
    (void) pointer; (void) bytes; (void) stream;
    error = "kv_pool: device backend unavailable in a non-GPU build";
    return false;
#endif
}

bool runtime_upload(void* context, void* destination, const void* source, uint64_t bytes,
                    void* stream, std::string& error) {
    (void) context;
#if STRATA_KV_POOL_GPU
    if (bytes > std::numeric_limits<size_t>::max()) {
        error = "kv_pool: upload size exceeds size_t";
        return false;
    }
    if (cudaMemcpyAsync(destination, source, (size_t) bytes, cudaMemcpyHostToDevice,
                        (cudaStream_t) stream) != cudaSuccess) {
        error = std::string("kv_pool: page-table upload failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
#else
    (void) destination; (void) source; (void) bytes; (void) stream;
    error = "kv_pool: device backend unavailable in a non-GPU build";
    return false;
#endif
}

int32_t table_entry(const std::vector<int32_t>& table, int64_t logical_page) {
    if (logical_page < 0 || (size_t) logical_page >= table.size()) return -1;
    return table[(size_t) logical_page];
}

}  // namespace

uint64_t qsa_state_sizing_bytes(const ModelGeometry& g, int64_t max_cells,
                                bool with_rope, int64_t ring_cells,
                                const KvStateSizingConfig& config) {
    const auto s = qsa_shapes(g);
    int64_t pages = (max_cells + s.page_size - 1) / s.page_size;
    int64_t slots = pages;
    int64_t pooled_rows = max_cells / s.idx_block + 2;
    int mode = 0;
    if (config.resident_cells > 0 && ring_cells >= 0) {
        if (ring_cells > 0) {
            const int64_t resident = (ring_cells + s.page_size - 1) / s.page_size;
            if (resident < pages) { mode = 2; slots = resident; pooled_rows = 2; }
        } else {
            const int64_t resident = (std::max(config.resident_cells, int64_t{20480}) +
                                      s.page_size - 1) / s.page_size;
            if (resident < pages) { mode = 1; slots = resident; }
        }
    }
    const bool pooled = config.pooled && ring_cells == 0 && mode == 0 && !config.elastic;
    const bool elastic = config.elastic && mode == 0;
    const bool hybrid = config.format == KvPoolFormat::Hybrid && ring_cells <= 0;
    const KvPoolFormat format = hybrid ? KvPoolFormat::Hybrid
                                      : ring_cells > 0 && config.format != KvPoolFormat::Fp16
                                        ? KvPoolFormat::Int8 : config.format;
    uint64_t n = 0;
    if (!pooled && !elastic) {
        uint64_t storage = 0;
        std::string error;
        if (!kv_pool_storage_bytes(s, slots, format, storage, error)) return 0;
        n += storage + 4 * 16;
    }
    n += (uint64_t) pages * 4;
    if (mode == 1) n += strata::kernels::kv_stream_map_bytes(slots) + 6 * 16;
    n += (uint64_t) (s.idx_block - 1) * s.idx_dim * 4;
    n += (uint64_t) s.idx_dim * 4;
    n += (uint64_t) pooled_rows * s.idx_dim * 4;
    n += 16;
    if (with_rope) n += (uint64_t) max_cells * (s.n_rot / 2) * 4 * 2;
    n += strata::kernels::qsa_step_bytes() + 16;
    n += (uint64_t) s.n_head * 4;
    return (n + 15) / 16 * 16 + 256;
}

bool kv_pool_page_price(const strata::kernels::QsaShapes& s, KvPoolFormat format,
                        uint64_t& bytes, std::string& error) {
    bytes = 0;
    if (!valid_shapes(s, format, error)) return false;
    uint64_t rows = 0;
    if (!checked_mul((uint64_t) s.page_size, (uint64_t) s.n_head_kv, rows)) {
        error = "kv_pool: page row count overflows";
        return false;
    }
    uint64_t values = 0;
    if (!checked_mul(rows, (uint64_t) s.head_dim, values)) {
        error = "kv_pool: page value count overflows";
        return false;
    }
    const uint64_t q4_head = strata::kernels::kv_q4_bytes_per_head((int) s.head_dim);
    if (format == KvPoolFormat::Hybrid) {
        // Exactly layer.cpp's hybrid formula: K codes + K scales + V Q4_0.
        uint64_t q4 = 0;
        if (!checked_mul(rows, q4_head, q4) || !checked_add(values, q4, bytes) ||
            !checked_mul(rows, (uint64_t) (s.head_dim / strata::kernels::KV_Q8_GROUP) * 2, values)) {
            error = "kv_pool: hybrid page price overflows";
            return false;
        }
        if (!checked_add(bytes, values, bytes)) {
            error = "kv_pool: hybrid page price overflows";
            return false;
        }
    } else {
        const uint64_t multiplier = format == KvPoolFormat::Fp16 ? 4 : format == KvPoolFormat::Int8 ? 2 : 0;
        if (format == KvPoolFormat::Q4) {
            uint64_t both = 0;
            if (!checked_mul(q4_head, 2, both) || !checked_mul(rows, both, bytes)) {
                error = "kv_pool: Q4 page price overflows";
                return false;
            }
        } else if (!checked_mul(values, multiplier, bytes)) {
            error = "kv_pool: page price overflows";
            return false;
        }
        if (format == KvPoolFormat::Int8 &&
            !checked_mul(rows, (uint64_t) (s.head_dim / strata::kernels::KV_Q8_GROUP) * 4, values)) {
            error = "kv_pool: INT8 scale price overflows";
            return false;
        }
        if (format == KvPoolFormat::Int8 && !checked_add(bytes, values, bytes)) {
            error = "kv_pool: INT8 page price overflows";
            return false;
        }
    }
    if (bytes == 0 || bytes > (uint64_t) std::numeric_limits<int64_t>::max()) {
        error = "kv_pool: invalid page price";
        return false;
    }
    return true;
}

bool kv_pool_storage_bytes(const strata::kernels::QsaShapes& s, int64_t pages,
                           KvPoolFormat format, uint64_t& bytes, std::string& error) {
    uint64_t price = 0;
    if (!kv_pool_page_price(s, format, price, error)) return false;
    if (pages < 0 || !checked_mul(price, (uint64_t) pages, bytes)) {
        error = "kv_pool: pool byte count overflows";
        bytes = 0;
        return false;
    }
    // Quantized allocations have a 64-byte safety tail in qsa_state_bytes().
    if (format != KvPoolFormat::Fp16 && !checked_add(bytes, 64, bytes)) {
        error = "kv_pool: quantized allocation tail overflows";
        bytes = 0;
        return false;
    }
    return true;
}

GlobalKvPool::GlobalKvPool(const ModelGeometry& g, int64_t qsa_layers, KvPoolFormat format,
                           int device, KvPoolDeviceOps ops)
    : shapes_(qsa_shapes(g)), format_(format), qsa_layers_(qsa_layers), device_(device), ops_(ops) {}

std::unique_ptr<GlobalKvPool> GlobalKvPool::create(int device, const ModelGeometry& g,
                                                    int64_t qsa_layers, int64_t capacity_bytes,
                                                    std::string& error, KvPoolDeviceOps ops,
                                                    std::optional<KvPoolFormat> format_override) {
    error.clear();
#if defined(STRATA_KV_POOL_HOST_TEST)
    if (!format_override) {
        error = "kv_pool: host test requires an explicit KV format";
        return nullptr;
    }
#endif
    KvPoolFormat format = KvPoolFormat::Fp16;
    if (format_override) format = *format_override;
#if !defined(STRATA_KV_POOL_HOST_TEST)
    else format = qsa_kv_hybrid() ? KvPoolFormat::Hybrid
                : qsa_kv_q4() ? KvPoolFormat::Q4
                : qsa_kv_int8() ? KvPoolFormat::Int8 : KvPoolFormat::Fp16;
#endif
    const strata::kernels::QsaShapes shapes = qsa_shapes(g);
    uint64_t price = 0;
    if (device < 0 || qsa_layers <= 0 || capacity_bytes <= 0 ||
        !kv_pool_page_price(shapes, format, price, error)) {
        if (error.empty()) error = "kv_pool: invalid device, layer count, or capacity";
        return nullptr;
    }
    if ((uint64_t) capacity_bytes < price) {
        error = "kv_pool: capacity is smaller than one complete page";
        return nullptr;
    }
    const int64_t count = capacity_bytes / (int64_t) price;
    if (count > std::numeric_limits<int32_t>::max()) {
        error = "kv_pool: physical page number exceeds int32";
        return nullptr;
    }
    const bool injected = ops.allocate && ops.release && ops.zero && ops.upload;
    if (ops.allocate || ops.release || ops.zero || ops.upload) {
        if (!injected) {
            error = "kv_pool: incomplete injected device backend";
            return nullptr;
        }
    } else {
#if !STRATA_KV_POOL_GPU
        error = "kv_pool: create built without a GPU backend (CUDA/HIP); use a GPU build or an injected device backend";
        return nullptr;
#endif
    }
    uint64_t total = 0;
    if (!kv_pool_storage_bytes(shapes, count, format, total, error)) return nullptr;
    if (total > std::numeric_limits<size_t>::max()) {
        error = "kv_pool: pool allocation exceeds size_t";
        return nullptr;
    }

    std::unique_ptr<GlobalKvPool> pool(new (std::nothrow) GlobalKvPool(g, qsa_layers, format, device, ops));
    if (!pool) {
        error = "kv_pool: host allocation failed";
        return nullptr;
    }
    pool->pages_ = count;
    pool->page_price_ = price;
    const int64_t heads_per_page = shapes.page_size * shapes.n_head_kv;
    const uint64_t values = (uint64_t) heads_per_page * (uint64_t) shapes.head_dim;
    const uint64_t scales = (uint64_t) heads_per_page * (uint64_t) (shapes.head_dim / 64) * 2;
    const uint64_t q4 = (uint64_t) heads_per_page * strata::kernels::kv_q4_bytes_per_head((int) shapes.head_dim);
    switch (format) {
    case KvPoolFormat::Fp16: pool->part_page_bytes_ = {values * 2, values * 2}; break;
    case KvPoolFormat::Int8: pool->part_page_bytes_ = {values, values, scales, scales}; break;
    case KvPoolFormat::Q4: pool->part_page_bytes_ = {q4, q4}; break;
    case KvPoolFormat::Hybrid: pool->part_page_bytes_ = {values, scales, q4}; break;
    }
    uint64_t sum = 0;
    for (uint64_t n : pool->part_page_bytes_) {
        if (!checked_add(sum, n, sum)) {
            error = "kv_pool: component price sum overflows";
            return nullptr;
        }
    }
    if (sum != price) {
        error = "kv_pool: component prices do not add up to page price";
        return nullptr;
    }
    try {
        pool->free_.resize((size_t) count);
    } catch (const std::bad_alloc&) {
        error = "kv_pool: host free-list allocation failed";
        return nullptr;
    }

#if STRATA_KV_POOL_GPU
    if (!injected) {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
            error = std::string("kv_pool: cannot select device: ") + cudaGetErrorString(cudaGetLastError());
            return nullptr;
        }
        struct Restore {
            int device;
            ~Restore() { (void) cudaSetDevice(device); }
        } restore{previous};
#else
    {
#endif
        for (size_t i = 0; i < pool->part_page_bytes_.size(); ++i) {
            const uint64_t component = pool->part_page_bytes_[i] * (uint64_t) count;
            const uint64_t allocation = component + (format == KvPoolFormat::Fp16 ? 0 : 64);
            if (injected) pool->parts_.push_back(ops.allocate(ops.context, allocation, error));
#if STRATA_KV_POOL_GPU
            else {
                void* pointer = nullptr;
                const cudaError_t status = cudaMalloc(&pointer, (size_t) allocation);
                if (status != cudaSuccess) error = std::string("kv_pool: cudaMalloc failed: ") + cudaGetErrorString(status);
                pool->parts_.push_back(pointer);
            }
#endif
            if (!pool->parts_.back()) {
                if (error.empty()) error = "kv_pool: device component allocation failed";
                return nullptr;
            }
        }
    }
    // Pop order is ascending, which makes fixtures and diagnostics stable; the
    // allocator never relies on physical identity for correctness.
    for (int64_t i = count; i-- > 0;) pool->free_[(size_t) i] = count - 1 - i;
    return pool;
}

GlobalKvPool::~GlobalKvPool() {
#if STRATA_KV_POOL_GPU
    int previous = 0;
    const bool restore = !ops_.allocate && cudaGetDevice(&previous) == cudaSuccess &&
                         cudaSetDevice(device_) == cudaSuccess;
#endif
    for (void* p : parts_) {
        if (ops_.allocate) ops_.release(ops_.context, p);
#if STRATA_KV_POOL_GPU
        else (void) cudaFree(p);
#endif
    }
#if STRATA_KV_POOL_GPU
    if (restore) (void) cudaSetDevice(previous);
#endif
}

int64_t GlobalKvPool::free_pages() const {
    std::lock_guard lock(mutex_);
    return (int64_t) free_.size();
}

int64_t GlobalKvPool::high_water_pages() const {
    std::lock_guard lock(mutex_);
    return high_water_;
}

bool GlobalKvPool::owns(const QsaState& st) const {
    std::lock_guard lock(mutex_);
    return leases_.find(const_cast<QsaState*>(&st)) != leases_.end();
}

bool GlobalKvPool::snapshot_page_table(const QsaState& st, std::vector<int32_t>& table,
                                       std::string& error) const {
    std::lock_guard lock(mutex_);
    const auto it = leases_.find(const_cast<QsaState*>(&st));
    if (it == leases_.end()) {
        table.clear();
        error = "kv_pool: state has no live lease";
        return false;
    }
    table = it->second.table;
    return true;
}

int32_t GlobalKvPool::physical_page(const QsaState& st, int64_t logical_page) const {
    std::lock_guard lock(mutex_);
    const auto it = leases_.find(const_cast<QsaState*>(&st));
    return it == leases_.end() ? -1 : table_entry(it->second.table, logical_page);
}

void GlobalKvPool::bind_main_state(QsaState& st) {
    // Binding owns addresses and an initially empty host mirror, not pages.
    assert(st.kv_mode == 0 && st.kv_elastic < 0 && st.page_table != nullptr);
    if (st.kv_mode != 0 || st.kv_elastic >= 0 || !st.page_table) return;
    st.kv_pool = this;
    st.kv_pool_committed = 0;
    st.kv_pool_write_top = 0;
    st.kv_pool_page_mirror = nullptr;
    st.k_pool = static_cast<uint16_t*>(parts_[0]);
    st.v_pool = parts_.size() > 1 ? static_cast<uint16_t*>(parts_[1]) : nullptr;
    st.k_q = st.v_q = nullptr;
    st.k_scale = st.v_scale = nullptr;
    st.k_q4 = st.v_q4 = nullptr;
    switch (format_) {
    case KvPoolFormat::Fp16: break;
    case KvPoolFormat::Int8:
        st.k_q = static_cast<int8_t*>(parts_[0]); st.v_q = static_cast<int8_t*>(parts_[1]);
        st.k_scale = static_cast<uint16_t*>(parts_[2]); st.v_scale = static_cast<uint16_t*>(parts_[3]); break;
    case KvPoolFormat::Q4:
        st.k_q4 = static_cast<uint8_t*>(parts_[0]); st.v_q4 = static_cast<uint8_t*>(parts_[1]); break;
    case KvPoolFormat::Hybrid:
        st.k_q = static_cast<int8_t*>(parts_[0]); st.k_scale = static_cast<uint16_t*>(parts_[1]);
        st.v_q4 = static_cast<uint8_t*>(parts_[2]); break;
    }
}

bool GlobalKvPool::bind_identity_for_test(QsaState& st, void* caller_stream,
                                          std::string& error) {
    if (!ensure(st, 0, st.max_cells, caller_stream, error)) return false;
    std::lock_guard lock(mutex_);
    const auto it = leases_.find(&st);
    if (it == leases_.end()) {
        error = "kv_pool: identity fixture has no lease";
        return false;
    }
    for (size_t i = 0; i < it->second.table.size(); ++i) {
        if (it->second.table[i] != (int32_t) i) {
            error = "kv_pool: identity fixture requires exclusive pages [0,n_pages)";
            return false;
        }
        st.kv_pool_page_mirror = &it->second.table;
    }
    return true;
}

void GlobalKvPool::recycle(std::vector<int32_t>& pages) {
    for (int32_t p : pages) free_.push_back(p);
    pages.clear();
}

void GlobalKvPool::reset(QsaState& st) {
    // The binding and stable pool bases survive release; only the lease ends.
    st.kv_pool_committed = 0;
    st.kv_pool_write_top = 0;
    st.kv_pool_page_mirror = nullptr;
}

bool GlobalKvPool::reserve(Lease& lease, QsaState& st, int64_t committed_cells,
                           int64_t new_write_cells, std::string& error) {
    const std::vector<int32_t> free_before = free_;
    // Caller holds mutex_.  A lease table is sized from QsaState's immutable
    // logical capacity and entries stay -1 until actually acquired.
    if (committed_cells < 0 || new_write_cells < committed_cells ||
        new_write_cells > st.max_cells || st.n_pages <= 0 || !st.page_table) {
        error = "kv_pool: invalid write range or page-table capacity";
        return false;
    }
    const int64_t wanted = (new_write_cells + shapes_.page_size - 1) / shapes_.page_size;
    if (wanted > st.n_pages) {
        error = "kv_pool: requested range exceeds logical page-table capacity";
        return false;
    }
    if (lease.table.empty()) {
        try { lease.table.assign((size_t) st.n_pages, -1); }
        catch (const std::bad_alloc&) { error = "kv_pool: host mirror allocation failed"; return false; }
    } else if ((int64_t) lease.table.size() != st.n_pages) {
        error = "kv_pool: logical page-table capacity changed";
        return false;
    }
    try { lease.pages.reserve(lease.pages.size() + (size_t) wanted); }
    catch (const std::bad_alloc&) { error = "kv_pool: host reservation allocation failed"; return false; }
    for (int64_t logical = 0; logical < wanted;) {
        const int64_t end = std::min(wanted, logical + kAcquireBatch);
        int64_t needed = 0;
        for (int64_t i = logical; i < end; ++i) needed += lease.table[(size_t) i] < 0;
        // Partial batches at the logical tail are charged by their actual
        // number of pages, never rounded up to eight.
        if ((int64_t) free_.size() < needed) {
            free_ = free_before;
            error = "kv_pool: insufficient pages for acquisition batch";
            return false;
        }
        for (int64_t i = logical; i < end; ++i) {
            int32_t& entry = lease.table[(size_t) i];
            if (entry >= 0) continue;
            entry = free_.back();
            free_.pop_back();
            lease.pages.push_back(entry);
        }
        logical = end;
    }
    lease.committed_cells = std::max(lease.committed_cells, committed_cells);
    lease.write_top = std::max(lease.write_top, new_write_cells);
    const int64_t used = pages_ - (int64_t) free_.size();
    high_water_ = std::max(high_water_, used);
    assert((int64_t) free_.size() >= 0 && used >= 0 && used <= pages_);
    return true;
}

bool GlobalKvPool::publish(QsaState& st, const std::vector<int32_t>& table,
                           const std::vector<std::pair<int64_t, int32_t>>& updates,
                           void* stream, std::string& error) {
    for (const auto& [logical, physical] : updates) {
        (void) logical;
        for (size_t part = 0; part < parts_.size(); ++part) {
            auto* address = static_cast<uint8_t*>(parts_[part]) + part_page_bytes_[part] * (uint64_t) physical;
            const bool ok = ops_.zero
                ? ops_.zero(ops_.context, address, part_page_bytes_[part], stream, error)
                : runtime_zero(ops_.context, address, part_page_bytes_[part], stream, error);
            if (!ok) return false;
        }
    }
    if (!updates.empty()) {
        // A pageable CUDA/HIP H2D copy stages synchronously before returning;
        // table remains the authoritative host mirror after this call.
        if (ops_.upload) return ops_.upload(ops_.context, st.page_table, table.data(),
                                            (uint64_t) table.size() * sizeof(int32_t), stream, error);
        return runtime_upload(ops_.context, st.page_table, table.data(),
                              (uint64_t) table.size() * sizeof(int32_t), stream, error);
    }
    return true;
}

bool GlobalKvPool::ensure(QsaState& st, int64_t committed_cells, int64_t new_write_cells,
                          void* caller_stream, std::string& error) {
    error.clear();
    std::lock_guard lock(mutex_);
    auto it = leases_.find(&st);
    const bool had_lease = it != leases_.end();
    if (!had_lease) it = leases_.emplace(&st, Lease{}).first;
    Lease candidate = it->second;
    const std::vector<int32_t> free_before = free_;
    const int64_t high_water_before = high_water_;
    if (!reserve(candidate, st, committed_cells, new_write_cells, error)) {
        // reserve() has already restored free_ and candidate on capacity error.
        // No live table or device entry has been touched.
        free_ = free_before;
        if (!had_lease) leases_.erase(&st);
        return false;
    }
    // Step-5 deterministic failure gate: STRATA_TEST_KV_POOL_FAIL_AFTER=N makes the N-th
    // successful reservation (this one counted) fail BEFORE any publish, then rolls back exactly
    // like a capacity error. Tests use it to prove the caller reserves-then-launches atomically:
    // reservation rollback, no launch, no KV write, free accounting balanced.
    {
        static std::atomic<int64_t> injected_counter{0};
        static const int64_t fail_after = [] {
            const char* v = std::getenv("STRATA_TEST_KV_POOL_FAIL_AFTER");
            return v != nullptr ? std::strtoll(v, nullptr, 10) : (int64_t) 0;
        }();
        if (fail_after > 0 && ++injected_counter == fail_after) {
            free_ = free_before;
            if (!had_lease) leases_.erase(&st);
            error = "kv_pool: injected failure (STRATA_TEST_KV_POOL_FAIL_AFTER)";
            return false;
        }
    }
    std::vector<std::pair<int64_t, int32_t>> updates;
    for (size_t i = 0; i < candidate.table.size(); ++i) {
        if (candidate.table[i] != table_entry(it->second.table, (int64_t) i))
            updates.emplace_back((int64_t) i, candidate.table[i]);
    }
    if (!publish(st, candidate.table, updates, caller_stream, error)) {
        free_ = free_before;
        high_water_ = high_water_before;
        if (!had_lease) leases_.erase(&st);
        return false;
    }
    it->second = std::move(candidate);
    st.kv_pool_committed = it->second.committed_cells;
    st.kv_pool_write_top = it->second.write_top;
    st.kv_pool_page_mirror = &it->second.table;
    return true;
}

bool GlobalKvPool::prepare_prefix(QsaState& st, int64_t image_cells, KvLeaseTxn& txn,
                                  std::string& error) {
    error.clear();
    const bool duplicate = std::any_of(txn.entries.begin(), txn.entries.end(),
                                       [&](const KvLeaseTxn::Entry& e) { return e.state == &st; });
    if (duplicate || image_cells < 0 || image_cells > st.max_cells) {
        error = "kv_pool: duplicate state or invalid image range";
        return false;
    }
    std::lock_guard lock(mutex_);
    Lease candidate;
    const std::vector<int32_t> free_before = free_;
    const int64_t high_water_before = high_water_;
    if (!reserve(candidate, st, 0, image_cells, error)) return false;
    KvLeaseTxn::Entry entry;
    entry.state = &st;
    entry.table = candidate.table;
    entry.pages = candidate.pages;
    entry.committed_cells = image_cells;
    entry.write_top = image_cells;
    try { txn.entries.push_back(std::move(entry)); }
    catch (const std::bad_alloc&) {
        free_ = free_before;
        high_water_ = high_water_before;
        error = "kv_pool: transaction host allocation failed";
        return false;
    }
    txn.prepared = true;
    return true;
}

bool GlobalKvPool::publish_prepared(QsaState& st, KvLeaseTxn& txn, void* caller_stream,
                                    std::string& error) {
    error.clear();
    std::lock_guard lock(mutex_);
    auto entry = std::find_if(txn.entries.begin(), txn.entries.end(),
                              [&](const KvLeaseTxn::Entry& e) { return e.state == &st; });
    if (!txn.prepared || entry == txn.entries.end()) {
        error = "kv_pool: state was not prepared by this transaction";
        return false;
    }
    Lease replacement;
    replacement.table = entry->table;
    replacement.pages = entry->pages;
    replacement.committed_cells = entry->committed_cells;
    replacement.write_top = entry->write_top;
    const auto old = leases_.find(&st);
    std::vector<std::pair<int64_t, int32_t>> updates;
    for (size_t i = 0; i < replacement.table.size(); ++i)
        if (replacement.table[i] >= 0) updates.emplace_back((int64_t) i, replacement.table[i]);
    if (!publish(st, replacement.table, updates, caller_stream, error)) return false;
    if (old != leases_.end()) {
        Lease retired = std::move(old->second);
        leases_.erase(old);
        txn.retired.emplace_back(&st, std::move(retired.pages));
    }
    Lease& live = leases_[&st];
    live = std::move(replacement);
    st.kv_pool_committed = live.committed_cells;
    st.kv_pool_write_top = live.write_top;
    st.kv_pool_page_mirror = &live.table;
    txn.entries.erase(entry);
    if (txn.entries.empty()) txn.prepared = false;
    return true;
}

void GlobalKvPool::finish_prepared(KvLeaseTxn& txn) {
    std::lock_guard lock(mutex_);
    // The caller has synchronized the scatter stream.  Only now may retired
    // physical pages become available to another state.
    for (auto& [state, pages] : txn.retired) { (void) state; recycle(pages); }
    txn.retired.clear();
}

void GlobalKvPool::abort_prepare(KvLeaseTxn& txn) {
    std::lock_guard lock(mutex_);
    for (auto& entry : txn.entries) {
        recycle(entry.pages);
        entry.table.clear();
    }
    txn.entries.clear();
    txn.prepared = false;
    // Published entries cannot be undone.  A caller aborting after publish
    // must first quiesce the device; those retired pages are then reclaimable.
    for (auto& [state, pages] : txn.retired) { (void) state; recycle(pages); }
    txn.retired.clear();
}

bool GlobalKvPool::release(QsaState& st, void* caller_stream, std::string& error) {
    (void) caller_stream;
    error.clear();
    std::lock_guard lock(mutex_);
    const auto it = leases_.find(&st);
    if (it == leases_.end()) { reset(st); return true; }
    Lease retired = std::move(it->second);
    leases_.erase(it);
    recycle(retired.pages);
    reset(st);
    return true;
}

bool GlobalKvPool::gather_logical_bytes(const std::vector<void*>& physical_parts,
                                        const std::vector<uint64_t>& part_page_bytes,
                                        const std::vector<int32_t>& table, int part,
                                        uint64_t logical_offset, void* destination,
                                        uint64_t bytes, std::string& error) {
    return transfer(physical_parts, part_page_bytes, table, part, logical_offset,
                    destination, bytes, false, error);
}

bool GlobalKvPool::scatter_logical_bytes(std::vector<void*>& physical_parts,
                                         const std::vector<uint64_t>& part_page_bytes,
                                         const std::vector<int32_t>& table, int part,
                                         uint64_t logical_offset, const void* source,
                                         uint64_t bytes, std::string& error) {
    return transfer(physical_parts, part_page_bytes, table, part, logical_offset,
                    const_cast<void*>(source), bytes, true, error);
}

bool GlobalKvPool::transfer(const std::vector<void*>& physical_parts,
                            const std::vector<uint64_t>& part_page_bytes,
                            const std::vector<int32_t>& table, int part,
                            uint64_t logical_offset, void* host, uint64_t bytes,
                            bool scatter, std::string& error) {
    error.clear();
    if (part < 0 || (size_t) part >= physical_parts.size() ||
        (size_t) part >= part_page_bytes.size() || part_page_bytes[(size_t) part] == 0 ||
        (!host && bytes != 0)) {
        error = "kv_pool: invalid logical byte transfer";
        return false;
    }
    const uint64_t page_bytes = part_page_bytes[(size_t) part];
    uint64_t done = 0;
    while (done < bytes) {
        const uint64_t at = logical_offset + done;
        if (at < logical_offset) { error = "kv_pool: logical byte offset overflows"; return false; }
        const uint64_t logical = at / page_bytes;
        const uint64_t within = at % page_bytes;
        if (logical >= table.size() || table[(size_t) logical] < 0) {
            error = "kv_pool: logical byte range is not resident";
            return false;
        }
        const uint64_t n = std::min(bytes - done, page_bytes - within);
        auto* physical = static_cast<uint8_t*>(physical_parts[(size_t) part]) +
                         (uint64_t) table[(size_t) logical] * page_bytes + within;
        auto* h = static_cast<uint8_t*>(host) + done;
        if (scatter) std::memcpy(physical, h, (size_t) n);
        else std::memcpy(h, physical, (size_t) n);
        done += n;
    }
    return true;
}

}  // namespace strata::core
