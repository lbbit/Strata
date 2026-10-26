#pragma once

#include "strata/core/geometry.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace strata::core {

struct QsaState;

// The testable storage-price half of layer.cpp's kv_pool_bytes().  These values
// describe one layer's pages; qsa_layers is deliberately not a multiplier.
enum class KvPoolFormat { Fp16, Int8, Q4, Hybrid };

struct KvStateSizingConfig {
    KvPoolFormat format = KvPoolFormat::Fp16;
    int64_t resident_cells = 0;
    bool elastic = false;
    bool pooled = false;
};

uint64_t qsa_state_sizing_bytes(const ModelGeometry& g, int64_t max_cells,
                                bool with_rope, int64_t ring_cells,
                                const KvStateSizingConfig& config);

// The single qsa_state_bytes() sizing implementation, also host-testable.
// pooled removes only K/V; page_table, indexer, RoPE and step are unchanged.

bool kv_pool_page_price(const strata::kernels::QsaShapes& shapes, KvPoolFormat format,
                        uint64_t& bytes, std::string& error);
bool kv_pool_storage_bytes(const strata::kernels::QsaShapes& shapes, int64_t pages,
                           KvPoolFormat format, uint64_t& bytes, std::string& error);

// Optional injection boundary.  Production leaves this empty and uses the CUDA/
// HIP runtime; host tests provide malloc/memcpy operations and never touch a GPU.
struct KvPoolDeviceOps {
    void* context = nullptr;
    void* (*allocate)(void* context, uint64_t bytes, std::string& error) = nullptr;
    void (*release)(void* context, void* pointer) = nullptr;
    bool (*zero)(void* context, void* pointer, uint64_t bytes, void* stream,
                 std::string& error) = nullptr;
    bool (*upload)(void* context, void* destination, const void* source, uint64_t bytes,
                   void* stream, std::string& error) = nullptr;
};

struct KvLeaseTxn {
    struct Entry {
        QsaState* state = nullptr;
        std::vector<int32_t> table;
        std::vector<int32_t> pages;
        int64_t committed_cells = 0;
        int64_t write_top = 0;
    };
    std::vector<Entry> entries;
    // Retired leases remain owned by the pool until finish_prepared(), after
    // the caller's same-stream scatter has completed.
    std::vector<std::pair<QsaState*, std::vector<int32_t>>> retired;
    bool prepared = false;
};

class GlobalKvPool {
public:
    static std::unique_ptr<GlobalKvPool> create(
        int device, const ModelGeometry& g, int64_t qsa_layers,
        int64_t capacity_bytes, std::string& error,
        KvPoolDeviceOps ops = {},
        std::optional<KvPoolFormat> format_override = std::nullopt);
    ~GlobalKvPool();

    GlobalKvPool(const GlobalKvPool&) = delete;
    GlobalKvPool& operator=(const GlobalKvPool&) = delete;

    // A successful ensure enqueues zeroing and its page-table upload on
    // caller_stream. prepare_prefix() is host-only. publish_prepared() uses
    // that same ordering rule: the caller must scatter on caller_stream and
    // synchronize it before finish_prepared(), or before release()/rollback
    // recycles any pages. No device operation may be in flight at release().
    bool ensure(QsaState& st, int64_t committed_cells, int64_t new_write_cells,
                void* caller_stream, std::string& error);
    bool prepare_prefix(QsaState& st, int64_t image_cells, KvLeaseTxn& txn,
                        std::string& error);
    bool publish_prepared(QsaState& st, KvLeaseTxn& txn, void* caller_stream,
                          std::string& error);
    void finish_prepared(KvLeaseTxn& txn);
    void abort_prepare(KvLeaseTxn& txn);
    bool release(QsaState& st, void* caller_stream, std::string& error);

    // Binds addresses only; it deliberately takes no pages.  A mode-0 main
    // QSA state's caller is responsible for not passing a drafter in step 2.
    void bind_main_state(QsaState& st);
    // Step-2 development fixture only: reserve every logical page of one
    // exclusively owned state and publish table[i] = i. Never use this for
    // concurrent sessions; step 3 replaces it with ensure() at write points.
    bool bind_identity_for_test(QsaState& st, void* caller_stream, std::string& error);
    int64_t pages() const { return pages_; }
    int64_t free_pages() const;
    int64_t high_water_pages() const;
    bool owns(const QsaState& st) const;

    // Quiesced-state host mirror access for the later snapshot/copy steps.
    // These return logical page -> physical page, never a device page table.
    bool snapshot_page_table(const QsaState& st, std::vector<int32_t>& table,
                             std::string& error) const;
    int32_t physical_page(const QsaState& st, int64_t logical_page) const;
    uint64_t page_price() const { return page_price_; }
    int64_t page_size() const { return shapes_.page_size; }
    int64_t qsa_layers() const { return qsa_layers_; }

    // Small host-only byte primitives shared by the unit test now and the
    // logical snapshot/copy adapters later.  A pool has one base per part.
    static bool gather_logical_bytes(const std::vector<void*>& physical_parts,
                                     const std::vector<uint64_t>& part_page_bytes,
                                     const std::vector<int32_t>& table, int part,
                                     uint64_t logical_offset, void* destination,
                                     uint64_t bytes, std::string& error);
    static bool scatter_logical_bytes(std::vector<void*>& physical_parts,
                                      const std::vector<uint64_t>& part_page_bytes,
                                      const std::vector<int32_t>& table, int part,
                                      uint64_t logical_offset, const void* source,
                                      uint64_t bytes, std::string& error);

private:
    struct Lease {
        std::vector<int32_t> table;
        std::vector<int32_t> pages;
        int64_t committed_cells = 0;
        int64_t write_top = 0;
    };

    GlobalKvPool(const ModelGeometry& g, int64_t qsa_layers, KvPoolFormat format,
                 int device, KvPoolDeviceOps ops);

    bool reserve(Lease& lease, QsaState& st, int64_t committed_cells,
                 int64_t new_write_cells, std::string& error);
    bool publish(QsaState& st, const std::vector<int32_t>& table,
                 const std::vector<std::pair<int64_t, int32_t>>& updates,
                 void* stream, std::string& error);
    void recycle(std::vector<int32_t>& pages);
    void reset(QsaState& st);
    static bool transfer(const std::vector<void*>& physical_parts,
                         const std::vector<uint64_t>& part_page_bytes,
                         const std::vector<int32_t>& table, int part,
                         uint64_t logical_offset, void* host, uint64_t bytes,
                         bool scatter, std::string& error);

    strata::kernels::QsaShapes shapes_;
    KvPoolFormat format_;
    int64_t qsa_layers_ = 0;
    int device_ = 0;
    KvPoolDeviceOps ops_;
    int64_t pages_ = 0;
    uint64_t page_price_ = 0;
    std::vector<uint64_t> part_page_bytes_;
    std::vector<void*> parts_;
    std::vector<int32_t> free_;
    std::unordered_map<QsaState*, Lease> leases_;
    int64_t high_water_ = 0;
    mutable std::mutex mutex_;
};

}  // namespace strata::core
