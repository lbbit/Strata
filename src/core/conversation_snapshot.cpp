#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "conversation_checked.hpp"
#include "conversation_copy.hpp"
#include "kv_pool.hpp"
#include <cuda_runtime.h>

#include <array>
#include <cstring>
#include <limits>
#include <memory>

namespace strata::core {
namespace {
struct Layout {
    int format;
    int64_t cells, pooled_rows, page_size;
    size_t data, scales, pooled, value_data, value_scales;
};

bool valid_extent(const QsaState& st, int64_t upto, std::string& error) {
    const int64_t page = strata::kernels::qsa_real_shapes().page_size;
    if (upto < 0 || upto > st.max_cells || upto > std::numeric_limits<int64_t>::max() - (page - 1)) {
        error = "conversation snapshot: invalid K/V extent";
        return false;
    }
    return true;
}

bool layout(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index, Layout& l, std::string& error) {
    if (!valid_extent(st, upto, error)) return false;
    if (st.kv_hybrid && (st.kv_mode == 2 || st.kv_q4 || st.kv_int8)) {   // a streamed one reads its host copy
        error = "conversation snapshot: hybrid K8V4 cannot be a ring and needs distinct format flags";
        return false;
    }
    const auto s = strata::kernels::qsa_real_shapes();
    const bool int8_keys = st.kv_int8 || st.kv_hybrid;
    if (g.n_head_kv <= 0 || g.head_dim <= 0 || g.head_dim > INT32_MAX || g.idx_key_dim <= 0 ||
        (st.kv_q4 && g.head_dim % 32) || (int8_keys && !st.kv_q4 && g.head_dim % 64)) {
        error = "conversation snapshot: invalid K/V geometry";
        return false;
    }
    const int64_t cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;
    const size_t per = st.kv_q4 ? (size_t) strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                              : (size_t) g.head_dim * (int8_keys ? 1 : 2);
    // Include the moving spare row, not only completed blocks. The checkpoint
    // restore reconstructs that row when rewinding to an earlier prefix.
    const int64_t pooled = index && upto > 0 ? upto / s.idx_block + 1 : 0;
    // Format 3 identifies snapshot K8V4 only; never pass it to the block movers.  Rotated INT8 K/V (#293,
    // STRATA_KV_ROT=1) is another format too (+16): its bytes mean nothing to a state that does not rotate, so a
    // snapshot or prompt checkpoint taken with the rotation never restores into one without it, nor the reverse.
    // (Q4_0 is always rotated: nothing to tell apart; without the rotation the format is the one it always was.)
    const int rotated = st.kv_rot && !st.kv_q4 && !st.kv_hybrid ? 16 : 0;
    l = {(st.kv_hybrid ? 3 : qsa_kv_format(st)) + rotated, cells, pooled, s.page_size, 0, 0, 0, 0, 0};
    using conversation_detail::product;
    if (!product(l.data, {(uint64_t) cells, (uint64_t) g.n_head_kv, per}) ||
        !product(l.scales, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                           int8_keys && !st.kv_q4 ? (uint64_t) (g.head_dim / 64) * 2 : 0}) ||
        !product(l.pooled, {(uint64_t) pooled, (uint64_t) g.idx_key_dim, sizeof(float)})) {
        error = "conversation snapshot: K/V byte count overflow";
        return false;
    }
    l.value_data = l.data;
    l.value_scales = l.scales;
    if (st.kv_hybrid) {
        if (!product(l.value_data, {(uint64_t) cells, (uint64_t) g.n_head_kv,
                                   (uint64_t) strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)})) {
            error = "conversation snapshot: hybrid V byte count overflow";
            return false;
        }
        l.value_scales = 0;
    }
    return true;
}

std::array<void*, 5> pools(const QsaState& st, bool resident = false) {
    const bool host = st.kv_mode != 0 && !resident;
    if (st.kv_hybrid)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q4 : st.v_q4, host ? st.host.k_scale : st.k_scale,
                nullptr, st.idx_pooled};
    if (st.kv_q4)
        return {host ? st.host.k_q4 : st.k_q4, host ? st.host.v_q4 : st.v_q4, nullptr, nullptr, st.idx_pooled};
    if (st.kv_int8)
        return {host ? st.host.k_q : st.k_q, host ? st.host.v_q : st.v_q,
                host ? st.host.k_scale : st.k_scale, host ? st.host.v_scale : st.v_scale, st.idx_pooled};
    return {host ? st.host.k_pool : st.k_pool, host ? st.host.v_pool : st.v_pool, nullptr, nullptr, st.idx_pooled};
}

bool valid(const QsaState& st, const Layout& l, int64_t upto, std::string& error) {
    if (upto < 0 || upto > st.max_cells || st.n_pages < 0 || l.cells / l.page_size > st.n_pages ||
        st.kv_mode < 0 || st.kv_mode > 2 || st.n_slots <= 0 ||
        (st.kv_mode == 0 && !st.kv_pool && st.n_slots < st.n_pages) ||
        (st.kv_pool && (st.kv_mode != 0 || st.kv_elastic >= 0 || !st.page_table)) ||
        l.pooled_rows > st.idx_pooled_rows || (st.kv_mode != 0 && !st.host.present())) {
        error = "conversation snapshot: invalid K/V extent or missing authoritative host pool";
        return false;
    }
    if (st.kv_mode == 1) {
        const auto& m = st.map;
        if (m.n_blocks != st.n_pages || m.n_slots != st.n_slots || !m.page_table || !m.slot_block ||
            !m.slot_stamp || !m.slot_ref || !m.ctl || !m.miss_block || !m.miss_slot) {
            error = "conversation snapshot: invalid streaming map";
            return false;
        }
    }
    if (st.kv_mode == 2 && (!st.page_table ||
        (st.kv_q4 ? !st.k_q4 || !st.v_q4 : st.kv_int8 ? !st.k_q || !st.v_q || !st.k_scale || !st.v_scale
                                                    : !st.k_pool || !st.v_pool))) {
        error = "conversation snapshot: invalid draft ring";
        return false;
    }
    return true;
}

bool transfer(void* dst, const void* src, size_t n, std::string& error) {
    if (!n) return true;
    if (!src || !dst) { error = "conversation snapshot: missing state buffer"; return false; }
    // Default handles both device allocations and device-mapped host pool aliases.
    const cudaError_t e = conversation_detail::copy_nonblocking(dst, src, n);
    if (e == cudaSuccess) return true;
    error = std::string("conversation snapshot copy: ") + cudaGetErrorString(e);
    return false;
}

struct PageView {
    QsaState* mutable_state = nullptr;
    GlobalKvPool* pool = nullptr;
    const QsaState* state = nullptr;
    std::shared_ptr<const std::vector<int32_t>> owned_table;
    const std::vector<int32_t>* table = nullptr;
    std::array<size_t, 5> page_bytes{};
    std::array<const std::vector<int32_t>*, 5> part_tables{};
};

bool page_view(PageView& view, const QsaState& st, const Layout& l, int64_t upto,
               std::string& error, const std::vector<int32_t>* prepared = nullptr) {
    view.pool = st.kv_pool;
    view.mutable_state = const_cast<QsaState*>(&st);
    view.state = &st;
    const size_t cells = (size_t) l.cells;
    const std::array<size_t, 5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    // copy_logical_range() divides a logical byte offset by page_bytes and scales the physical page number by it,
    // so page_bytes must be the size of ONE PHYSICAL PAGE - the page table maps granules, not cells. sizes[i] is the
    // whole-part byte count over `cells` cells, so per-cell bytes are sizes[i]/cells and one page is that times the
    // page granule (l.page_size cells). (Passing per-cell bytes made every pooled page straddle four physical pages.)
    for (size_t i = 0; i < 4; ++i) view.page_bytes[i] = cells ? (sizes[i] / cells) * (size_t) l.page_size : 0;
    if (!st.kv_pool) return true;
    if (st.kv_mode != 0 || st.kv_elastic >= 0 || !st.page_table ||
        l.cells / l.page_size > st.n_pages) {
        error = "conversation snapshot: invalid pooled QSA state";
        return false;
    }
    if (prepared) {
        // publish_prepared() consumes and erases the transaction entry. Own the
        // table so scatter never retains a pointer into that erased entry.
        view.owned_table = std::make_shared<std::vector<int32_t>>(*prepared);
        view.table = view.owned_table.get();
        for (size_t i = 0; i < view.part_tables.size(); ++i) view.part_tables[i] = view.table;
    } else if (l.cells) {
        auto table = std::make_shared<std::vector<int32_t>>();
        if (!st.kv_pool->snapshot_page_table(st, *table, error)) return false;
        view.owned_table = std::move(table);
        view.table = view.owned_table.get();
        for (size_t i = 0; i < view.part_tables.size(); ++i) view.part_tables[i] = view.table;
    } else {
        return true;   // an empty snapshot needs no lease
    }
    if (!view.table || view.table->size() != (size_t) st.n_pages) {
        error = "conversation snapshot: invalid pooled page-table mirror";
        return false;
    }
    const int64_t pages = (upto + l.page_size - 1) / l.page_size;
    for (int64_t i = 0; i < pages; ++i) {
        const int32_t physical = (*view.table)[(size_t) i];
        if (physical < 0 || physical >= st.kv_pool->pages()) {
            error = "conversation snapshot: pooled logical page is not resident";
            return false;
        }
    }
    return true;
}

bool copy_pages(const PageView& view, const std::array<void*, 5>& flat, size_t part,
                size_t offset, void* host, size_t bytes, bool scatter,
                std::string& error, bool queued = false) {
    if (!bytes) return true;
    const QsaState& st = *view.state;
    if (!st.kv_pool || part == 4) {
        void* device = static_cast<uint8_t*>(flat[part]) + offset;
        return transfer(scatter ? device : host, scatter ? host : device, bytes, error);
    }
    if (!flat[part] || !host || !view.page_bytes[part] || !view.table) {
        error = "conversation snapshot: missing pooled state buffer or page table";
        return false;
    }
    const cudaError_t e = conversation_detail::copy_logical_range(
        flat[part], view.page_bytes[part], view.part_tables[part], offset, host, bytes, scatter, queued);
    if (e == cudaSuccess) return true;
    error = std::string("conversation snapshot logical-page copy: ") + cudaGetErrorString(e) +
            " (part=" + std::to_string(part) + ", logical_offset=" + std::to_string(offset) +
            ", bytes=" + std::to_string(bytes) + ", page_bytes=" + std::to_string(view.page_bytes[part]) + ")";
    return false;
}

} // namespace

bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                   int64_t upto, bool index, size_t& bytes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> buffers = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    bytes = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const size_t n = buffers[i]->allocation_peak(sizes[i]);
        if (n == SIZE_MAX || !conversation_detail::add(bytes, n)) {
            error = "conversation snapshot: segmented K/V allocation overflow";
            return false;
        }
    }
    return true;
}

size_t conversation_kv_bytes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index) {
    size_t bytes = 0;
    std::string error;
    return conversation_kv_capture_bytes({}, st, g, upto, index, bytes, error) ? bytes : 0;
}

bool conversation_kv_save(ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error,
                          int64_t unchanged_tokens, size_t* reused_bytes) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    if (unchanged_tokens < 0 || unchanged_tokens > upto || unchanged_tokens > image.cells) {
        error = "conversation snapshot: invalid unchanged prefix";
        return false;
    }
    const int64_t whole_cells = (unchanged_tokens / l.page_size) * l.page_size;
    image.format = l.format;
    image.cells = l.cells;
    image.heads = g.n_head_kv;
    image.head_dim = g.head_dim;
    image.page_size = l.page_size;
    image.pooled_rows = l.pooled_rows;
    image.idx_dim = g.idx_key_dim;
    const auto src = pools(st);
    PageView view;
    if (!page_view(view, st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<ConversationBuffer*,5> dst = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    for (size_t i = 0; i < dst.size(); ++i) {
        // Recopy the partial page and the indexer's moving spare row. Completed
        // pages/rows strictly before the first rewritten token remain identical.
        const size_t keep = i == 4 ? (index ? size_t(unchanged_tokens / strata::kernels::qsa_real_shapes().idx_block) * g.idx_key_dim * 4 : 0)
                                  : l.cells ? (sizes[i] / size_t(l.cells)) * size_t(whole_cells) : 0;
        if (keep > dst[i]->size()) { error = "conversation snapshot: missing reusable prefix"; return false; }
        dst[i]->resize(sizes[i]);
        if (!dst[i]->visit(keep, sizes[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
                if (st.kv_pool && i < 4)
                    return copy_pages(view, src, i, at, p, n, false, error);
                return transfer(p, src[i] ? static_cast<const uint8_t*>(src[i]) + at : nullptr, n, error);
            })) return false;
        if (reused_bytes) *reused_bytes += keep;
    }
    return true;
}

bool conversation_kv_validate(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                              int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    if (!valid(st, l, upto, error)) return false;
    const std::array<size_t,5> sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const std::array<const ConversationBuffer*,5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    if (image.format != l.format || image.cells != l.cells || image.heads != g.n_head_kv ||
        image.head_dim != g.head_dim || image.page_size != l.page_size || image.pooled_rows != l.pooled_rows ||
        image.idx_dim != g.idx_key_dim) {
        error = "conversation snapshot: incompatible K/V geometry";
        return false;
    }
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i]->size() != sizes[i]) { error = "conversation snapshot: invalid K/V payload size"; return false; }
    const auto dst = pools(st);
    for (size_t i = 0; i < src.size(); ++i)
        if (sizes[i] && !dst[i]) { error = "conversation snapshot: missing target state buffer"; return false; }
    return true;
}

namespace {
// After the authoritative pools were rewritten: the VRAM slots of a streamed state still hold the outgoing
// conversation, so resolve must refill them before any attention reads (shared by restore and direct copy).
bool refresh_residency(const QsaState& st, const ModelGeometry& g, int64_t upto, std::string& error) {
    if (st.kv_mode == 1) strata::kernels::kv_stream_reset(st.map, nullptr);
    if (st.kv_mode == 2 && upto > 0) {
        auto shapes = strata::kernels::qsa_real_shapes();
        shapes.n_head_kv = g.n_head_kv; shapes.head_dim = g.head_dim;
        const int64_t end = (upto + shapes.page_size - 1) / shapes.page_size;
        strata::kernels::kv_ring_restore(qsa_attn_pools(st), st.host, qsa_kv_format(st),
                                        std::max<int64_t>(0, end - st.n_slots), end, st.n_slots, shapes, nullptr);
    }
    const auto status = cudaGetLastError();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot residency restore: ") + cudaGetErrorString(status);
    return false;
}
} // namespace

bool conversation_kv_copy(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                          int64_t upto, bool index, std::string& error, bool queued) {
    Layout lf{}, lt{};
    if (!layout(from, g, upto, index, lf, error) || !valid(from, lf, upto, error) ||
        !layout(to, g, upto, index, lt, error) || !valid(to, lt, upto, error)) return false;
    if (lf.format != lt.format || lf.cells != lt.cells || lf.page_size != lt.page_size ||
        lf.pooled_rows != lt.pooled_rows || lf.data != lt.data || lf.value_data != lt.value_data ||
        lf.scales != lt.scales || lf.value_scales != lt.value_scales || lf.pooled != lt.pooled) {
        error = "conversation snapshot: direct K/V copy between different layouts";
        return false;
    }
    // The same bytes conversation_kv_save + conversation_kv_restore move, without the host image: each part
    // straight from the source's authoritative pool into the target's (device or mapped host memory), through the
    // private non-blocking stream the other snapshot copies use (complete on return, before the residency refresh).
    // `queued`: the copies are only queued, and a streamed state's refresh waits for them on the device (an event).
    const auto src = pools(from);
    const auto dst = pools(to);
    const std::array<size_t,5> sizes = {lf.data, lf.value_data, lf.scales, lf.value_scales, lf.pooled};
    PageView source, target;
    KvLeaseTxn txn;
    bool replacement = false;
    auto abort = [&] {
        (void) conversation_detail::synchronize_private_stream();
        if (replacement) to.kv_pool->abort_prepare(txn);
    };
    if (!page_view(source, from, lf, upto, error)) return false;
    if (to.kv_pool && lf.cells) {
        QsaState& mutable_to = const_cast<QsaState&>(to);
        if (!to.kv_pool->prepare_prefix(mutable_to, lf.cells, txn, error)) return false;
        replacement = true;
        auto entry = std::find_if(txn.entries.begin(), txn.entries.end(),
                                  [&](const auto& e) { return e.state == &mutable_to; });
        if (entry == txn.entries.end()) { error = "conversation snapshot: missing prepared target"; abort(); return false; }
        cudaStream_t copy_stream = nullptr;
        if (const cudaError_t e = conversation_detail::private_stream(copy_stream); e != cudaSuccess) {
            error = std::string("conversation snapshot direct copy: ") + cudaGetErrorString(e); abort(); return false;
        }
        // publish_prepared() erases the entry; scatter must own its page table.
        target.owned_table = std::make_shared<std::vector<int32_t>>(entry->table);
        if (!page_view(target, to, lt, upto, error, target.owned_table.get()) ||
            !to.kv_pool->publish_prepared(mutable_to, txn, copy_stream, error)) { abort(); return false; }
        // Same ordering as restore: land the publish zero/upload before the scatter reads the table / writes pages.
        if (const cudaError_t _se = cudaDeviceSynchronize(); _se != cudaSuccess) {
            error = std::string("conversation snapshot direct copy: ") + cudaGetErrorString(_se); abort(); return false;
        }
    } else if (!page_view(target, to, lt, upto, error)) return false;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (!sizes[i]) continue;
        if (!src[i] || !dst[i]) { error = "conversation snapshot: missing state buffer"; abort(); return false; }
        cudaError_t e = cudaSuccess;
        if (from.kv_pool && i < 4) {
            auto* logical = static_cast<uint8_t*>(src[i]);
            e = conversation_detail::copy_logical_range(dst[i], target.page_bytes[i], target.part_tables[i],
                                                       0, logical, sizes[i], true, queued,
                                                       source.part_tables[i]);
        } else {
            e = conversation_detail::copy_nonblocking(dst[i], src[i], sizes[i], queued);
        }
        if (e != cudaSuccess) { error = std::string("conversation snapshot direct copy: ") + cudaGetErrorString(e); abort(); return false; }
    }
    if (replacement) {
        if (const cudaError_t e = conversation_detail::synchronize_private_stream(); e != cudaSuccess) {
            error = std::string("conversation snapshot direct copy: ") + cudaGetErrorString(e); abort(); return false;
        }
        to.kv_pool->finish_prepared(txn);
    }
    if (queued && to.kv_mode != 0) {
        const cudaError_t e = conversation_detail::order_after_queued();
        if (e != cudaSuccess) { error = std::string("conversation snapshot direct copy: ") + cudaGetErrorString(e); return false; }
    }
    return refresh_residency(to, g, upto, error);
}

bool conversation_kv_restore(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                             int64_t upto, bool index, std::string& error) {
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    return conversation_kv_restore_many({&image}, {const_cast<QsaState*>(&st)}, g, upto, upto, index, error);
}

bool conversation_kv_restore_prefix(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                                    int64_t captured, int64_t upto, bool index, std::string& error) {
    if (upto < 0 || upto > captured) {
        error = "conversation snapshot: prefix beyond the captured K/V";
        return false;
    }
    if (!conversation_kv_validate(image, st, g, captured, index, error)) return false;
    return conversation_kv_restore_many({&image}, {const_cast<QsaState*>(&st)}, g, captured, upto, index, error);
}

bool conversation_kv_restore_many(const std::vector<const ConversationKv*>& images,
                                  const std::vector<QsaState*>& states, const ModelGeometry& g,
                                  int64_t captured, int64_t upto, bool index, std::string& error) {
    if (images.size() != states.size() || upto < 0 || upto > captured) {
        error = "conversation snapshot: invalid multi-layer restore";
        return false;
    }
    std::vector<Layout> layouts(images.size());
    std::vector<PageView> views(images.size());
    std::vector<std::array<void*, 5>> destinations(images.size());
    std::vector<std::array<size_t, 5>> sizes(images.size());
    KvLeaseTxn txn;
    bool replacement = false;
    auto abort = [&] {
        (void) conversation_detail::synchronize_private_stream();
        if (replacement) states.front()->kv_pool->abort_prepare(txn);
    };
    // Validate and prepare everything before publishing any layer. All main
    // layers of one stage share a pool; a non-pooled draft is never in `states`.
    for (size_t j = 0; j < images.size(); ++j) {
        const auto& image = *images[j];
        QsaState& st = *states[j];
        if (!conversation_kv_validate(image, st, g, captured, index, error) ||
            !layout(st, g, upto, index, layouts[j], error) ||
            !valid(st, layouts[j], upto, error)) return false;
        const Layout& l = layouts[j];
        sizes[j] = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
        const std::array<const ConversationBuffer*, 5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
        for (size_t i = 0; i < src.size(); ++i)
            if (sizes[j][i] > src[i]->size()) { error = "conversation snapshot: prefix beyond the K/V payload"; return false; }
        destinations[j] = pools(st);
        if (st.kv_pool && l.cells) {
            if (j == 0) replacement = true;
            if (!states.front()->kv_pool || states.front()->kv_pool != st.kv_pool ||
                !st.kv_pool->prepare_prefix(st, l.cells, txn, error)) { abort(); return false; }
            auto entry = std::find_if(txn.entries.begin(), txn.entries.end(),
                                      [&](const auto& e) { return e.state == &st; });
            if (entry == txn.entries.end()) { abort(); return false; }
            // Publishing erases this entry; keep the prepared mapping alive.
            views[j].owned_table = std::make_shared<std::vector<int32_t>>(entry->table);
            if (!page_view(views[j], st, l, upto, error, views[j].owned_table.get())) {
                abort(); return false;
            }
        } else if (!page_view(views[j], st, l, upto, error)) { abort(); return false; }
    }
    cudaStream_t copy_stream = nullptr;
    if (replacement) {
        if (const cudaError_t e = conversation_detail::private_stream(copy_stream); e != cudaSuccess) {
            error = std::string("conversation snapshot restore: ") + cudaGetErrorString(e); abort(); return false;
        }
    }
    for (size_t j = 0; j < states.size(); ++j)
        if (states[j]->kv_pool && layouts[j].cells &&
            !states[j]->kv_pool->publish_prepared(*states[j], txn, copy_stream, error)) { abort(); return false; }
    // The page table publish (publish_prepared -> publish: an async zero of every newly acquired physical page and an
    // async H2D upload of the mirrored page table) and this scatter use independent streams of the pool; a scatter
    // that reads a page table or writes a page mid-zero is not ordered against them, so a handful of pages can come
    // back stale. Restore is a checkpoint-load path, never hot, so quiesce the device once here before scattering:
    // every publish zero/upload has landed and the scatter sees the final table and clean pages.
    if (replacement) {
        const cudaError_t _se = cudaDeviceSynchronize();
        if (_se != cudaSuccess) { error = std::string("restore pre-scatter sync: ") + cudaGetErrorString(_se); abort(); return false; }
    }
    for (size_t j = 0; j < states.size(); ++j) {
        const auto& image = *images[j];
        const std::array<const ConversationBuffer*, 5> src = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
        for (size_t i = 0; i < src.size(); ++i)
            if (!src[i]->visit(0, sizes[j][i], [&](const uint8_t* p, size_t n, size_t at) {
                    if (states[j]->kv_pool && i < 4)
                        return copy_pages(views[j], destinations[j], i, at,
                                          const_cast<uint8_t*>(p), n, true, error);
                    return transfer(static_cast<uint8_t*>(destinations[j][i]) + at, p, n, error);
                })) { abort(); return false; }
    }
    if (replacement) {
        if (const cudaError_t e = conversation_detail::synchronize_private_stream(); e != cudaSuccess) {
            error = std::string("conversation snapshot restore: ") + cudaGetErrorString(e); abort(); return false;
        }
        states.front()->kv_pool->finish_prepared(txn);
    }
    for (size_t j = 0; j < states.size(); ++j)
        if (!refresh_residency(*states[j], g, upto, error)) return false;
    return true;
}

bool conversation_kv_part_sizes(const QsaState& st, const ModelGeometry& g, int64_t upto, bool index,
                                std::array<uint64_t, 5>& sizes, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    return true;
}

bool conversation_kv_source(SessionKvSource& out, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, std::string& error) {
    Layout l{};
    if (!layout(st, g, upto, index, l, error) || !valid(st, l, upto, error)) return false;
    SessionKvSource s;
    s.format = l.format; s.cells = l.cells; s.heads = g.n_head_kv; s.head_dim = g.head_dim;
    s.page_size = l.page_size; s.pooled_rows = l.pooled_rows; s.idx_dim = g.idx_key_dim;
    s.sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};
    const auto src = pools(st);
    const auto sizes = s.sizes;
    auto view = std::make_shared<PageView>();
    if (!page_view(*view, st, l, upto, error)) return false;
    auto why = std::make_shared<std::string>();
    s.read = [src, sizes, why, view](size_t part, size_t offset, void* dst, size_t n) {
        if (part >= 5 || offset > sizes[part] || n > sizes[part] - offset) { *why = "K/V read out of range"; return false; }
        if (view->pool && part < 4) return copy_pages(*view, src, part, offset, dst, n, false, *why);
        return transfer(dst, src[part] ? static_cast<const uint8_t*>(src[part]) + offset : nullptr, n, *why);
    };
    s.error = [why] { return *why; };
    out = std::move(s);
    return true;
}

bool conversation_kv_verify(const ConversationKv& image, const QsaState& st, const ModelGeometry& g,
                            int64_t upto, bool index, uint64_t& fingerprint, std::string& error) {
    if (!conversation_kv_validate(image, st, g, upto, index, error)) return false;
    std::array<uint8_t, 65536> buffer;
    uint64_t hash = 1469598103934665603ull;
    const auto authoritative = pools(st);
    Layout l{};
    if (!layout(st, g, upto, index, l, error)) return false;
    PageView view;
    if (!page_view(view, st, l, upto, error)) return false;
    auto compare = [&](size_t part, const uint8_t* expected, size_t logical_offset,
                       const void* device, size_t bytes, bool include_hash) {
        for (size_t offset = 0; offset < bytes;) {
            const size_t n = std::min(buffer.size(), bytes - offset);
            if (st.kv_pool && part < 4) {
                if (!copy_pages(view, authoritative, part, logical_offset + offset,
                                buffer.data(), n, false, error)) return false;
            } else if (!transfer(buffer.data(), static_cast<const uint8_t*>(device) + offset, n, error)) return false;
            if (std::memcmp(buffer.data(), expected + offset, n) != 0) {
                error = "conversation snapshot: restored K/V bytes differ";
                return false;
            }
            if (include_hash) for (size_t i = 0; i < n; ++i) { hash ^= buffer[i]; hash *= 1099511628211ull; }
            offset += n;
        }
        return true;
    };
    const std::array<const ConversationBuffer*, 5> saved = {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
    for (size_t i = 0; i < saved.size(); ++i)
        if (!saved[i]->visit(0, saved[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
                return compare(i, p, at, static_cast<const uint8_t*>(authoritative[i]) + at, n, true);
            })) return false;
    if (st.kv_mode == 2 && image.cells > 0) {
        const auto resident = pools(st, true);
        const int64_t end = image.cells / image.page_size;
        const int64_t begin = std::max<int64_t>(0, end - st.n_slots);
        for (size_t i = 0; i < 4; ++i) {
            if (saved[i]->empty()) continue;
            const size_t page_bytes = saved[i]->size() / size_t(end);
            for (int64_t page = begin; page < end; ++page)
                if (!saved[i]->visit(size_t(page) * page_bytes, page_bytes,
                        [&](const uint8_t* p, size_t n, size_t at) {
                            const size_t offset = size_t(page % st.n_slots) * page_bytes + at - size_t(page) * page_bytes;
                            return compare(i, p, at, static_cast<const uint8_t*>(resident[i]) + offset, n, false);
                        })) return false;
        }
    }
    fingerprint = hash;
    return true;
}

} // namespace strata::core
