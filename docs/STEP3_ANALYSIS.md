我会先按要求完整重读文档指定章节，再核对现有分支与步骤 2 实现；随后最小接入、编译，并用真实模型做 greedy 逐 token 对照。

I’ll inspect the requested design sections and C++ call paths, then report insertion points with exact line references and signatures. I won’t modify files.I’ll inspect the referenced documentation and trace pool allocation, expert-cache setup, workspace reservations, stage state, and validation in `generate.cpp`. I’ll only report recommendations; no files will be changed.The repository already contains step‑1 allocator code and partial step‑2 state sizing/binding. I’m tracing the live startup sequence now, especially the gap between session allocation and expert-cache opening, to avoid recommending pool creation at a point where later workspace allocations could fail.The launch paths are close to the plan’s line references. I’m checking stage-local QSA indexing and the capture helpers now; these determine where an `ensure` loop can safely go and what position it must reserve for repeated batch-MTP rows.The important position detail is that solo/pipeline reserve the whole speculative window, while a batch-MTP slot should be ensured once per contiguous slot group—not once per row—so its committed prefix stays at the first row and rejected tail rows remain write-top only. `stage_batch()` is the common pre-capture/pre-launch point for both batch APIs.A key ordering constraint is visible: pooled sizing must be enabled before each `session_bytes()`/`session_init()`, but the prefill and verifier workspaces are currently initialized only after the expert caches. Therefore, simply creating pools after all sessions would bind too late and misprice the arenas. I’m separating the required startup insertion points from the smaller launch-time changes needed for step 3.Static inspection of `/home/lbb/code/Strata_vllm` at `e0a4cc46`; working tree was clean. No edits or runtime tests.

The pool/state APIs already exist, but **none of the requested prefill/decode launch paths currently calls `GlobalKvPool::ensure()`**.

## 1. Current layer/pool APIs

`src/core/kv_pool.hpp:87`:

```cpp
bool ensure(QsaState& st, int64_t committed_cells, int64_t new_write_cells,
            void* caller_stream, std::string& error);
```

`ensure()` is implemented at `src/core/kv_pool.cpp:531`. It reserves through `ceil(new_write_cells / page_size)`, zeroes newly acquired pages, and uploads the page table on `caller_stream`. The write interval is therefore effectively `[committed_cells, new_write_cells)`, with the endpoint used for coverage.

Relevant current APIs:

```cpp
// include/strata/core/layer.hpp:293
void qsa_set_kv_pool(GlobalKvPool* pool);

// include/strata/core/layer.hpp:340
uint64_t qsa_state_init(const ModelGeometry& g, int64_t max_cells, void* base,
                        QsaState& st, const QsaState* share_rope = nullptr,
                        int64_t ring_cells = 0);

// include/strata/core/layer.hpp:343
strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState& st);
```

`QsaState` already has the required fields at `layer.hpp:242-248`:

```cpp
GlobalKvPool* kv_pool = nullptr;
int64_t kv_pool_committed = 0;
int64_t kv_pool_write_top = 0;
std::vector<int32_t>* kv_pool_page_mirror = nullptr;
```

The fixed device pointers are `st.page_table` and the K/V bases. `qsa_attn_pools()` at `src/core/layer.cpp:1004` only packages those pointers; it does not allocate:

```cpp
strata::kernels::QsaAttnPools pools;
pools.page_table = st.page_table;
// ... fixed K/V bases ...
return pools;
```

For iterating every QSA state in one stage/session, use:

```cpp
for (int64_t i = 0; i < ss.qsa_alloc; ++i) {
    core::QsaState& st = ss.qsa_states[ss.qsa_ord0 + i];
    if (st.kv_pool == nullptr) continue;
    // st.kv_pool->ensure(st, committed, write, stream, err)
}
```

Do **not** iterate `0..g.n_qsa_layers()` on a split stage. `SessionState` documents global ordinal indexing at `include/strata/core/session.hpp:53-62`.

---

## 2. Prefill

### Current signatures and call graph

```cpp
// include/strata/prefill/prefill.hpp:174
bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

// include/strata/prefill/prefill.hpp:262
bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0,
              std::string& err);
```

Call graph:

```text
generate.cpp read_part lambda
  -> sp.run(ids.data() + q, r - q, q, e)       generate.cpp:13480-13484
  -> Prefill::run()                            prefill.cpp:5131
     -> Prefill::run_impl()                    prefill.cpp:5142
        -> next_->run_impl(tokens+c0,T,p0,...) prefill.cpp:4931/4938
        -> QSA append/attention                prefill.cpp:3284 onward
```

### Exact current chunk anchor

`src/prefill/prefill.cpp:2719-2722`:

```cpp
for (int64_t c0 = 0; c0 < n; c0 += C) {
    // ...
    const int64_t T = std::min(C, n - c0), p0 = pos0 + c0;
    core::progress_at("reading the prompt (batched): preparing the chunk from token", p0);
```

### Minimal insertion

Insert **immediately after `T`/`p0` are computed and before `progress_at`/embedding uploads**, inside every `run_impl()` chunk:

```cpp
const int64_t T = std::min(C, n - c0), p0 = pos0 + c0;

for (int64_t i = 0; i < ss.qsa_alloc; ++i) {
    core::QsaState& st = ss.qsa_states[ss.qsa_ord0 + i];
    if (st.kv_pool != nullptr &&
        !st.kv_pool->ensure(st, p0, p0 + T, m.cs, err))
        return false;
}

core::progress_at(/* existing call */);
```

This automatically covers later split stages because each receives its own `run_impl(tokens+c0,T,p0,...)`. `Prefill::run()` needs no additional whole-prompt reservation, and no ensure belongs around the initial short/forced chunk only.

Semantics: token row `t` writes logical cell `p0+t`; reserve committed=`p0`, write-top=`p0+T`.

---

## 3. Solo decode: `run()` and `prelaunch()`

### Current signatures

```cpp
// include/strata/core/verify.hpp:105
bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool,
         void* user, int32_t* out, std::string& err);

// include/strata/core/verify.hpp:497, private
bool prelaunch(int T, const int32_t* tokens, int64_t pos0,
               std::string& err);
```

Main serve caller: `generate.cpp:15089`; other direct callers are at `13219` and `16609`.

### Current call graph

```text
Verifier::run()                                      verify.cpp:2865
  -> capture(T), capture_commit()                    2887
  -> next_->capture(), next_->capture_commit()       2888-2893
  -> stage_inputs()                                  2897
  -> cudaGraphLaunch/HostGraphLaunch                 2947-2953
  -> helper thread: next_->prelaunch()               2977-2981
       -> stage_inputs()                             2847
       -> cudaGraphLaunch()                          2857
  -> next_->run() after this stage synchronizes      3163
```

### Minimal insertion in `prelaunch()`

`verify.cpp:2845-2847` currently reads:

```cpp
if (pos0 + T > ss_->qsa_states[ss_->qsa_primary()].max_cells) {
    err = "verify: the window runs past the context"; return false;
}
const Clock::time_point t0 = Clock::now();
stage_inputs(T, tokens, pos0);
```

Insert the main-state loop between the bounds check and `stage_inputs()`:

```cpp
for (int64_t i = 0; i < ss_->qsa_alloc; ++i) {
    QsaState& st = ss_->qsa_states[ss_->qsa_ord0 + i];
    if (st.kv_pool != nullptr &&
        !st.kv_pool->ensure(st, pos0, pos0 + T, cs_, err))
        return false;
}
```

This placement is mandatory: it precedes both input staging and the real graph launch.

### Minimal insertion in `run()`

At `verify.cpp:2881-2893`, after `pre` is known:

```cpp
const bool pre = prelaunched_ == T;
prelaunched_ = 0;
```

Ensure this stage only when `!pre`; a prelaunched graph was already covered by `prelaunch()`:

```cpp
if (!pre) {
    for (int64_t i = 0; i < ss.qsa_alloc; ++i) {
        QsaState& st = ss.qsa_states[ss.qsa_ord0 + i];
        if (st.kv_pool != nullptr &&
            !st.kv_pool->ensure(st, pos0, pos0 + T, cs_, err))
            return false;
    }
}
```

Place it before the first current-stage launch; putting it before `capture(T,err)` also keeps all allocation outside capture.

Inside the existing `if (chain)` block, ensure `*next_->ss_` on `next_->cs_` before the helper can launch:

```cpp
if (chain) {
    const OnDevice on_next(next_->device_);
    next_->refresh_ar();

    for (int64_t i = 0; i < next_->ss_->qsa_alloc; ++i) {
        QsaState& st = next_->ss_->qsa_states[next_->ss_->qsa_ord0 + i];
        if (st.kv_pool != nullptr &&
            !st.kv_pool->ensure(st, pos0, pos0 + T, next_->cs_, err))
            return false;
    }

    if (!next_->capture(T, err) || !next_->capture_commit(err)) return false;
}
```

The repeated ensure in `prelaunch()` is harmless and normally allocates nothing; it also protects `prelaunch()` as an independent entry point.

Semantics: reserve the **entire** `[pos0,pos0+T)` window, not merely the subsequently accepted prefix. Rejected MTP tail pages remain allocated/write-top; `commit()` does not reduce them.

---

## 4. Batch windows

### Current signatures and call graph

```cpp
// verify.hpp:218
bool run_slot_rows(const int* rows, int S, const int32_t* tokens,
                   const int64_t* pos, PoolMultiFn pool, void* user,
                   int32_t* out, std::string& err);

// verify.hpp:445, private
bool stage_batch(const int* rows, int S, int hbase,
                 const int32_t* tokens, const int64_t* pos,
                 std::string& err);

// verify.hpp:233
bool batch_launch(int base, int S, const int32_t* tokens,
                  const int64_t* pos, std::string& err);
```

```text
generate.cpp::batch_step()
  -> ver.run_slot_rows(...)                    generate.cpp:11583
     -> pre_launch_                            verify.cpp:3784
     -> stage_batch(rows,S,0,...)              verify.cpp:3785
        -> capture_batch()
        -> capture_commit_batch()
     -> cudaGraphLaunch()                      verify.cpp:3786
     -> next_->run_slot_rows()                 verify.cpp:3811/3875
     -> commit_slots()/commit_slot_prefixes()

generate.cpp --batch-groups pump()
  -> construct rows/tokens/positions           generate.cpp:11875-11886
  -> stage_verifier(k).batch_launch(...)       generate.cpp:11900
     -> rows[t] = base+t                       verify.cpp:3950-3951
     -> stage_batch(rows,S,base,...)           verify.cpp:3953
     -> cudaGraphLaunch(window)                verify.cpp:3954
     -> cudaGraphLaunch(commit)                verify.cpp:3955-3956
```

### Single minimal insertion point: `stage_batch()`

`stage_batch()` already receives every row from both APIs and runs **before capture, staging, and launch**. Insert after its existing per-row context validation and before `refresh_ar()` at `verify.cpp:3714`.

For correct batch-MTP commit semantics, ensure once per contiguous slot group:

```cpp
for (int first = 0; first < S;) {
    int last = first + 1;
    while (last < S && rows[last] == rows[first]) ++last;

    SessionState& sx = *slots_[(size_t) rows[first]];
    const int64_t committed = pos[first];
    const int64_t write = pos[last - 1] + 1;
    for (int64_t i = 0; i < sx.qsa_alloc; ++i) {
        QsaState& st = sx.qsa_states[sx.qsa_ord0 + i];
        if (st.kv_pool != nullptr &&
            !st.kv_pool->ensure(st, committed, write, cs_, err))
            return false;
    }
    first = last;
}

refresh_ar();  // existing line
```

No additional ensure is needed in `run_slot_rows()`, `batch_launch()`, or `generate.cpp::batch_step()`; that would duplicate the same reservation on every stage.

Row semantics:

- `rows[t]` is the slot ID; `pos[t]` is that row’s absolute logical cell position.
- `hbase`/`base` selects hand-off rows. It is **not** a KV position.
- Plain batch: one row per slot, so committed=`pos[t]`, write=`pos[t]+1`.
- Batch MTP: a slot owns consecutive rows with consecutive positions (`batch_rows.hpp`). Committed is the group’s first position; write-top is the last position plus one. This avoids advancing committed once per rejected draft row.

### Inactive pipeline pad rows

At `generate.cpp:11881-11886`:

```cpp
G.tok[t] = sl.active ? sl.x : 0;
G.pos[t] = sl.active ? sl.p : 0;
if (!sl.active) sl.cached = false;   // its pad row writes its state
```

Therefore inactive slots before the last active slot really do write position 0 and must receive a valid lease through `stage_batch()`.

However, **`sl.cached=false` currently does not release/invalidate the old pool lease**. Ensure insertion alone does not complete the required lifecycle: before such a pad row launches, the old device-cache live state and old lease must be invalidated/released at a quiesced boundary, otherwise a retained stale lease could later be reused. This remains a step-5 lifecycle obligation.

---

## 5. Pipeline windows: `pl_launch()`

### Current signature/call graph

```cpp
// verify.hpp:294
bool pl_launch(int T, const int32_t* tokens, int64_t pos0,
               std::string& err);
```

```text
generate.cpp callers:
  prompt pipeline:       13143, 13162
  speculative pipeline:  14661, 14691, 14710

Verifier::pl_launch()                       verify.cpp:4228
  -> optional pl_stage()/stage_inputs()     4248
  -> cudaGraphLaunch(exec_[T],cs_)          4258
  -> pl_commit_async() by caller
```

`prestage()` only calls `pl_stage()`; it does not launch. Ensure belongs in `pl_launch()`, not `prestage()`.

Insert after context/capture validation and before `pre_launch_`, or at latest before `cudaGraphLaunch()`:

```cpp
for (int64_t i = 0; i < ss.qsa_alloc; ++i) {
    QsaState& st = ss.qsa_states[ss.qsa_ord0 + i];
    if (st.kv_pool != nullptr &&
        !st.kv_pool->ensure(st, pos0, pos0 + T, cs_, err))
        return false;
}
```

The best anchor is immediately after the `"pipelined window not prepared"` check at `verify.cpp:4235-4238`, before `pre_launch_` at `4240`. A prestaged window may be restaged with different tokens/positions, so only the actual `pl_launch()` arguments are authoritative.

---

## 6. Graph-capture constraints

The insertion points above keep allocation outside all capture functions:

- Solo capture: `capture()` → `record_window()`, `verify.cpp:2454`, `1106`.
- Batch capture: `capture_batch()`, `verify.cpp:3495`.
- Pipeline precapture: `capture_all()`, `verify.cpp:4174`; `pl_launch()` never captures.
- Input-only helpers: `stage_inputs()`, `pl_stage()`, `prestage()`.

Do not put ensure in `record_window()`, `capture()`, `capture_batch()`, `capture_commit*()`, or `prestage()`.

Required behavior:

1. Graphs retain the fixed `st.page_table` and fixed pool-base pointers. Only table **contents** change; no graph recapture.
2. Zeroing and H2D table upload must be enqueued on the same `cs_`/`m.cs` before graph launch.
3. No ensure/allocation occurs while recording a graph. Capture uses test state and does not execute KV writes.
4. A failed ensure returns before the corresponding window/chunk launch.
5. No page replacement/free/structural table change while that window is in flight. The pool mutex protects accounting, not illegal concurrent writes.
6. `commit()`, `commit_slot_prefixes()`, and `pl_commit_async()` remain recurrent/indexer commit paths; they need no KV ensure.

Both `.cpp` files will need access to the private pool declaration, e.g. `#include "core/kv_pool.hpp"` from `src/prefill/prefill.cpp` and the local declaration from `src/core/verify.cpp`; `layer.hpp` itself only forward-declares `GlobalKvPool`.## Current state

Read `docs/IMPL_PLAN_KV_POOL.md` §3.6, step 3, and step 6, plus the current implementation.

- HEAD `e0a4cc46` contains step‑1 allocator code and step‑2 QSA sizing/binding support.
- `generate.cpp` currently has **no** `--kv-pool` options, pool creation, startup binding, validation, or INFO integration.
- No files were edited. Findings below are static analysis; no builds or GPU tests were run.

## 1. Precise startup pool-creation point

### Minimal viable insertion

Create the per-stage pools **immediately before the CUDA0 main-session block at `src/program/generate.cpp:5194–5196`**, after the final layer ranges and existing memory-reservation policy have been resolved, but before:

- CUDA0 session allocation at `5215–5219`
- later-stage sessions at `5251–5252`
- batch-slot sessions at `5395–5400`
- primary expert-cache sizing/opening at `6017–6415`, especially `open/open_sized` at `6363–6364`
- later-stage cache opening at `6550–6570`, especially `6566`

At this point the split search is finished, so every stage’s `[lb, le)` and device are known.

### Important limitation: workspaces are reserved, not yet allocated

There is **no existing single location** that is simultaneously after all actual workspaces and before expert-cache opening:

- Prefill workspaces initialize at `generate.cpp:8801–8827`.
- Verifier workspaces initialize around `9080–9209`.
- Pipeline graphs are captured at `9305`.
- All are after expert-cache creation.

Therefore, the minimal step‑3 design must create pools after the existing **hard byte reservations** are computable, not after actual workspace initialization. Moving actual prefill/verifier initialization ahead of the caches would require separating workspace allocation from cache binding and is not a minimal step‑3 change.

### Required capacity ledger

Do not use bare `cudaMemGetInfo() * 50%`.

Before pool creation, with pooled QSA sizing enabled, calculate per device:

```text
available =
    current_free
  - main_session_bytes
  - requested_batch_slots * slot_session_bytes
  - vram_reserve / vram_reserve_later
  - owned-prefill reservation, when not borrowing
  - verifier/window reservation
  - pipeline second-verifier and GDN-snapshot reservation
  - drafter/head reservation on the device that owns them
```

Then:

- explicit `--kv-pool-mib N`: request `N MiB` per stage and fail if it violates the hard ledger;
- automatic size: `max(available, 0) * 50%`;
- fail before READY if the hard requirements cannot fit. Do not rely on `Verifier::instantiate_evicting()` to rescue an under-reserved startup.

The existing relevant reservations are at:

- `generate.cpp:4489–4514`: prefill, `kWindowMib`, `kDrafterMib`, pipeline estimates
- `4563–4572`: `stage_room()`
- `6019–6025`: primary pipeline reservation
- `6030–6040`: owned-prefill pricing
- `6111–6114`: primary cache reserve, including draft binding

For the early pool calculation, the corresponding reserve terms must be hoisted/reused, not newly invented. In particular, resolve any pool-path WDDM reserve adjustment before sizing the pool rather than letting `6088–6096` increase it afterward.

### Why not simply create pools at `6016`?

That is visibly just before `ExpertCache xcache`, but it is too late for the current initialization API:

- all session arenas have already been sized and carved;
- `qsa_state_init()` binds the pool during initialization;
- `session_zero()` has already run.

A `6016` insertion would require deferred-pool markers, deferred base binding, and changed zeroing behavior. That is more invasive than creating pools before the sessions.

## 2. Per-stage loops and state binding

Use one `GlobalKvPool` per stage GPU, shared by all eligible main QSA layers on that stage—not one pool per layer.

Add pool ownership alongside the primary session and in `GpuStage` (`generate.cpp:1208–1233`). Before creating pools, enable a separate **pooled-sizing flag**, because split-search calls to `session_bytes()` occur before pool objects exist:

- `session_bytes()` in split-skip estimate: `4211`
- split-search stage pricing: `4579`

A small `qsa_set_kv_pool_sizing(bool)` or equivalent is needed. `qsa_state_bytes()` should treat `sizing_enabled || g_kv_pool != nullptr` as pooled; `qsa_state_init()` should still require an actual pool before binding.

Binding sequence:

1. **CUDA0 main session**  
   Before `generate.cpp:5215`, call `qsa_set_kv_pool(primary_pool.get())`.  
   `session_init()` then binds every eligible QSA state through `qsa_state_init()`.

2. **Later-stage main sessions**  
   In the loop at `5247–5254`, select `st.kv_pool.get()` before both `session_bytes()` and `session_init()`.

3. **Batch slots**  
   In the stage loop at `5390–5412`, select the pool for stage `k` before calculating `bytes` and before every slot’s `session_init()`:
   - `k == 0`: primary pool
   - otherwise: `stages[k - 1]->kv_pool`

   Thus `bslot_ss[k][b]` shares the same stage pool as the corresponding main session.

4. **Drafters**  
   Immediately after all main/slot sessions and before `MtpDrafter mtp` following the batch block, clear both the active binding and pooled-sizing mode. Do not rely solely on `ring_cells`: the plan explicitly excludes drafters even when a drafter happens to use mode 0.

The eligible iteration everywhere is:

```cpp
for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
    QsaState& st = ss.qsa_states[ss.qsa_ord0 + j];
    if (st.kv_pool != nullptr) { /* ensure/bind diagnostics */ }
}
```

`split_same` verifiers and the two pipeline parity verifiers share their stage’s existing `SessionState` and pool; they do not need additional pools. Production startup must not use `bind_identity_for_test()`.

## 3. QSA sizing facts

`session_bytes()` at `src/core/session.cpp:56–81` prices all QSA states in the stage’s layer range. It calls `qsa_state_bytes()`, which reaches the common implementation in `kv_pool.cpp`.

Pooled sizing at `kv_pool.cpp:118–136` removes only K/V storage:

```cpp
pooled = config.pooled && ring_cells == 0 && mode == 0 && !config.elastic;
```

It retains:

- device page table
- indexer tail/dead/pooled/block position
- RoPE table
- step/status/position state
- alignment and other session buffers

Pool capacity is instead:

```text
pages = floor(capacity_bytes / page_price)
```

`qsa_layers` is metadata/assertion information, **not a byte multiplier**.

One allocation-tail discrepancy should be accounted for in the budget: `kv_pool_storage_bytes()` adds one 64-byte quantized tail, while `GlobalKvPool::create()` adds 64 bytes **per allocated component**—up to 256 bytes for INT8. It is small, but the hard ledger should describe actual allocations.

## 4. Step‑3 launch-time `ensure()` hooks

All reservations must occur on the stream that will perform the subsequent KV writes, and never during graph capture.

| Path | Exact insertion and range |
|---|---|
| Prefill, every stage/chunk | `src/prefill/prefill.cpp:2721`, immediately after `T` and `p0` are computed and before chunk work. For every local QSA state: `ensure(st, p0, p0 + T, m.cs, err)`. This automatically covers recursive/asynchronous stage calls at `4931–4938`. |
| Solo normal decode | `src/core/verify.cpp:2865` `Verifier::run()`. After required capture/capture-commit and before `stage_inputs()` or any launch. Ensure `[pos0, pos0 + T)` over `*ss_`, using `cs_`. Skip when `prelaunched_ == T`; that launch was already covered by `prelaunch()`. |
| Split prelaunch | `verify.cpp:2839` `prelaunch()`. After existing validation and before `stage_inputs()` at `2848`, ensure the next stage’s own `ss_` on its `cs_`. This is essential because `run()` does not launch a prelaunched graph again. |
| Ordinary batch and batch-groups | Put the common implementation in `stage_batch()` at `3695`, after `capture_batch()`/`capture_commit_batch()` and before return/launch. It covers both `run_slot_rows()` (`3779`) and `batch_launch()` (`3947`). For each unique slot, use its own `slots_[rows[t]]`, `committed = first position`, and `new_write = max(pos[t] + 1)` across all its rows. |
| Pipeline windows | `verify.cpp:4228` `pl_launch()`, after preparation/context checks and before `pre_launch_`, staging, or `cudaGraphLaunch`. Ensure `[pos0, pos0 + T)` over that verifier’s `ss_` on `cs_`. `prestage()` and `capture_all()` need no allocation. |

With the common `stage_batch()` hook, duplicate loops in `generate.cpp::batch_step()` (`11583`) and pipeline `pump()` (`11900`) are unnecessary. The inactive pad rows prepared at `11886–11887` are nevertheless included in `stage_batch()`’s row coverage.

Rejected MTP tails remain allocated; only the logical committed position rolls back. Step 3 should not add tail-page reclamation.

## 5. Args and environment validation

Add:

- `Options` fields beside `kv_grow` at `generate.cpp:486–488`
- usage text beside KV options at `833–838`
- `--kv-pool` and `--kv-pool-mib N` parsing beside `--kv-grow` at `2481–2483`

Perform centralized validation **before `qsa_set_kv_resident()` at `3204`**, before any split sizing or VMM initialization:

- `kv_pool && kv_resident > 0`: reject
- `kv_pool && (o.kv_grow || STRATA_KV_GROW enabled)`: reject
- `kv_pool && STRATA_BATCH_ATTN_ROWS` numerically nonzero: reject
- explicit `kv_pool_mib <= 0`: reject
- unsupported non-CUDA/HIP backend: reject
- checked MiB-to-byte conversion
- capacity below one complete page: reject from `GlobalKvPool::create()`

The existing consumers are:

- `STRATA_KV_GROW`: `generate.cpp:5199`
- `STRATA_BATCH_ATTN_ROWS`: `verify.cpp:1805–1807`

Validation must use the same enabled semantics and must not silently alter either setting. All pool-only checks remain conditional on `--kv-pool`.

## 6. INFO and diagnostics

For **step 3 only**, keep stdout protocol unchanged and emit one startup stderr diagnostic per stage at pool creation, for example:

```text
strata generate: KV pool: CUDA0 stage=0 qsa_layers=3 capacity_mib=2048
page_bytes=... pages=... free=... high_water=0
```

The full dynamic INFO work belongs to step 6:

- startup INFO insertion point: `generate.cpp:11078–11103`
- enabled only: append `kv_pool_pages=N kv_pool_free=M`
- choose the stage with minimum `free_pages()` as the bottleneck
- per-stage capacity/high-water remains on stderr
- runtime allocate/free INFO and server/Prometheus handling are step 6, not required for step 3
- pool-off output must gain no keys or diagnostics

## Minimal step‑3 change set

1. Add the small pooled-sizing mode needed before pool objects exist.
2. Add options, usage, and centralized fail-fast validation.
3. Create per-stage pools before the session block using the hard reservation ledger.
4. Bind primary, later-stage, and all batch-slot sessions to their stage pool; explicitly exclude drafters.
5. Add the five launch-coverage hooks above, with batch coverage centralized in `stage_batch()`.
6. Add startup per-stage diagnostics and fail-closed error propagation.

Do **not** include snapshot/SAVE gather-scatter, lease release/replacement, BDONE lifecycle changes, slot-cache invalidation, Prometheus/server changes, or scheduling work in step 3. Those belong to steps 4–6.已完整重读指定章节。现有代码尚未调用 `ensure()`；启动顺序里 session arena、workspace 预留和 expert cache 彼此交错，我会沿用现有显存预留账，不用裸 `cudaMemGetInfo()` 计算池容量，并把 reservation 放在 graph capture 和实际 launch 之前。

