# 阶段 1 实现计划：全局 KV 页池

> 状态：实现前计划，尚未实现或实测。本文行号基于当前工作区代码，新增代码之后的行号需要重新核对。
>
> 范围只包括 `docs/VLLM_REDESIGN.md` §7 阶段 1：`kv_mode == 0` 的主 QSA 层全局页池、按需分配、结束回收、真实水位，以及所有依赖逻辑 KV 布局的主机拷贝。调度器、抢占、跨请求前缀共享、`kv_mode == 1` 池化一律不做。
>
> 总开关 `--kv-pool` 缺省关闭。关闭时不得改变既有分配、拷贝顺序、INFO、指标和数值结果；本功能是 opt-in，默认路径必须与上一发布版 byte-identical。

## 0. 范围、术语与必须先解决的代码事实

### 0.1 MVP 边界

- 每个 stage GPU、每种实际启用的 KV 格式一个静态池；页大小沿用 4 cell。
- 只池化主模型 QSA 状态，即 `qsa_state_init(..., ring_cells == 0)` 且最终 `kv_mode == 0` 的状态。
- MTP drafter 不池化，包括它碰巧以 `kv_mode == 0` 分配时；不能仅凭 `kv_mode` 判断是否入池。
- FP16、INT8、Q4_0、K8V4 复用同一分页物理布局，但分别按真实字节单价建池；MVP 同时只使用配置选定的一种格式。
- `idx_tail/idx_dead/idx_pooled/idx_block_pos`、RoPE、step、GDN、PLE 仍是每会话状态，不进入池。
- 不增加引用计数、块哈希、写时复制、waiting/running 调度队列或抢占。
- 页耗尽时，在发起写入前 fail closed，返回现有 `ERR`，不得让 kernel 读到 `-1` 或越界；MVP 不承诺新增排队协议。验收负载必须不触发该路径。

### 0.2 阅读源码后补充的三个必要改动

这三点不能只照设计文档中的文件清单机械实现：

1. **磁盘 SAVE 也要逻辑页寻址。** `conversation_kv_source()`（`src/core/conversation_snapshot.cpp:298-309`）直接按 `src[part] + offset` 读物理池。它不是第五种快照语义，而是 SAVE 的流式 gather 入口；必须与 save/restore 共用逻辑页读取器，否则文件内容会静默错误。
2. **idle slot 的页不能既回收又当活会话复用。** `BDONE` 后 `BSlot.cached` 可保持为 true（`src/program/generate.cpp:11650` 附近），后续 `slot_source`/`copy_from_slot()` 会复用设备状态（入口选择约 `12620-12650`）。因此“请求结束回收”必须先完成现有快照/停泊保存，再使设备缓存失效，最后回收；或者保留页直到缓存真正被替换。不能提前 free 后仍走原 slot 快路径。
3. **池容量是 stage 上全部主 QSA 层的总容量。** `session_bytes()` 经 `qsa_state_bytes()` 为每个 QSA 层定价（`src/core/session.cpp:60-81`），不能只把 `--kv-pool-mib` 当成每层容量。静态池应在每 stage 的会话/slot 状态扣除 KV 后、专家缓存打开前分配，使剩余显存仍能交给专家缓存；否则既可能启动 OOM，也测不到“等池容下专家多驻留”。

## 1. 现状精确梳理

### 1.1 KV 池当前如何分配

主分配调用链：

```text
generate.cpp 的 stage/slot 初始化
  -> session_bytes(g, max_cells, K, layer_lo, layer_hi)
       -> 每个 QSA 层调用 qsa_state_bytes()
  -> cudaMalloc(session arena)
  -> session_init()
       -> 每层 qsa_state_init()
            -> kv_plan()
            -> 从本 session arena 的 Cursor 中 carve K/V、page_table 等
```

具体事实：

- `kv_plan()`（`src/core/layer.cpp:738-756`）先计算 `pages=ceil(max_cells/page_size)`，默认 `slots=pages`。只有 `g_kv_resident > 0` 且不是强制全驻留的 ring 时才改成流式或环形。
- `kv_pool_bytes()`（`layer.cpp:758-771`）按页数计算 FP16/INT8/Q4_0/K8V4 字节数。它只负责单价和总量，不分配内存。
- `qsa_state_bytes()`（`layer.cpp:774-791`）把 VRAM K/V slots、`page_table`、indexer、RoPE、step 等全部计入一个会话 arena；VMM elastic 的 mode-0 K/V 是唯一不计入 arena 的现有分支。
- `qsa_state_init()`（`layer.cpp:794` 起）在 `803-854` 附近从 `Cursor` carve K/V；`856-860` 分配页表；`895-928` 为 mode 1/2 分配恒等布局的 host authoritative copy。
- 主会话的实际分配在 `src/program/generate.cpp:5217-5225`，其他 stage 在其后同样执行；slot 会话集中在约 `9280-9330` 的 `bslot_ss` 初始化循环。每个 stage 的 arena 只含本 stage 层段，但 slot 数按各 stage 能容纳的最小值收敛。
- `--kv-grow` 是另一条 VMM 路径：`qsa_set_kv_elastic()` 在 `generate.cpp:5212` 调用；elastic pool 的 registry 和映射逻辑在 `layer.cpp:625-730` 附近。它不改变全局空闲页所有权，也不能跨会话回收复用。
- `qsa_state_zero()`（`layer.cpp:937` 起）目前清零整片 K/V slots。共享池化后不能继续这样做，否则新会话会清掉其他会话的页。

### 1.2 页表当前如何填充，恒等映射从哪里来

- `QsaState::page_table` 是 device `(n_pages,)` int32，定义于 `include/strata/core/layer.hpp:229-236`；mode 0 的注释明确说它是 identity table。
- 真正的 identity 初始化在 `qsa_state_init()` 的 `layer.cpp:963-967`：主机 `tab[i]=i`，再 `cudaMemcpy` 到 device。
- mode 1 在 `layer.cpp:968-969` 调 `kv_stream_reset()`，页表兼作驻留图；mode 2 在 `970-971` 调 `kv_ring_table()`，映射为 `block % n_slots`。
- 分页布局在 `include/strata/kernels/qsa.hpp:63-76`：`[page][kv_head][page_size][head_dim]`；`qsa_real_shapes()` 在 `105-116` 设置 `page_size=4`。物理页号来自 `table[t/page_size]`，`qsa.hpp:176-180` 明确页表不假设恒等。
- 页表容量仍按 `max_cells` 保留；MVP 只改变其中的物理页号，不改变页表地址、`n_pages`、launch capacity。

### 1.3 快照/拷贝的 4 条路径及 SAVE 相关入口

| 路径 | 入口与准确行号 | 当前寻址方式 |
| --- | --- | --- |
| 保存/停泊 gather | `conversation_kv_save()`，`conversation_snapshot.cpp:147-180` | `pools(st)` 返回物理数组；`visit()` 的绝对偏移 `at` 同时用于 host image 和 `src[i]+at`（`165-177`）。完整物理前缀还用于增量复用，`whole_cells` 在 `157` 计算。 |
| 恢复/借用前缀 scatter | `conversation_kv_restore()`，`253-261`；`conversation_kv_restore_prefix()`，`263-283` | host image 的连续字节直接写到 `dst[i]+at`，默认逻辑页号等于物理页号。 |
| 槽间直拷 | `conversation_kv_copy()`，`223-250` | 对五个数组直接 `copy_nonblocking(dst[i],src[i],sizes[i])`；`generate.cpp` 的 `direct_slot_copy()` 在 `11404-11424` 调用。 |
| verify 指纹/回读 | `conversation_kv_verify()`，`315-356` | `compare()` 从 `device+offset` 读取；主 authoritative payload 在 `340-345` 按物理前缀比较和 FNV 哈希，draft ring 另有专门分支。 |
| SAVE 流式读取（同一 gather 问题的另一入口） | `conversation_kv_source()`，`298-309` | lambda 捕获 `src` 后从 `src[part]+offset` 读，磁盘 writer 以为拿到 identity-layout 字节流。 |

共同的间接寻址来源是 `pools()`（`conversation_snapshot.cpp:76-88`）。`layout()` 计算的是逻辑快照大小，并未把逻辑页转换成物理页。`conversation_copy.hpp:20-45` 只提供 private non-blocking stream 和同步/queued 语义，不理解 KV 页。

### 1.4 页表在哪些 kernel/launch 中使用

**solo/layer 路径：**

- `qsa_attn_pools()` 在 `layer.cpp:976-985` 将 `st.page_table` 和各格式池基址装入 `QsaAttnPools`。
- append：`layer.cpp:1216-1227`，调用 `kv_append_q8_step`、`kv_append_q4_step`、`kv_append_step`。
- gather：`layer.cpp:1269-1276`，调用量化 gather 或 FP16 `kv_gather_step`。
- 直接 decode attention：`layer.cpp:1265-1266` 的 `qsa_decode_attn_step`；`QsaAttnPools` 定义于 `include/strata/kernels/qsa_decode_attn.hpp:23-33`。
- capturable gather API 见 `qsa.hpp:334-340`；append API 见 `365-368`。容量定型 launch 的约束见 `qsa.hpp:326-340` 和 `layer.cpp:1250-1259`。

**verify/batch 路径：**

- 单窗口 append：`src/core/verify.cpp:1753-1766`；逐行 append：`1774-1786`。
- `STRATA_BATCH_ATTN_ROWS=1` 环境读取在 `1804-1807`；逐行 pools 组装在 `1815-1819`，`qsa_decode_attn_batch_rows()` 在 `1826-1829`。
- slot 融合 attention 的 pools 组装和 launch 在 `1837-1859`；普通逐行 batch 在 `1861-1874`；非 batch 路径在 `1888-1899`。
- 这些 graph 在 `Verifier::capture()`（`verify.cpp:2454` 起）捕获；普通 launch 在 `run()` 的 `2948-2960`，prelaunch 在 `2857`，batch launch 在 `run_slot_rows()` 的 `3785`。页表内容必须在这些 launch 前就绪。

**prefill 路径：**

- `Prefill::run_impl()` 在 `src/prefill/prefill.cpp:2587`，外层 `Prefill::run()` 在 `5131`。
- 主 QSA 的 prompt append 位于 `prefill.cpp:3342-3366` 附近，量化格式调 `kv_append_q8/q4`，FP16 调 prefill `kv_append`。
- prompt attention 经 `QsaAttnPools` 读页表；`qsa_prompt_attn.cu` 中多处 `page_table[cell/page_size]`。`src/prefill/kernels.cu` 的 `kv_append` 实现同样接收 `page_table`（按当前文件中的函数定义位置核对，不固定引用行号）。
- MTP 的 prompt draft 写入在 `Prefill::draft_kv()`（`prefill.cpp:1715` 起），不属于 MVP 池。

## 2. 目标结构与接口伪代码

### 2.1 `GlobalKvPool` 的最小职责

```cpp
// src/core/kv_pool.hpp；只声明，实现包含 layer.hpp 以避免头文件循环。
namespace strata::core {
struct ModelGeometry;
struct QsaState;

class GlobalKvPool {
public:
    static std::unique_ptr<GlobalKvPool> create(
        int device, const ModelGeometry& g, int64_t qsa_layers,
        int64_t capacity_bytes, std::string& error);

    // 仅调用方主机互斥；device 拷贝在 caller_stream 上排序。
    // 不缩小 committed_cells；只补 [old_write_cells, new_write_cells) 所需新页。
    bool ensure(QsaState& st, int64_t committed_cells, int64_t new_write_cells,
                void* caller_stream, std::string& error);

    // restore/direct-copy 的目标先取得新 lease，旧 lease 延迟到 scatter 完成后再还。
    // 仅供 prepare-all-then-copy 使用；此调用不发布映射、不释放旧 lease。
    bool prepare_prefix(QsaState& st, int64_t image_cells, KvLeaseTxn& txn,
                        std::string& error);
    bool publish_prepared(QsaState& st, KvLeaseTxn& txn, void* caller_stream,
                          std::string& error); // 页表更新成功后才替换旧 lease
    void abort_prepare(KvLeaseTxn& txn);       // 全部新页回池，旧映射原样保留
    bool release(QsaState& st, void* caller_stream, std::string& error);

    void bind_main_state(QsaState& st);       // 只绑定固定池基址和标记，不取页
    int64_t pages() const;
    int64_t free_pages() const;               // 原子读取，供 INFO
    int64_t high_water_pages() const;
    bool owns(const QsaState& st) const;
private:
    // 固定 device base、每部件 page_bytes、host free list、每 state host mirror。
};
}
```

实现要点：

- `create()` 用 `kv_pool_bytes(shapes, 1, ...)` 的**真实页总价**求 `pages=floor(capacity_bytes/page_price)`，再按部件一次性 `cudaMalloc`。总额必须覆盖该 stage 所有主 QSA 层共享这一组页；`qsa_layers` 用于断言/日志和容量估算，不再给每层复制一个池。
- 每个 `QsaState` 增加 `GlobalKvPool* kv_pool`、`int64_t kv_pool_committed`、`int64_t kv_pool_write_top` 及 host 页表镜像的句柄。graph 只捕获 device pool base 和 device page-table pointer。
- host mirror 是页分配权威；device page table 只是其镜像。空闲链表和 host mirror 用一把池级 mutex 保护，因为现有 `STRATA_PIPED_CONCURRENT` 可让 prefill 与窗口主机代码并存；不得新增后台线程。
- `ensure()` 以 8 页为取页批次，但只按实际逻辑页记账。新页先在 caller stream 上清零，再 H2D 更新对应页表项，最后才允许 launch。批取不足时整个批次回滚并返回失败。
- `committed_cells` 是已提交逻辑前缀；`new_write_cells` 是本 chunk/window 可能写入的最右边界。MVP 可以保留本请求内被拒绝 draft 的尾页，逻辑提交点回退不等于立即 free。
- `release()` 只在设备静默、没有 snapshot/source 正在读取该状态时调用；清空 host mirror 和 lease 后才将页号放回 free list。
- `n_pages/max_cells` 不变。mode 0 的 `n_slots` 可继续表示逻辑容量，不能再被解释为“本 state 独占的物理 slots”。

### 2.2 所有写入点的统一顺序

```text
prefill chunk [pos0, pos0+T):
  pool.ensure(main_state, committed=pos0, write=pos0+T, stream=prefill_stream)
  -> Prefill::run_impl()

solo verify window [pos0, pos0+T):
  pool.ensure(main_state, committed=pos0, write=pos0+T, stream=verifier_stream)
  -> Verifier::run()/prelaunch()

batch rows:
  对每个 slot、每个 stage 求 max(pos[row] + 1)
  -> 在各 stage 的 caller stream 上 ensure
  -> Verifier::stage_batch()/run_slot_rows()/batch_launch()

pipeline-windows:
  在 pl_launch 前按该窗口最大写入位置 ensure；
  错误窗口的页保留到请求/状态替换边界，回滚只移动逻辑 commit 点。

restore/direct-copy（先备齐全部层，再写任何目标状态）:
  枚举本次所有 stage 的全部目标 main QsaState
  -> 对每个 state prepare_prefix()，只记账，不发布、不碰旧 lease
  -> 任一层/任一 stage 失败：abort 整个 txn，旧状态及 free 计数不变
  -> 全部 prepare 成功后，逐层 publish 页表并 scatter
  -> 所有 device copy 成功后才允许 release 旧 lease

注意：异步 CUDA copy 失败可能已修改目标物理页，不能假装旧设备内容仍可继续推理。
此时按现有失败契约终止使用该 session/引擎；txn 只在设备静默后清理。
这里的“全成功再释放”解决的是页耗尽可安全回滚，不把 I/O 失败伪装成可恢复。
```

`Verifier::commit()`、`commit_slot_prefixes()`、`pl_commit_async()` 继续只提交 recurrent/indexer 状态；MVP 不在这些热路径立即回收尾页。这样避免把投机拒绝变成新的复杂页事务，同时满足“页分配先于窗口 launch”。

## 3. 按文件的改动清单

### 3.1 新增 `src/core/kv_pool.hpp` / `src/core/kv_pool.cpp`

| 函数 | 改什么 | 为什么 | 风险 |
| --- | --- | --- | --- |
| `create()` | 固定分配各 KV 部件的池基址，初始化 free list、页单价、格式和水位；检查 int32 页号及乘法溢出。 | 池基址终身不变，HIP 不依赖 VMM。 | 容量单位错、把总量错算成每层量、启动预算挤压 graph/专家缓存。 |
| `bind_main_state()` | 将共享 K/V base 写入 `QsaState`，设置 pool 标记，page table 暂不取页。 | 既有 kernel 不需要改签名。 | drafter 或 mode 1 被误绑定；必须显式拒绝。 |
| `ensure()` | 每 8 页取页、清零、更新 host mirror 和 device page table，维护 committed/write 高水位。 | 按需分配且保证 launch 前页有效。 | 异步拷贝流排序、部分失败回滚、并发分配、capacity 计算错误。 |
| `prepare_prefix()` / `publish_prepared()` / `abort_prepare()` | 一次事务覆盖 restore/direct-copy 涉及的所有 stage、所有目标层；先纯主机预留全部新页，成功后才逐层发布页表和 scatter，旧 lease 延迟释放。 | 防止第 N 层耗尽时前 N-1 层已半恢复；不足时旧映射/数据/free 计数不变。 | 瞬时双倍占用；异步 copy 失败仍须按现有 fatal 契约处理，不能继续用旧内容。 |
| `release()` | O(页数) 归还 lease，清状态句柄，更新 free/high-water。 | 真正跨请求复用。 | idle slot 仍被当成 live cache；在途 graph/snapshot UAF。 |
| `snapshot_page_table()` / `physical_page()` | 给快照 helper 提供同步后的 host 逻辑→物理映射。 | 禁止主机直接 D2H 猜页表，也禁止物理前缀假设。 | 拷贝期间页表被改写；调用契约必须要求 quiesced。 |

新增 `src/core/kv_pool_test.cpp`：纯主机 free-list/水位测试，以及 CUDA/HIP 可用时的页表更新和耗尽回滚测试。测试不依赖大模型。另设测试专用 `STRATA_TEST_KV_POOL_FAIL_AFTER=N`（仅 test build 或未显式用于生产的诊断路径），在第 N 个 reservation 返回确定性失败并记录 launch counter；生产启动仍用正常错误路径，禁止把故障注入默认开启。

### 3.2 `include/strata/core/layer.hpp`

- 在 `QsaState`（当前 `209-274`）增加池绑定和主机 lease 元数据，不改变既有字段的偏移语义假设之外的内容；不增加 device kernel 参数。
- 增加 `qsa_set_kv_pool(GlobalKvPool*)` 或等价的“下一组 main states 绑定池”初始化接口；`ring_cells != 0` 永远不绑定。
- 更新 `qsa_state_bytes/init/zero` 的契约注释：池化 state 的 K/V base 不归 session arena 所有，不能整池清零。
- 风险：全局初始化器把池错误地带入 drafter；测试必须同时创建 main 和 draft，断言 draft 仍为私有布局。

### 3.3 `src/core/layer.cpp`

| 函数 | 改什么 | 为什么 | 风险 |
| --- | --- | --- | --- |
| `kv_plan()`，`738-756` | 保持 mode/slot/pooled_rows 计算；池开关只作用于 `ring_cells==0 && mode==0`，与 resident/elastic 冲突提前拒绝。 | 不触碰 kv_mode 1/2。 | 在 sizing 和 init 之间选择不一致，导致 carve 错位。 |
| `kv_pool_bytes()`，`758-771` | 提取可复用的单页价格查询；保留原函数给默认路径和 drafter。 | 池容量和快照页拷贝使用同一真实布局单价。 | K8V4 三个部件及 scale 字节漏算。 |
| `qsa_state_bytes()`，`774-791` | 仅池化 main state 时跳过 K/V pool bytes；page table/indexer/RoPE/step 原样计入。 | KV 不再按 `max_context` 计入每会话。 | split search 和 slot fit 的定价若漏改会使用错误 arena。 |
| `qsa_state_init()`，`794-971` | 池化 main state 从 `GlobalKvPool` 绑定 base，不从 Cursor carve K/V；仍从 arena carve page table 等；不再上传 identity。 | 复用既有 kernel 的固定指针。 | `Cursor` 前后偏移必须与 bytes 完全一致；初始空 lease 不能留下旧页号。 |
| `qsa_state_zero()`，`937-975` | 池化 state 不清共享池，仍清 indexer；页清零只在 acquire 时进行。 | 防止跨会话破坏。 | stale padding 进入快照； acquire 清零是必要门禁。 |
| `qsa_attn_pools()`，`976-985` | 原则上不改；增加 `!pool || base/table 有效` 断言。 | kernel 路径零改动。 | 把 lease 元数据误当 device 指针。 |

`src/core/session.cpp` 不需要另写一套扣费公式，它已通过 `session_bytes()` 间接使用 `qsa_state_bytes()`（`60-81`）；必须加 sizing/init 一致性测试，不能重复实现容量算法。

### 3.4 `src/core/conversation_snapshot.cpp` + `src/core/conversation_copy.hpp`

在 `conversation_copy.hpp` 增加**主机侧拷贝原语**，不增加数值 kernel：

```cpp
copy_logical_range(st, part, logical_byte_offset, dst_or_src, bytes, direction,
                   page_table_snapshot, queued=false);
// 将 offset 分成 logical page；每页按 part_page_bytes 和 physical_page 定位。
// pooled(indexer) part 仍为普通连续拷贝。
```

在 `conversation_snapshot.cpp` 增加 `PageView`，一次保存/恢复调用只读一次 host mirror。`valid()` 在池化 mode 0 时检查 lease 覆盖 `ceil(upto/page_size)`，而不是要求私有 `n_slots` 整片存在。

| 函数 | 改什么 | 为什么 | 风险 |
| --- | --- | --- | --- |
| `conversation_kv_save()`，`147-180` | K/V 四个 part 按逻辑页 gather 到 identity-layout image；indexer pooled 仍平铺。增量复用按**逻辑** `whole_cells` 跳过完整页，部分页重拷。 | image 格式不变，物理布局不影响快照。 | 大量 4 KiB 左右碎拷贝、分段 `ConversationBuffer::visit()` 的 `at` 处理错误。 |
| `conversation_kv_restore()`，`253-261` | 反向 scatter；使用 `replace_prefix()` 准备 lease。 | 兼容旧/新 identity-layout 文件和停泊镜像。 | 旧 lease 过早释放、部分失败留下不可继续的状态。 |
| `conversation_kv_restore_prefix()`，`263-283` | 只 scatter `[0, upto)` 对应逻辑页；保留原 prefix 语义。 | 借用检查点不依赖物理前缀。 | `captured` 与 `upto` 的页边界和 moving spare row。 |
| `conversation_kv_copy()`，`223-250` | 源按逻辑页 gather、目标按自身页表 scatter；目标先独立取页，成功后释放旧 lease。保留 `queued` 和 `refresh_residency()` 的顺序。 | `STRATA_SLOT_COPY_DIRECT=1` 不能退回物理前缀拷贝。 | 非原子多层直拷失败、私有 stream 与后续 legacy stream 排序。 |
| `conversation_kv_verify()`，`315-356` | authoritative K/V 按逻辑页回读、比较和哈希；image 顺序不变；mode 2 分支原样保留。 | 指纹应与物理分配无关。 | 用物理页序哈希造成假失败，或漏掉页内部分偏移。 |
| `conversation_kv_source()`，`298-309` | `read(part,offset,...)` 通过同一逻辑页 helper；image metadata/sizes 不变。 | SAVE 文件仍是 identity layout，可跨开关恢复。 | 16 MiB writer read 跨多页、lambda 生命周期超过 lease。 |
| `pools()/valid()`，`76-112` | 池化 state 仍返回共享 base；valid 检查 pool lease；mode 1/2 原逻辑不动。 | 最小侵入复用部件数组描述。 | 把 host authoritative 语义错误地带入 mode 0。 |

性能策略按设计先实现每 8 页批量排程，不新增 kernel。每个 batch 内的 K/V/scales 仍各是独立部件，不能把不连续 host segment 当成连续内存。`copy_nonblocking()` 的完成/queued 契约和 `progress_beat()` 必须保留。

### 3.5 `src/core/verify.cpp` 与 `src/prefill/prefill.cpp`

设计文档希望 verify 行组装零改动；MVP 只增加 launch 前置保证，不改任何 kernel：

- `verify.cpp`：在 `run()`（2865）、`prelaunch()`（2839）、`stage_batch()`（3695）、`batch_launch()`（3947）、`pl_launch()`（4228）的主机入口增加 ensure 调用或检查调用方已完成的 reservation。**`prelaunch()` 自身必须在 `stage_inputs()` 和真正 `cudaGraphLaunch()` 前完成本 stage 的 reservation**；不能依赖稍后 `run()`，因为 split overlap 下 `run()` 看到 `prelaunched_` 后不会再 launch。`run()` 准备当前和 next-stage graph 时，也必须在首个 launch 前备齐两侧页。graph capture 本身只执行测试状态，不能申请真实新页。
- `prefill.cpp`：在 `run_impl()`（2587）写 K/V 前确保 `[pos0,pos0+n)`。`Prefill::run()` 外层各 chunk 也要传递准确的 reservation，不能只在整个 prompt 开始时按短 chunk 预留一次。
- `src/program/generate.cpp::batch_step()` 在 `ver.run_slot_rows()`（当前 `11547`）前统一计算并预留各行，避免每个 stage 对页所有权理解不同。
- `--batch-groups` 的 `pump()` 在 `generate.cpp:11875-11903` 组 row：inactive slot 也可能作为 pad row 写状态，并在 `11886` 清 `cached`。这里必须在 launch 前使它的设备缓存失效并释放旧 lease，再为 pad row 分配合法页；不得让 inactive pad row 写已回收页，也不得保留一个已被 pad 覆盖的“可复用”lease。
- 风险：同一 `QsaState` 被两个 stream 同时 ensure/write；现有执行纪律必须保证不会发生，mutex 只保护池账本，不能把非法并发变成合法。

### 3.6 `src/program/generate.cpp`

1. **配置解析和拒绝规则**
   - `Options`（当前 KV 字段在 `486-488`）增加 `kv_pool=false`、`kv_pool_mib=-1`。
   - 在 `--kv-grow` 解析（`2482-2483`）附近加入 `--kv-pool` 和 `--kv-pool-mib N`，usage 同步。
   - 在 `qsa_set_kv_resident()`/VMM 初始化前做集中校验：
     - `kv_pool && (kv_resident>0 || kv_mode!=0)`：硬错误；MVP 不池化流式 host authoritative copy。
     - `kv_pool && (kv_grow || STRATA_KV_GROW 非 0)`：硬错误，不允许一个静默关闭另一个。
     - `kv_pool && STRATA_BATCH_ATTN_ROWS 非 0`：启动硬错误，消息明确为互斥；不能自动改环境，也不能继续跑一半功能。
     - 非 CUDA/HIP 后端、`kv_pool_mib<=0`（显式给值时）、容量不足一页：硬错误。未显式开启时完全不做这些检查，保持默认行为。
   - `--pipeline-windows` 不能因池化被静默改变；其自动/显式路径都走相同 launch 前置 ensure。
2. **池生命周期**
   - 采用明确的启动顺序：配置/VMM 校验 → 全部 main session 和全部 slot session 的 sizing 与实际分配 → prefill/verifier 的现有 workspace 预算落定 → **每 stage 创建池** → 专家缓存 `open/open_sized` → graph capture 和 READY。若实际代码无法移动 slot 初始化，则先把全部 slot 的扣 KV 后字节作为 hard reservation，禁止只按 main session 后的瞬时 free 计算池。
   - 显式 `--kv-pool-mib` 为每 stage 上限；缺省用“扣除现有 `vram_reserve*`、全部 slot/workspace hard reservation 和 graph 预算后的可分配显存 ×50%”，不能裸用 `cudaMemGetInfo()`。`Verifier::instantiate_evicting()` 后续可能 evict graph 腾显存，因此必需 workspace/graph 不满足时必须在 READY 前失败，不能等运行时 OOM。
   - 启动日志打印上述分配顺序和每 stage 预算。单卡、多卡 split、`--batch 8`、graph capture 开启都加启动测试。
   - 在所有 main `session_init` 前绑定该 stage 池，drafter 初始化前解除 main 绑定或依赖 `ring_cells` 强制排除。
   - 请求从空状态开始、restore、slot copy、BYIELD、cache 替换时明确 lease 转移；不能依赖 `session_zero()` 隐式 free。
   - `BDONE` 两条结束分支（约 `11620-11655`、`11800-11840`）：若要保留下轮复用，先经现有 snapshot/parking 路径保存，再把 `cached` 标为不可直接设备复用，最后释放 lease。`copy_to_slot/copy_from_slot` 成功后也要清掉被替换 lease。
   - 进程退出先停请求和 graph，再释放 session 引用，最后析构各池；不依赖 CUDA context 已销毁后的 `cudaFree`。
3. **INFO**
   - 启动主 INFO（`11078-11103`）在开关开启时附加 `kv_pool_pages=N kv_pool_free=M`；关闭时一个 key 也不增加。
   - 每次 allocate/free 后输出同格式 INFO 并 flush。多 stage 用 `free_pages` 最小的瓶颈 stage 及其 `pages`，因为各 stage 同页价/同容量配置；另在 stderr 打每 stage capacity/high-water 便于排查。
   - 只读观测面按设计可在 `/v1/status` 增加 `kv_pool` 对象；不得改 HTTP API 既有字段，也不在 MVP 增加 preempt 计数。

### 3.7 `serve/server.py` + `serve/prometheus.py`

- 把 `1067-1070` 的 INFO 键值解析提取成共享 helper；`_pump()`（`1185` 起）收到运行时 INFO 也更新 `Engine.info`，不能只解析启动前 INFO。
- `info` 从只读启动事实变成可更新状态后，用锁或原子替换 dict；`metrics()` 在 `4581-4583` 构造 engine dict 时取一致快照。不转发 INFO 为 token，不改变 GEN/BGEN 协议。
- `serve/prometheus.py:73-82`：只有 `kv_pool_pages` 为正且 `kv_pool_free` 合法时，使用 `(pages-free)/pages`；缺字段时保留原“运行请求上下文占比”公式，保证默认测试和旧引擎 byte-identical。HELP 文案按所选数据源区分。
- `vllm:num_preemptions_total` 继续恒 0（`95`），阶段 1 不接通。
- `serve/test_prometheus.py` 增加 pool off fallback、pool on 水位、idle retained pages、非法字段 fallback；fake engine 协议测试增加运行中 INFO 更新。

### 3.8 构建与测试文件

- `CMakeLists.txt` 将 `kv_pool.cpp` 纳入 CUDA/HIP engine，并注册 `kv_pool_test`；snapshot 测试扩展非恒等页表和 SAVE source。
- 新增 `tools/kv_pool_stress.py` 只用于验收：构造 8 个均不超过 8K token 的真实请求、记录 BADM/BDONE/ERR 和 INFO 最低 free，并循环复用 slot。
- 单流 benchmark 可使用 `tools/batch_test.py` 的 `Engine` 类写独立 A/B 驱动，不把 wall-clock 采样噪声伪装成引擎指标。

## 4. 实现顺序：9 步，每步独立验证

以下 `CFG`、`EXE` 由执行者填当前模型配置和引擎绝对路径。GPU 命令需要模型和空闲显卡；本文没有代跑这些测试。

### 步骤 1：只落地纯分配器和主机单测，不接引擎

- **改动**：`kv_pool.{hpp,cpp}` 的容量计算、free list、8 页批取、耗尽回滚、水位；CMake 和 `kv_pool_test`。此步不绑定 `QsaState`。
- **验证**：
  ```sh
  cmake --build build-hip --target kv_pool_test -j2
  ctest --test-dir build-hip -R '^kv_pool_test$' --output-on-failure
  ```
- **预期**：分配/回收后 free 守恒，重复 10 万轮无双重分配，耗尽不改变账本，capacity 对四种格式逐字节符合 `kv_pool_bytes()`。
- **排查**：先打印 `page_price/capacity/pages/free`，用固定 seed 对照独立 `std::set` 模型；不要用模型输出排查 free-list。

### 步骤 2：绑定 state 和 sizing，但先使用 identity lease

- **改动**：`QsaState` 池标记、`kv_plan/qsa_state_bytes/init/zero`；启动分配池。为开发自测提供一次性 identity 分配 fixture，正式路径仍不启用。
- **验证**：
  ```sh
  cmake --build build-hip --target conversation_snapshot_test strata -j2
  ctest --test-dir build-hip -R '^(kv_pool_test|conversation_snapshot_test|conversation_validation_test)$' --output-on-failure
  python3 tools/batch_test.py --exe "$EXE" --config "$CFG" --batch 2 --n 2 \
    --extra '--pcie-frac 0 --adapt-every 1000000 --kv-pool --kv-pool-mib 2048'
  ```
- **预期**：session arena 只减少 QSA K/V 部分；drafter 大小不变；identity lease 下 greedy IDENTICAL；默认不带 `--kv-pool` 的既有日志和输出不变。
- **排查**：若启动错/Cursor 越界，对比每层 `qsa_state_bytes` 与 `qsa_state_init` 返回值；若仅 draft 错，检查 `ring_cells` 是否被误池化；若跨会话错，优先查 `qsa_state_zero` 是否仍清整池。

### 步骤 3：非恒等页表接入 prefill/decode，先做无 snapshot 新会话

- **改动**：`ensure()` 接入 prefill、solo、batch、pipeline launch；启动互斥校验；页获取清零和 H2D stream 排序。
- **验证**：
  ```sh
  ctest --test-dir build-hip -R '^(qsa_parity|kv_q8_parity|kv_q4_parity|kv_hybrid_parity|kv_window_parity|verify_batch_parity)$' --output-on-failure
  python3 tools/batch_test.py --exe "$EXE" --config "$CFG" --batch 8 --n 8 --max-new 128 \
    --extra '--pcie-frac 0 --adapt-every 1000000 --kv-pool --kv-pool-mib 2048 --prompt-cache 0'
  STRATA_BATCH_ATTN_ROWS=1 "$EXE" --serve --kv-pool ${OTHER_REQUIRED_ARGS} 2>&1 | tee /tmp/kv_pool_mutex.log
  ```
- **预期**：页表非恒等仍逐 token 一致；page table pointer/graph capture 次数不因页内容改变；互斥在 READY 前明确失败；默认路径不受影响。
- **排查**：`--no-capture` A/B 区分 graph 与分配；在 launch 前 dump host mirror；非法页号先于任何 attention 数值分析。若只有 MTP 拒绝后错，查 write-top 是否按整个 window 而非 accepted prefix 预留。

### 步骤 4：快照、直拷、verify、SAVE 统一逻辑页 gather/scatter

- **改动**：完成 §3.4；测试用固定置换页表，不依赖随机物理分配碰巧 identity。
- **必须新增的精确用例**：真实 pooled `QsaState` + 固定非恒等映射；`ConversationBuffer` 16 MiB segment 边界与 4-cell page 边界交叉；非页对齐 `restore_prefix(upto)`；`unchanged_tokens` 落在页内；`queued=false/true` 两种 direct copy；SAVE read 跨页和跨 segment；在第 N 层 prepare 强制耗尽，断言所有旧映射、旧数据和 free 计数不变；物理页复用给另一会话后无 stale lease/padding 泄漏。
- **验证**：
  ```sh
  cmake --build build-hip --target conversation_snapshot_test conversation_validation_test conversation_file_test -j2
  ctest --test-dir build-hip -R '^(conversation_snapshot_test|conversation_validation_test|conversation_file_test)$' --output-on-failure
  STRATA_SNAPSHOT_VERIFY=1 python3 tools/parking_test.py --exe "$EXE" --config "$CFG" \
    --extra '--conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0 --adapt-every 1000000 --kv-pool --kv-pool-mib 2048'
  python3 tools/batch_interleave_test.py --exe "$EXE" --config "$CFG" --flush 8 \
    --extra '--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --kv-pool --kv-pool-mib 2048'
  ```
- **预期**：snapshot byte size/format 不变；两种物理布局得到同指纹；停泊和 slot checkpoint 输出 IDENTICAL；`STRATA_SLOT_COPY_DIRECT=1` 再跑一遍也通过；SAVE 新引擎写出的文件可由缺省旧路径恢复，反向亦然。
- **排查**：先按 `part/logical_page/physical_page/offset` 打首个差异；indexer pooled 错与 K/V 错分开；`queued` 只在同步边界后检查，SAVE 失败单独检查 `conversation_kv_source()`。

### 步骤 5：请求/slot 生命周期回收和池耗尽保护

- **改动**：实现 lease replace/release，处理 BDONE、nonfinite、BSTOP、BYIELD、slot 换主、restore 失败、QUIT；idle cached slot 必须先保存再失效或继续持有 lease。
- **验证**：
  ```sh
  python3 tools/kv_pool_stress.py --exe "$EXE" --config "$CFG" --slots 8 --rounds 16 \
    --max-tokens 8192 --extra '--kv-pool --kv-pool-mib 2048 --pcie-frac 0 --adapt-every 1000000'
  python3 tools/parking_test.py --exe "$EXE" --config "$CFG" \
    --extra '--kv-pool --kv-pool-mib 2048 --pcie-frac 0 --adapt-every 1000000'
  ```
- **预期**：每轮结束 free 回到仅保留合法 lease 的水位，无单调泄漏；下一轮复用输出一致；人为一页池在写前返回 ERR，不崩溃、不 illegal memory access。
- **确定性失败门禁**：对 prefill、solo `run/prelaunch`、batch `run_slot_rows/batch_launch`、pipeline `pl_launch` 分别设置 `STRATA_TEST_KV_POOL_FAIL_AFTER`，断言 reservation rollback、launch counter 不增加、首个 KV write 未发生；只验证返回 ERR 不足以证明没有启动 graph。另加 split next-stage `prelaunch()` 跨新页用例，以及同 pipeline group 内“cached idle slot 位于 active slot 之前”的 pad-row 用例。
- **排查**：打印 lease owner/generation 和 free 守恒；若复用后错而 parking 正常，重点查 `slot_source/cached`，不要先查 attention。

### 步骤 6：INFO、服务指标和配置拒绝

- **改动**：动态 INFO、server 锁、Prometheus 真实水位及单测；`--kv-pool-mib` 启动预算日志。
- **验证**：
  ```sh
  python3 -m pytest serve/test_prometheus.py serve/test_parallel.py serve/test_admit_order.py \
    serve/test_slot_alloc.py serve/early_close_test.py -q
  curl -fsS -H 'Accept: text/plain' http://127.0.0.1:8080/metrics | grep 'vllm:kv_cache_usage_perc'
  curl -fsS http://127.0.0.1:8080/metrics | python3 -m json.tool | grep -E 'kv_pool_(pages|free)'
  ```
- **预期**：开关开时 gauge 与 INFO 一致，请求结束按真实 lease 回落；开关关/旧引擎仍使用旧公式；`kv_resident`、`kv_grow`、`STRATA_BATCH_ATTN_ROWS` 组合均 fail fast。
- **排查**：运行时 INFO 不更新就查 `_pump()` 是否仍只当普通控制行；多线程偶发旧值查 `Engine.info` 快照锁；指标异常先验证 free 范围，不在 Python 静默 clamp 真实错误。

### 步骤 7：默认路径和全功能回归

- **改动**：只修回归，不扩大范围。
- **验证**：
  ```sh
  cmake --build build-hip -j2
  ctest --test-dir build-hip --output-on-failure
  python3 -m pytest serve -q
  python3 tools/batch_test.py --exe "$EXE" --config "$CFG" --batch 8 --n 8 \
    --extra '--pcie-frac 0 --adapt-every 1000000' --dump /tmp/kv_off.json
  python3 tools/batch_test.py --exe "$EXE" --config "$CFG" --batch 8 --n 8 \
    --extra '--pcie-frac 0 --adapt-every 1000000 --kv-pool --kv-pool-mib 3072' --dump /tmp/kv_on.json
  python3 - <<'PY'
  import json
  a=json.load(open('/tmp/kv_off.json')); b=json.load(open('/tmp/kv_on.json'))
  assert a == b, 'solo/batch token arrays differ'
  print('PASS: all solo and batch token arrays identical')
  PY
  ```
- **预期**：ctest 和 serve 全绿；off/on 的 solo、batch 数组完全一致；关闭开关的六条回答 SHA-256 与同日基线一致。
- **排查**：先用 dump 找 `solo/batch + slot + token index`；only-batch 错查页 lease 和行映射；only-restore 错查逻辑页拷贝；off 也错则立即回退该步，不能归因于 pool。

### 步骤 8：性能、容量和 HIP 双卡验收

- **改动**：只允许在测量证明需要时调整 8 页批量拷贝排程，不改变格式/调度。
- **验证**：执行 §5 的容量、TTFT/TPOT、快照延迟三组 A/B；双 7900 XTX 用 `--backend hip --gpus 0,1` 对应的现有 config，并加 `--trim-stage-weights --batch 8`。
- **预期**：满足 §7 的全部 5 条验收；保存原始 JSON、引擎日志、硬件/构建版本，不在文档中预填结果。
- **排查**：TTFT 退化看 chunk 页表 H2D 和 acquire memset；TPOT 退化看是否误在每层/每 token 同步；快照退化用分段计时区分页表快照、D2H/H2D、sync；显存无收益查池是否在专家 cache 之后分配。

### 步骤 9：后端构建矩阵与最终门禁

- **验证**：
  ```sh
  # HIP（gfx1100 示例）
  cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100
  cmake --build build-hip -j2 && ctest --test-dir build-hip --output-on-failure
  # CUDA（新目录，不覆盖 HIP）
  cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF
  cmake --build build-cuda -j2 && ctest --test-dir build-cuda --output-on-failure
  # SYCL 独立后端：至少编译确认默认关闭不受影响；不支持 --kv-pool 时应明确拒绝。
  cmake -S . -B build-sycl -DSTRATA_ENABLE_SYCL=ON \
    -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF
  cmake --build build-sycl -j2
  python3 -m pytest serve -q
  ```
- **预期**：所有触及的 backend 构建通过；不支持后端默认路径不变；最终 git diff 只含本计划列出的实现/测试文件。
- **排查**：后端特定失败先确认 pool 源文件是否错误依赖 CUDA-only API；HIP 映射可复用现有 runtime 抽象；Metal/SYCL 不应因未启用功能链接新 runtime。

## 5. 对照 §7 的测试计划与可执行命令

### 5.1 greedy 开/关逐 token 一致

主门禁用步骤 7 的 `batch_test.py --dump` 加 JSON 全等断言；再跑：

```sh
python3 tools/batch_interleave_test.py --exe "$EXE" --config "$CFG" --flush 8 \
  --extra '--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --kv-pool --kv-pool-mib 3072'
python3 tools/parking_test.py --exe "$EXE" --config "$CFG" \
  --extra '--pcie-frac 0 --adapt-every 1000000 --kv-pool --kv-pool-mib 3072'
STRATA_SLOT_COPY_DIRECT=1 python3 tools/batch_interleave_test.py --exe "$EXE" --config "$CFG" --flush 8 \
  --extra '--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow --kv-pool --kv-pool-mib 3072'
```

四个 KV 格式分别重复 `--kv fp16|int8|q4_0|k8v4`。期望所有脚本报 IDENTICAL，快照指纹不随物理置换改变。

### 5.2 池为现状 1/4 时，8 槽 ≤8K 不排队/拒绝

INT8 下设计给定基线约为 `8×0.39 GiB=3.12 GiB/stage`，1/4 为 `0.78 GiB/stage`，取 `--kv-pool-mib 800`（800 MiB，略大于估算）；正式验收还要用启动日志中的实际页价计算，不把 0.39 当精确常量。

```sh
python3 tools/kv_pool_stress.py --exe "$EXE" --config "$CFG" --slots 8 --rounds 16 \
  --min-tokens 1024 --max-tokens 8192 --output /tmp/kv_pool_800.json \
  --extra '--kv int8 --kv-pool --kv-pool-mib 800 --pcie-frac 0 --adapt-every 1000000'
python3 - <<'PY'
import json
r=json.load(open('/tmp/kv_pool_800.json'))
assert r['slots']==8 and r['rounds']==16
assert r['errors']==[] and r['pool_exhausted']==0 and r['queued_due_to_pool']==0
assert r['max_total_cells']<=8192 and r['min_free_pages']>=0
assert r['nonidentity_allocations']>0 and r['physical_page_reuses']>0
assert r['free_accounting_balanced']
print('PASS: 8 slots <=8K total cells, no pool queue/refusal; high_water=',r['high_water_pages'])
PY
```

等池容专家收益用两个独立进程的启动 INFO 比较 `expert_slots/expert_cache_mib`：off 臂按相同 800 MiB 总 KV 预算构造，on 臂保持其他配置相同；记录实际驻留专家数，不预写“必然增加 1600”。

### 5.3 单流 TTFT/TPOT 回归 ≤2%

使用同一进程模型、同一 prompt、greedy、禁止 conversation reuse；两臂分别冷启动，交替顺序做至少 12 次，保存首 token wall time 和 DONE 的 `prompt_ms/decode_ms`。下面驱动可直接运行：

```sh
for arm in off on; do
  EXTRA='--pcie-frac 0 --adapt-every 1000000 --prompt-cache 0'
  [ "$arm" = on ] && EXTRA="$EXTRA --kv-pool --kv-pool-mib 800"
  EXE="$EXE" CFG="$CFG" EXTRA="$EXTRA" ARM="$arm" python3 - <<'PY'
import json, os, time
from tools.batch_test import Engine, tokenizer
cfg=json.load(open(os.environ['CFG'])); tok=tokenizer(cfg['tokenizer'])
e=Engine(os.environ['EXE'],cfg,0,{'STRATA_IQ_MT_MIN':'1'},os.environ['EXTRA'].split())
out=e.lines(); ids=tok.encode('<|im_start|>user\nExplain cache locality in detail.<|im_end|>\n<|im_start|>assistant\n',parse_special=True)
rows=[]
for i in range(12):
    e.send('GEN 128 ckpt=0 '+','.join(map(str,ids)))
    t0=time.perf_counter(); first=None; toks=[]
    for line in out:
        if line.startswith('T '):
            if first is None: first=time.perf_counter()
            toks.append(int(line.split()[1]))
        elif line.startswith('DONE '):
            end=time.perf_counter(); f=line.split()
            rows.append({'ttft_s':first-t0,'tpot_s':(end-first)/(len(toks)-1),
                         'prompt_ms':float(f[3]),'decode_ms':float(f[4]),'ids':toks})
            break
        elif line.startswith('ERR'): raise RuntimeError(line)
e.send('QUIT'); e.p.wait(timeout=180)
json.dump(rows,open('/tmp/kv_latency_'+os.environ['ARM']+'.json','w'))
PY
done
python3 - <<'PY'
import json,statistics
A=json.load(open('/tmp/kv_latency_off.json')); B=json.load(open('/tmp/kv_latency_on.json'))
assert [r['ids'] for r in A]==[r['ids'] for r in B]
for key in ('ttft_s','tpot_s'):
    a=statistics.median(r[key] for r in A); b=statistics.median(r[key] for r in B)
    print(key,'off=',a,'on=',b,'ratio=',b/a)
    assert b<=a*1.02, key+' regression exceeds 2%'
print('PASS: single-stream TTFT/TPOT within 2%')
PY
```

同时报告 p50/p90 和各次样本；若机器噪声使 medians 接近门槛，增加同日交替样本，不能只挑较快一次。

### 5.4 快照保存/恢复与停泊往返延迟 ≤10%

```sh
for arm in off on; do
  EXTRA='--conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0 --adapt-every 1000000'
  [ "$arm" = on ] && EXTRA="$EXTRA --kv-pool --kv-pool-mib 800"
  for i in 1 2 3 4 5; do
    BATCH_TEST_LOG="/tmp/parking_${arm}_${i}.log" python3 tools/parking_test.py \
      --exe "$EXE" --config "$CFG" --extra "$EXTRA" > "/tmp/parking_${arm}_${i}.out"
  done
done
python3 - <<'PY'
import re,statistics
D={}
for arm in ('off','on'):
    saves=[]; restores=[]
    for i in range(1,6):
        text=open(f'/tmp/parking_{arm}_{i}.log').read()
        saves += [float(x) for x in re.findall(r'parked \d+ tokens in ([0-9.]+) ms',text)]
        restores += [float(x) for x in re.findall(r'restored \d+ tokens \([^)]*\) in ([0-9.]+) ms',text)]
    assert saves and restores, (arm,'no real park/restore observed')
    D[arm]=(statistics.median(saves),statistics.median(restores))
for j,name in enumerate(('save','restore')):
    a=D['off'][j]; b=D['on'][j]
    print(name,'off_ms=',a,'on_ms=',b,'ratio=',b/a)
    assert b<=a*1.10, name+' regression exceeds 10%'
print('PASS: snapshot save/restore latency within 10%; parking round trips already token-compared')
PY
```

`parking_test.py` 自身必须成功且确有 `restored`/reused KV；若实际正则因日志文本调整失配，应修测试解析而不是把缺失数据当 0 ms。

### 5.5 ctest + serve 全绿

```sh
ctest --test-dir build-cuda --output-on-failure
ctest --test-dir build-hip --output-on-failure
python3 -m pytest serve -q
python3 -m pytest serve/test_prometheus.py serve/test_parallel.py serve/test_admit_order.py \
  serve/test_slot_alloc.py serve/early_close_test.py -q
```

此外保留 `batch_test.py`、`batch_interleave_test.py`、`parking_test.py`、`conversation_restore_speed.py` 四族 GPU 回归。任何 skip 都要列出原因，不能把 SKIP 当功能验证通过。

## 6. 必须写进代码的断言与不变量

以下是 debug/test 必开断言；会影响安全而非仅性能的正式路径必须同时 fail closed，不能只依赖 `assert` 被 Release 编译掉。

```cpp
// 所有权/格式
assert(!st.kv_pool || (st.kv_mode == 0 && st.kv_elastic < 0));
assert(!st.kv_pool || main_qsa_state);                  // drafter 永不入池
assert(!st.kv_pool || st.page_table != nullptr);
assert(pool.free_pages() >= 0 && pool.free_pages() <= pool.pages());
assert(pool.allocated_pages() + pool.free_pages() == pool.pages());
assert(no_physical_page_appears_in_two_live_leases());
assert(host_mirror.size() == st.n_pages);

// 地址和 graph
assert(st.k_pool == pool.fixed_k_base());               // 或当前格式对应 base
assert(st.page_table == captured_page_table_pointer);
assert(pool.base_addresses_unchanged_since_create());
assert(graph_capture_count_unchanged_after_table_content_update());

// launch 前容量：所有可能写到的 cell 均有有效物理页
for (row in window_rows)
    assert(pool.covers(row.state, row.pos + 1));
assert(pool.ensure_completed_before_window_launch);
assert(no_structural_page_table_change_while_window_in_flight);
assert(page_table_upload_is_ordered_before_kernel_on_same_stream());

// 生命周期
assert(!window_in_flight && !snapshot_read_in_flight && !copy_queued_before_release);
assert(!slot.cached_device_live || slot.lease != nullptr);
assert(old_lease_released_only_after_replacement_scatter_success);
assert(no ensure/allocation during graph capture);       // capture fixture 单独处理

// 快照/文件
assert(snapshot.page_size == shapes.page_size);
assert(logical_page < st.n_pages && 0 <= physical_page && physical_page < pool.pages());
assert(indexer_pooled_copy_does_not_use_kv_page_table);
assert(fingerprint_independent_of_physical_permutation);
```

额外的工程约束：

1. **页表内容变化不重捕获 graph。** 若实现需要更新 pool base 或 page-table pointer 才能继续，说明设计/实现错误，必须停下修复，不能用重捕获掩盖。
2. **窗口内禁止结构性变更。** 已存在映射的内容由 kernel append 正常更新；host 不得同时改页号、free 或 replace lease。
3. **committed 与 write-top 分开。** MTP/pipeline 的未接受行可以占尾页，但不能推进 committed 前缀；下轮重写相同逻辑页是合法的。
4. **默认路径零新输出。** 不开启 `--kv-pool` 时无新 INFO、无 pool 分配、无 gather/scatter 分支、无指标语义变化。
5. **池耗尽不越界。** 任一 stage/任一层/任一部件取页失败，整个 reservation 回滚，窗口或 chunk 不启动。

## 7. 完成定义（DoD）

- 9 步均有对应提交和验证记录；off/on 默认路径和 greedy 对比全等。
- §5 五项验收全部有原始日志；性能数字注明硬件、后端、模型、配置和同日基线。
- CUDA/HIP 构建与测试通过，SYCL 默认构建未受影响；未运行的平台明确标为未验证。
- `STRATA_BATCH_ATTN_ROWS=1`、`--kv-grow`、`--kv-resident` 的互斥均有自动测试。
- 非恒等页表覆盖 prefill、solo、batch、pipeline、快照、direct copy、verify 和 SAVE；无“碰巧 identity”假阳性。
- 没有任何调度器、抢占、前缀共享、kv_mode 1 池化或 >8 行 kernel 改动混入阶段 1。
