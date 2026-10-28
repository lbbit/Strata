# 阶段 1（全局 KV 页池）项目终止总结 — 2026-10-27

## 决定
用户实测后认为效果不佳，**项目到此为止，后续不再推进**，生产继续使用官方原版（不启用 `--kv-pool`）。

## 交付物（已提交到 main，待推送）
9 个步骤全部完成并提交，14+ 个 commit（`e0a4cc46` … `de2708d6`）：

| 步骤 | 内容 | 结论 |
|---|---|---|
| 1 | GlobalKvPool 纯分配器 + 主机单测 | ✅ |
| 2 | QsaState 绑定 + sizing/init 池感知分叉 | ✅ |
| 3 | `--kv-pool` 接入 prefill/decode（修后端宏 bug） | ✅ |
| 4 | 快照/直拷/verify/SAVE 逻辑页 gather/scatter（修 page_bytes 少乘页粒度、publish↔scatter 跨流竞态） | ✅ |
| 5 | lease 回收、**池耗尽不崩溃**（原会 exit(1)）、`STRATA_TEST_KV_POOL_FAIL_AFTER` 门禁、`kv_pool_stress.py` | ✅ |
| 6 | INFO/Prometheus/Monitor 展示 + **动态水位刷新**（POOL 行）+ Monitor 页并发/池占用卡片 | ✅ |
| 7 | off/on token 数组逐位一致（正确性硬证据） | ✅ |
| 8 | 容量收益：context 32768 下 expert_slots **+1087 / +2.76 GiB（+20%）**；但 **TTFT/TPOT 性能净负** | ⚠️ |
| 9 | 全量 ctest 147/152（97%），本特性相关全绿；5 个失败为环境既有 | ✅ |

## 关键结论（诚实记录）
1. **功能与正确性：已充分验证。** 池化不改变任何输出 token（off/on solo/batch 逐位相同）、快照/恢复/直拷正确、池耗尽优雅报错不崩溃、无泄漏（顺序 30 请求、batch8 40 请求 free/high_water 稳态）。
2. **容量收益真实存在**：大 context 下专家缓存多驻留 +20%（2.76 GiB）。
3. **但性能净负**：decode 慢 4–20%、prompt 慢 0–20%（8192 与 32768 均如此）。
   - 根因：`ensure()` 每 QSA 层 × 每 verify 窗口的 O(n_pages) 开销（`Lease candidate = it->second` 整表 2.4MB 拷贝、`free_before = free_` 整空闲表拷贝、整表遍历求 updates、整表 H2D 上传）。
   - 专家多驻留的收益被这些每-token 开销抵消并反超。
4. **优化尝试**：
   - 尝试 1（增量页表上传 + 去整表遍历）：A/B 复测为**负优化**，已回退。
   - 尝试 2（`committed_pages` 增量记账，去掉 free_ 拷贝与整表遍历，O(new pages)）：编译通过、
     kv_pool_test 1807028 / snapshot 4286 / file 201 全绿。**A/B 复测（32768，medians over 2 rounds）**：

     | 请求 | A_off decode | B_on decode | Δ |
     |---|---|---|---|
     | chat0 | 30.2 | 26.8 | -11% |
     | chat1 | 32.6 | 28.0 | -14% |
     | chat2 | 29.8 | 27.0 | -9% |
     | long  | 34.8 | 27.5 | **-21%** |

     **仍净负 ~10-21%**（比尝试 1 略好，但远未转正）。结论：`ensure()` 分配开销不是唯一瓶颈，
     页表 H2D / 每页 memset / 池化 attention 间接寻址本身也有开销。
   - 尝试 2 的改动**留在工作区、未提交**（因项目终止且未达性能目标）；若日后重启，可从此分支继续。

## 给后来者的建议（若日后重启该项目）
- 性能必须先解决 `ensure()` 的每-token O(n_pages) 开销，否则无论容量收益多大，decode 都是净负。
- 尝试 2 的增量记账是正确方向（把分配从 O(context) 降到 O(new pages)）；需完整 A/B 验证 + 默认回归 + 全量 ctest 后再合并。
- 池容量必须覆盖 `batch × slot` 稳态（否则满载新请求合理报 ERR）。
- `--kv-pool` 应保持 opt-in 默认关闭，直到性能转正。
