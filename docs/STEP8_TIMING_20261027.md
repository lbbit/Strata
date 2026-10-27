# 步骤 8 TTFT/TPOT 计时 A/B（2026-10-27）

工具：`tools/ab_engine.py`（同 config，A 臂无 `--kv-pool`，B 臂加 `--kv-pool`；每轮各重启一次，交替顺序；
greedy 固定 prompt；比较 server 自报的 decode/prompt tok/s 的 medians）。真实模型 ud-iq4_xs，batch 单流，int8。

## 8192 context（--kv-pool-mib 800）— medians over 2 rounds
| 请求 | A_off decode | B_on decode | Δdecode | A_off prompt | B_on prompt | Δprompt |
|---|---|---|---|---|---|---|
| chat0 | 30.2 | 27.2 | **-10%** | 30.0 | 28.9 | -4% |
| chat1 | 31.8 | 28.8 | **-9%** | 28.3 | 23.9 | -16% |
| chat2 | 30.9 | 29.1 | **-6%** | 32.2 | 25.6 | -20% |
| long  | 35.0 | 32.1 | **-8%** | 877 | 767 | -13% |

**结论（8192）**：小 context 下 `--kv-pool` 的 decode 慢 ~6-10%、prompt 慢 ~13-20%。原因：小 context 时池化
省不出多少显存（专家驻留仅 +0.57%，见 STEP8_BENEFIT），而池的页管理开销（每 chunk ensure 的页表 H2D +
acquire memset）是净增加。**该场景不满足"TTFT/TPOT 回归 ≤2%"的验收目标。**

## 关键判断
池化的性能收益依赖 **大 context**：只有满配 KV arena 显著吃掉显存、专家缓存明显受限时，`+20% expert_slots`
（STEP8_BENEFIT @32768）才能抵消页管理开销。因此：
- **小 context（≤8192）：不建议开 --kv-pool**（默认满配反而更快）。
- **大 context：需实测 TTFT/TPOT 是否转正**（见下 32768 A/B）。

## 32768 context（--kv-pool-mib 3072）— medians over 2 rounds
| 请求 | A_off decode | B_on decode | Δdecode | A_off prompt | B_on prompt | Δprompt |
|---|---|---|---|---|---|---|
| chat0 | 29.8 | 26.5 | **-11%** | 28.8 | 25.0 | -13% |
| chat1 | 33.2 | 27.2 | **-18%** | 29.1 | 25.2 | -13% |
| chat2 | 29.8 | 28.5 | **-4%** | 27.8 | 27.6 | -1% |
| long  | 36.7 | 29.4 | **-20%** | 819 | 808 | -1% |

## ⚠️ 诚实结论：性能净负，未达验收目标
即便在 **32768 大 context**（专家驻留 +20%，见 STEP8_BENEFIT），`--kv-pool` 的 decode 仍慢 4-20%、prompt 慢 0-13%，
**没有转正**。即"专家多驻留"的收益被**当前池实现的页管理开销完全抵消甚至反超**。

根因（对应计划排查指引"TTFT 退化看 chunk 页表 H2D 和 acquire memset"）：每次 ensure 新页都做
`cudaMemsetAsync` 清零 + 页表 `cudaMemcpyAsync` H2D，**每 chunk、每层、每 token 都可能触发**，是纯开销；
而 decode 瓶颈未必在专家命中率，故 +20% expert_slots 未能换来 decode 加速。

## 建议（不粉饰）
1. **当前形态下 `--kv-pool` 不适合默认开启**：默认满配 KV 更快。保持 opt-in、默认关闭。
2. 要让它有性能收益，需优化 ensure 路径（计划步骤 8 允许的最小改动范围内）：
   - 页表 H2D：只上传**变化的页**（增量），而非整表；或用 pinned staging + 事件排序。
   - acquire memset：仅对**首次**分配页清零一次（页复用时本已被上一会话写过，是否需要零取决于是否读到未初始化；
     若不要求零可省）。评估去掉每 chunk 清零的可行性。
   - 批量：一次 ensure 合并更多 chunk 的页，减少同步次数。
3. 这些优化超出当前"实现正确"的范围，属后续性能专项；本次步骤先把功能与正确性做扎实、如实记录性能现状。

