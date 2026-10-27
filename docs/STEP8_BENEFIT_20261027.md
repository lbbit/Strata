# 步骤 8 收益数据（2026-10-27）

## 收益机制
`--kv-pool` 把主 QSA K/V 从"每 slot 按 max_context 满配 arena"改为"按实际用量从共享页池按需分配"，
省下的 VRAM 让给专家缓存（decoder 的 expert cache），减少 PCIe/内存换入换出。

## A/B 实测（真实模型 ud-iq4_xs，单卡 GPU0，batch=8，--kv int8，expert-cache auto）
对比启动 INFO 的 `expert_slots` / `expert_cache_mib`。off 臂 = 无 --kv-pool（KV 满配）；on 臂 = --kv-pool-mib 800。

| context | 配置 | expert_slots | expert_cache_mib | 备注 |
|---|---|---|---|---|
| 8192  | OFF | 6657 | 16902 | KV 本身不大，满配 arena 也不贵 |
| 8192  | ON  | 6695 | 16999 | +38 slots / +97 MiB (+0.57%) |
| **32768** | **OFF** | **5444** | **13822** | 大 context，满配 KV arena 吃掉大量显存 |
| **32768** | **ON** | **6531** | **16582** | **+1087 slots / +2760 MiB (+20.0%)** |

引擎启动额：`KV pool: capacity_mib=800 page_bytes=4224 pages=198593`（on 臂）。
`vram_free_mib=606` 两臂相同（保留一致）。

## 结论
- 收益方向正确，且**随 context 增大而放大**：8192 时 +0.57%，32768 时 **+20.0% 的专家驻留**（1087 slots / 2.76 GiB）。
- 这与设计预期一致：context 越大，满配 KV arena 越贵，池化按需分配省出的显存越多，让给专家缓存越多。
- 该模型 expert blob 约 2.66 MB，+1087 slots ≈ 2.76 GiB 常驻专家 → 更高 expert cache 命中率 → decode 更快（TPOT 收益，尚未做 TTFT/TPOT 计时，属步骤 8 剩余）。

## 正确性旁证（步骤 7 核心验收，已通过）
同 prompt、greedy(temperature=0)、max-new 24，`batch_test.py --dump`：
**off vs on 的 solo 数组 IDENTICAL、batch 数组 IDENTICAL** → 池化不改变任何输出 token。

## 复现命令
```
EXE=/home/lbb/code/Strata_vllm/build-hip/strata
# off @32768
$EXE --serve --pack /home/lbb/strata/packs/ud-iq4_xs-uncensored \
  --native <shard1.gguf> --ple-gguf <shard1.gguf> \
  --expert-profile /home/lbb/strata/data/expert-profile.bin --expert-cache auto \
  --prefill auto --spec 4 --spec-min-p 0.5 --mtp <mtp-rt> \
  --max-context 32768 --kv int8 --gpu 0 --batch 8
# on @32768: 追加 --kv-pool --kv-pool-mib 800
# 读 INFO 行的 expert_slots= / expert_cache_mib=
```
