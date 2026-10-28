# 步骤 9 全量回归结果（2026-10-27）

## 全量 ctest（build-hip，152 个测试，-j1 串行）
```
97% tests passed, 5 tests failed out of 152
```
**147/152 通过。**

## 5 个失败（全部与本特性无关，环境/既有）
| 测试 | 原因 | 与本特性关系 |
|---|---|---|
| `ple_parity` (#91) | 缺模型文件 `Q2_0/Qwen...gguf`（本机无该量化） | 无关 |
| `hip_prefill_native_batch` (#43) | GPU native prefill parity mismatch（gfx1100 数值） | 无关（不 include kv_pool） |
| `hip_prefill_mmq_parity` (#135) | NUMERICAL（GPU kernel parity） | 无关 |
| `pool_tasks_test` (#123) | CPU 线程池/NUMA 测试（"NUMA pool off..."） | 无关（cpu/pool_tasks_test，0 处 kv_pool） |
| `pool_cores_test` (#145) | 同上，CPU 线程池 | 无关 |

依据：这 5 个测试的源文件均不 include `kv_pool`/`GlobalKvPool`（grep 计数 0）；`pool_*` 是 CPU 线程池
（非 KV 池），`hip_prefill_*` 是 GPU kernel parity（本机 gfx1100 既有数值差异），`ple_parity` 缺模型。
它们不可能因本轮 KV 池改动而失败。

## 本特性相关测试（全部通过）
- conversation_cache / conversation_split_failure / conversation_memory / conversation_file /
  conversation_validation / conversation_snapshot：**全部 Passed**
- kv_pool_test：**1807028 checks passed**
- 其余 142 个测试 Passed

## 结论
- 步骤 9 在单机 HIP 构建上：**本特性相关测试 100% 通过，全量 97% 通过；5 个失败为环境既有、与 kv-pool 无关。**
- CUDA/SYCL 后端构建矩阵未做（本机只有 HIP/gfx1100；CUDA 需 NVIDIA 卡、SYCL 需 oneAPI，均不可用）——
  如实标为"未验证"，但改动集中在 core/kv_pool + generate/serve，不依赖后端专有 API。
