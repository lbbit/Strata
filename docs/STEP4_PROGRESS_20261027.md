# 步骤 4 进展存档（2026-10-27 重启前）

## 一句话状态
步骤 4（池化后快照/直拷/verify/SAVE 统一逻辑页 gather/scatter）**代码已完成、全部测试门禁通过**（file 201 / validation 972 / snapshot 4286×22连跑0失败 / kv_pool 1.8M / strata 0.1.48.1），**尚未提交**。唯一未做的是"真实模型 + --kv-pool 两轮对话实证"。**重启原因是我操作失误把 GPU0 驱动搞 hung，与代码无关。**

---

## 一、代码改动（已持久化在 /home/lbb/code/Strata_vllm 工作区，未 commit）
- 分支 `main`，HEAD `d8b66279`（步骤 3 的提交）
- `git diff --stat`：5 files, +541 −39
  - `include/strata/core/conversation_snapshot.hpp` +5
  - `src/core/conversation_copy.hpp` +53
  - `src/core/conversation_snapshot.cpp` +266
  - `src/core/conversation_snapshot_test.cpp` +239
  - `src/core/conversation_state.cpp` +17
- **备份 patch**：`/home/lbb/step4_wip_20261027_112830.patch`（762 行）
- 回滚/恢复：`cd /home/lbb/code/Strata_vllm && git checkout -- .`（丢弃）或 `git apply /home/lbb/step4_wip_*.patch`（重放）

## 二、本轮修掉的两个真 BUG

### BUG 1：page_bytes 少乘页粒度（生产代码，关键）
文件 `src/core/conversation_snapshot.cpp` 的 `page_view()`：
```cpp
// 错（每 cell 字节）：
view.page_bytes[i] = cells ? sizes[i] / cells : 0;                    // 73728/72 = 1024
// 对（每物理页字节，copy_logical_range 的契约）：
view.page_bytes[i] = cells ? (sizes[i] / cells) * (size_t) l.page_size : 0;  // 1024*4 = 4096
```
原因：`copy_logical_range()` 用 `logical = at/page_bytes` 和 `physical + table[logical]*page_bytes` 寻址，第二参数必须是整页字节。原来让每个池化页被当 4 页寻址，gather/scatter 全错位。**默认 BF16 下错误值 1024、正确值 4096**——这正是 Copilot 卡 50 分钟的谜题答案。**此公式不要再动。**

### BUG 2：publish 与 scatter 跨流竞态
`GlobalKvPool::publish_prepared()→publish()` 在流上做 `cudaMemsetAsync`（清零新物理页）+ `cudaMemcpyAsync`（上传页表）；snapshot scatter 走私有 non-blocking 流，两者默认不互序 → scatter 可能读到旧页表/写进正被清零的页。症状：`restore roundtrip` 约 50% 失败，`fmt=1(INT8)` 的 K 从第 12 页起错。
**修复**：在 `conversation_kv_restore_many` 与 `conversation_kv_copy` 两处，publish 之后、scatter 之前加 `cudaDeviceSynchronize()`（带中文注释）。
**证据**：加 device sync 后 10/10 通过；改 stream sync 只 2/12 通过（说明还有别的流参与）；最终 device sync 后 **22 连跑 0 失败**。
（restore/copy 都是 checkpoint 加载、非热路径，device 同步代价可接受。）

另一个已修的小 BUG（Copilot 遗留）：`conversation_kv_save()` 增量路径 `copy_pages(..., keep + at, ...)` 应为 `at`（visit 的 `at` 是绝对偏移），全量保存 keep=0 不暴露、增量 keep>0 才错。

## 三、测试门禁结果（GPU0 出事前、修复后实测）
| 测试 | 结果 |
|---|---|
| conversation_file_test | 201 checks passed |
| conversation_validation_test | 972 host-only checks passed |
| conversation_snapshot_test | 4286 checks；连跑 22 次全 exit=0 |
| kv_pool_test | 1,807,027 checks passed |
| strata --version | 0.1.48.1 |
覆盖：FP16/INT8/Q4/Hybrid 四格式池化 gather/scatter、非恒等映射布局无关性、指纹一致、SAVE source 跨页+跨16MiB段、direct copy(queued true/false)、restore 往返、非页对齐前缀、页内 unchanged_tokens、4096页跨段大用例。

## 四、⚠️ GPU0 事故（我操作失误，需重启恢复）
- 原因：我为测稳定性**并发跑了多个测试循环/ctest** 抢同一张 GPU0 → 驱动层 hang（进程 D 状态，栈 `amdgpu_info_ioctl`，kill -9 无效）。
- 处理链：
  1. `sudo rocm-smi --gpureset --device 0` → "reset succeeded"
  2. 但僵尸驱不掉 → `unbind 0000:85:00.0` 成功
  3. **rebind 失败**，内核报 `amdgpu: Fatal error during GPU init` / `probe of 0000:85:00.0 failed with error -17` / `Oops: 0002`
  4. 现状：**GPU0 unbound + 内核 Oops，需重启恢复**
- **生产未受影响**：`qwen38-llm.service` 一直 active，`/health` = `{"status":"ok","max_context":262144,...}`，生产在 **GPU1**。
- 重启后：`rocm-smi` 应能重新枚举 GPU0（通常自愈；若仍 -17 可能需断电冷启）。
- 重启后清理：可能有若干 D 状态僵尸 `conversation_snapshot_test`/`sh -c echo ...bind`，重启后自然消失。

## 五、重启后的待办（按顺序）
1. 确认 `rocm-smi --showmeminfo vram` 能看到 GPU[0] 与 GPU[1]。
2. 单实例（**绝不并发**）跑一次：`cd build-hip && HIP_VISIBLE_DEVICES=0 ./kv_pool_test`，再 `HIP_VISIBLE_DEVICES=0 ./conversation_snapshot_test`，确认修复在干净环境稳定。build 目录二进制已是修复版（11:03-11:06 构建），无需重编；若重编用 `ninja conversation_snapshot_test conversation_validation_test conversation_file_test strata kv_pool_test -j2`。
3. **做真实模型 + --kv-pool 两轮对话实证**（步骤 4 唯一缺口）：真实模型 + `--kv-pool` 起服务，两轮同会话对话，确认第二轮连贯正确、日志无 snapshot 错误。⚠️ 测前 `sudo systemctl stop qwen38-llm.service`，**测完无论成败都要 `sudo systemctl start qwen38-llm.service`**；单卡 `HIP_VISIBLE_DEVICES`；`cd /home/lbb/code/Strata_vllm` 才能 `python -m serve.server`；`--port` 命令行给；`--no-api-key` 不进引擎 args；`--spec` 是 native pack 必需别删。
4. 提交步骤 4（`git diff` 已就绪）。
5. 进步骤 5。**步骤 5 最大风险预判**：多会话/多租户共享池时的 lease 生命周期与"第 N 层 prepare 耗尽回滚"的一致性；以及 pool 未启用 vs 启用的路径分叉是否会在生产默认路径上被意外触发（务必保默认 byte-identical）。

## 六、协作/环境注意（教训）
- deepseek-v4.1-flash **支持工具调用**（已验证），但抗 503 差：续接常被中断并留半成品/回退。每次续接前先 `git diff`+编译+跑测试确认基线。
- CLI 调用：`bash /home/lbb/qwen3.8-27b-project/scripts/copilot-run.sh --cwd /home/lbb/code/Strata_vllm --resume c64f5c5b-678f-491c-be0b-d642a436a9cf -f <prompt>`（会话 id 见下）。当前 `~/.copilot/byok.env` 模型=deepseek-v4.1-flash；原 kimi 配置备份在 `~/.copilot/byok.env.bak.kimi`。
- **稳定性测试必须单实例串行**，这是我这次翻车的直接原因。
- 端点偶发 503，curl 直连探测：`https://tmp.mysteps.top/v1`。
- **绝不动 /home/lbb/strata（生产）。**

## 七、上一轮 Copilot 会话 id
`c64f5c5b-678f-491c-be0b-d642a436a9cf`（步骤 4 全程续接此会话）。
