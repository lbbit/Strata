# 步骤 5 进展（2026-10-27 下午，进行中）

## 一句话状态
步骤 5（请求/slot 生命周期回收 + 池耗尽保护）**进行中，未提交**。已完成 `GlobalKvPool::release()` 到 `SessionState` 销毁点的接线（generate.cpp +~50 行），并**用真实模型 + 小池（512MiB）实测证伪了"顺序对话会泄漏"的预判**——30 请求/29 次 parking，`free/high_water` 恒定不变。剩余：BDONE/换主/eviction 的 lease 回收、耗尽保护、`tools/kv_pool_stress.py`、`STRATA_TEST_KV_POOL_FAIL_AFTER` 门禁。

## 当前代码改动（未提交，工作区 generate.cpp）
- 新增 `release_slot_session(SessionState&, stage, slot)` helper：同步后逐 QSA `kv_pool->release()`，`STRATA_KV_POOL_DEBUG=1` 打印释放页数。
- 新增 `shrink_slot_sessions(keep)` / `clear_slot_sessions()`：在 `bslot_ss` 启动 sizing 缩减/清空**销毁 SessionState 前**先 release（否则 lease 表留悬空 QSA 指针 → 崩）。
- 替换了原来的 `v.resize(fit)` / `bslot_ss.clear()` / `v.resize(n)` 三处。
- 新增 `POOLWATCH` 诊断行（park 后打印 free/high_water/parked/evictions）——**验证用，最终需删除或降级到 DEBUG**。
- **顺序单会话（--kv-pool-mib 512）**：30 个独立请求、29 次 parking，POOLWATCH `free=126872 high_water=228` **从头到尾恒定** → 无泄漏。
- **batch 8 并发（--batch 8 --kv-pool-mib 512）**：40 个并发请求、37 次 parking，high_water `0→1164→1864→1924` 后**平台化在 1924 恒定**（后 20+ 次 park 不再涨）。1924 ≈ 8 slots × 240 页 = 8 个并发 slot 各自正常占用量，与顺序 228≈1×240 吻合 → **也不是泄漏，是填满 8 个 slot lease 后的稳态**。
- 结论：**顺序与 batch8 场景池均不泄漏**，`ensure(pos0,pos0+T)` 按会话位置分配、lease 复用同一 slot 时 committed 语义稳定。lease 按 slot 存活、稳态后不增。
- 意味：步骤 5 的"release() 从不被调用"是**隐患**（slot 销毁时若不在销毁前 release 会留悬空指针，本轮已修），但**不是活跃泄漏源**。真正要验证的是：**池耗尽时优雅报错不崩**、以及 **slot 销毁/换主路径的 lease 正确回收**（本轮 shrink/clear 已接）。

## 2026-10-27 下午补记：修掉「池耗尽即崩溃」BUG（本轮关键成果）
- 现象：4MiB 极小池 + batch8 发 8 个长 prompt → 之前引擎 **exit(1) 崩溃**，客户端全部失败。
- 根因：`generate.cpp` prompt 读阶段 `if (!sp_ok)` 分支对任何失败都 `printf("ERR")` 后**无条件 `return 1`**（从 serve 主循环返回 → 进程退出）。只有 `cudaPeekAtLastError()!=success`（真 CUDA 故障）才应 `_Exit`。
- 修复：识别 `err` 含 `kv_pool`（池耗尽）且无 CUDA 错误时 → `cancelled=true; break`（报 ERR、结束当前请求、refill、继续服务下一个请求）；其余维持原逻辑。
- 实测（修复后，4MiB 池 + batch8，8 长 prompt）：**3 成功 / 5 优雅报 `insufficient pages`（invalid_request_error），引擎保持 `loaded:true` 存活**。默认回归全绿（file201/val972/snap4286/kvp1.8M）。
- 说明：tiny 4MiB 是"耗尽演示"配置——8 个 slot 稳态需 ~1800 页 > 992 页，故满载后新请求合理报 ERR。真实部署池容量须覆盖 batch×slot 稳态（计划步骤6/8 要按此设默认 `--kv-pool-mib`）。
- 当前工作区改动：`src/program/generate.cpp` +62/−3（release 接线 + 耗尽不崩），**未提交**。

## 仍未做（步骤5 真正要覆盖的场景）
1. **多 slot 并发 + slot 换主**（`--slots 8`）：`copy_to_slot` 用 `conversation_kv_copy` 把主会话拷进 slot（`into_slot`），旧 lease committed 只增——**这个场景才可能暴露真泄漏**，需用 doc 的 `tools/kv_pool_stress.py --slots 8 --rounds 16` 压。
2. **池耗尽**：极小 `--kv-pool-mib` 造 `ensure` 失败，断言第一个 KV write 前返回 ERR、不崩、不 illegal memory access。
3. **`STRATA_TEST_KV_POOL_FAIL_AFTER`** 确定性失败门禁（prefill/solo/batch/pipeline 分别断言 reservation rollback、launch counter 不增、首 KV write 未发生）。
4. **BYIELD/QUIT/restore 失败/slot 换主** 的 lease 回收。
5. `tools/kv_pool_stress.py`（参照 `tools/parking_test.py` + `batch_test.Engine`）。
6. 清理 POOLWATCH 诊断。

## 环境/方法备忘
- dev 配置：`/tmp/strata-dev-kvpool.json`（dev 二进制+真实 iq4u+`--kv-pool`+`--spec 4 --spec-min-p 0.5 --mtp .../mtp-rt --gpu 0,1 --max-context 8192`）；小池版 `/tmp/strata-dev-small.json`（`--kv-pool-mib 512`，port 18081，含 STRATA_KV_POOL_DEBUG=1）。
- 起服务：`cd /home/lbb/code/Strata_vllm && setsid python3 -m serve.server --engine strata --config <json> --port <p> >... 2>&1 </dev/null & disown`，等 `/health` `loaded:true`（约 60s）。
- 停服务：按 PID `kill -9`（**别 pkill 含配置名的串，会误杀自己 bash**）；**测完必须 `sudo systemctl start qwen38-llm.service` 恢复生产**（生产占 GPU0,1，测前先 stop 腾显存）。
- 单实例串行，别并发压 GPU（曾致 GPU0 hang 需重启）。
- 编译：`cd build-hip && ninja strata -j2`（约 60-90s）。
- Copilot 会话 id：`c64f5c5b-678f-491c-be0b-d642a436a9cf`（deepseek-v4.1-flash；它面对大任务倾向先长篇规划就停，需拆成"只做一件事+必须动手"的小指令，且多进程会互踩文件）。
