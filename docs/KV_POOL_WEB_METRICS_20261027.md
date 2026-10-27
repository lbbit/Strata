# KV 池数据接入 Web/Metrics（2026-10-27）

## 需求
`--kv-pool` 的池化数据（总页数、空闲页数）需要能在 web 页面 / metrics 查到，且符合既有外部风格，不破坏原样式，尽量简洁。

## 实现（三处，最小改动）
1. **引擎 INFO（src/program/generate.cpp）**：INFO 行尾部追加
   ` kv_pool_pages=<N> kv_pool_free=<M>`，**仅当 `--kv-pool` 时**输出。
   - 用外层 `std::string info_tail` 持有（避免先前 lambda 局部 `c_str()` 的悬空指针 bug）。
   - 无 `--kv-pool` 时 INFO 尾与原版逐字节一致（`...prefill_pipe=0 engine=0.1.48.1`）。
2. **serve/server.py `prometheus()`**：当 engine.info 含 `kv_pool_pages` 时，追加两个 gauge
   `strata:kv_pool_pages` / `strata:kv_pool_free`（沿用既有 `(name,kind,help,value)` rows 风格，
   无池时完全不输出）。`/metrics`(JSON) 的 `engine` 字段自动携带这两个键（无需改结构）。
3. **serve/web/app.js `renderAbout`**：在 `["KV cache", ...]` 行后加一行
   `["KV pool", "<in-use> / <total> pages in use (P%), shared logical-page pool"]`，
   仅当 `eng.kv_pool_pages` 存在时显示（否则 null，facts() 自动跳过）——与既有行同风格。

## 实测（真实模型 ud-iq4_xs + --kv-pool-mib 800）
- 引擎 INFO：`... batch_slots=8 slot_cache=1 batch_groups=1 kv_pool_pages=198593 kv_pool_free=198593 engine=0.1.48.1`（无乱码）
- 无池 INFO 尾：`... prefill_pipe=0 engine=0.1.48.1`（无 kv_pool，默认不变）
- GET /metrics：`engine.kv_pool_pages=2552180, kv_pool_free=2552180`
- GET /metrics/prometheus：
  `strata:kv_pool_pages 2552180` / `strata:kv_pool_free 2552180`（含 HELP/TYPE）
- GET / 和 /web/app.js 均 HTTP 200
- serve.test_prometheus unittest：6/6 OK（格式未破坏）
- 默认回归：file 201 / validation 972 / snapshot 4286 / kv_pool 1807028 全绿

## 局限 / 排入步骤 6 的计划
当前 INFO 是**启动时快照**，`/metrics` 的 `engine.kv_pool_free` 不随请求实时刷新
（实测发请求后 free 仍等于启动值）。**动态水位刷新**属步骤 6「INFO/服务指标」范围：
- 方案：步骤 6 让引擎周期性（或按请求结束）重发 INFO 行，或新增轻量 `POOL` 查询指令回一行
  `POOL free=.. high_water=..`；serve 的 `_pump()` 更新 `engine.info` 快照；前端/Monitor 自然刷新。
- 步骤 6 单测：gauge 与 INFO 一致、请求结束按真实 lease 回落、无池/旧引擎仍用旧公式。
