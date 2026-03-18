# 提交与验证记录

## 2026-03-18 context compile 内存收敛尝试

### 背景
- `QWEN_HLS_ENABLE_EXTRACT=0 make catapult_prefill_attention_context` 当前没有新的 `Error:`，但 `compile` 阶段内存持续升高，已观察到约 `15 GB` 峰值。
- 现有 `ac_channel` 改造主要落在 `q_context_output` tile stream top；当前 `context_stage_catapult` 仍是大数组端口 + query tile 循环，并未真正变成 channel 化 top。
- 更关键的是，`hls/prefill_only/qwen_prefill_attention_kernel.cpp` 是单一大翻译单元，里面同时保留了大量 `#pragma hls_design ccore/top`。从当前 log 看，context flow 在 `compile` 时仍会枚举该文件里很多与 context 无关的 design routine。

### 本轮动作
- 先尝试在 `hls/prefill_only/qwen_prefill_attention_kernel.cpp` 中用条件编译隐藏无关 `hls_design` 边界。
- 验证后确认：Catapult 仍会扫描同一原始源文件里的这些 pragma，单靠条件编译不足以缩小 compile 看到的 design routine 集合。
- 新建 `hls/prefill_only/qwen_prefill_attention_context_stage_catapult.cpp`，作为 context 专用翻译单元；物理上只保留 `qwen_prefill_attention_context_stage_catapult` 自己的 `hls_design` 边界，其余函数退回普通 helper。
- 更新 `script/run_catapult_prefill_attention_context.tcl`，让 context flow 直接编译这个专用翻译单元。

### 当前判断
- `ac_channel` 之前没有显著降低这条 flow 的编译内存，主要不是因为 channel 本身无效，而是因为：
  1. 当前被综合的 top 仍不是 stream top，而是数组型 context stage。
  2. 同一翻译单元内暴露给 Catapult 的 design routine 过多，compile 仍在处理一大批与当前 top 无关的层级边界。
- 3. 从本轮试验看，Catapult 对同一源文件中的 raw pragma 扫描早于我们期望的条件编译裁剪，因此“用宏关掉 pragma”不如“直接切到独立翻译单元”可靠。
- 因此本轮优先收窄“编译面”，而不是继续在同一个大文件里堆更多 channel 包装。

### 2026-03-18 新观测：compile 内存再次失控
- 最新一次 `QWEN_HLS_ENABLE_EXTRACT=0 make catapult_prefill_attention_context` 没有出现新的前端 `Error:`，但 `compile` 内存继续爬升并最终失控。
- 从 `work/tmp/catapult_prefill_attention_context_latest.log` 可见，内存从约 `13.8 GB` 长时间平台期后，在约 `691s` 之后再次上升：
  - `721.90s -> 35284004kB`
  - `901.88s -> 43279396kB`
  - `1081.27s -> 51012644kB`
  - `1111.14s -> 52388900kB`
  - `1200.82s -> 56255524kB`
- 这说明“拆成 context 专用翻译单元”确实减少了早期无关 design routine 干扰，但没有解决 `context` 本体 compile 图在后半段继续膨胀的问题。

### 新增执行期内存守卫
- 新增 `script/run_catapult_with_memory_guard.sh`，由 `make catapult_prefill_attention_context` 调用。
- 守卫同时采样两类指标：
  1. Catapult 进程树 RSS 总和。
  2. Catapult 日志中最近一次打印的 `memory usage ... kB`。
- 默认阈值为 `52428800 kB`，即 `50 GB`；任一指标超阈值则主动终止任务并返回 `148`。
- 监控输出落在 `work/tmp/catapult_prefill_attention_context_monitor.log`，保留时间戳、RSS、日志内存值和 kill 原因，便于后续对齐 compile 曲线。
- 快速验证：使用 `QWEN_HLS_MEMORY_LIMIT_KB=1024 QWEN_HLS_MEMORY_POLL_SEC=1 make catapult_prefill_attention_context`，守卫在首个采样点观测到 `rss_tree_kb=12736` 后主动终止，按预期返回 `148`，说明 kill 机制和监控日志已生效。

### 当前设计与 Catapult 推荐 coding style 的差异
- 参考 `Mgc_home/shared/examples/design_partitioning/ccore/ccore_flow.cpp` 和 `.../ScopeBasedCCORE/top.cpp`，官方示例里的 `ccore` 都满足两个特征：
  1. 单个 `ccore` 职责非常窄，通常只封装一个小算子或一个小 scope。
  2. 接口非常窄，主要是标量、少量寄存器变量或局部中间量。
- 当前 `qwen_prefill_attention_context_stage_catapult` 与之相反：
  1. 顶层仍是大数组端口：`q_proj_buffer`、`k_cache`、`v_cache`、`context_buffer`，没有把 context 这条路径真正改成 `ac_channel` stream top。
  2. 单个 `ccore` 内仍保留 `query -> head-group -> key-tile -> key -> dim` 多层嵌套，并在同一块里完成 softmax 两遍扫描、分母更新和 value 累加。
  3. `tile_config` 作为结构体整体输入，仍让 Catapult 面对较宽、较动态的控制组合，而不是更接近 `DirectInput` 的离散小配置。
- 因此“已经引入了 `ac_channel`”并不等价于“这条 compile 图已经按 Catapult 推荐方式被切成小块”。当前真正的问题是：context 主体还没有被拆成官方例子那种窄接口、小 `ccore`、块间显式连接的结构。

### 主线结论
- 短期：先用 50 GB 守卫避免机器被拖死，并持续记录 compile 内存曲线。
- 中期：继续按 `cache reader -> score core -> softmax/context core` 三段式拆分 context 主体，而不是只在外层增加 channel 痕迹。
- 提交时应包含守卫脚本、Makefile 接线、context 专用翻译单元、Tcl 调整和本日志更新。

## 2026-03-18 后续推进：query 级内部三段化

### 本轮动作
- 在 `hls/prefill_only/qwen_prefill_attention_context_stage_catapult.cpp` 内，把 `prefill_attention_context_block_fp` 和 `prefill_attention_context_query_tile_fp` 从“单循环内直接 load + compute + store”改成内部三段：
  1. `stream_context_query_tasks_*` 负责 query loader
  2. `compute_context_query_tasks` 负责 query compute
  3. `store_context_result_packets` 负责 result store
- 三段之间通过 `ac_channel<ContextQueryTaskPacket>` 和 `ac_channel<ContextResultPacket>` 连接，保持外部函数签名不变。

### 当前意义
- 这一步还没有把 `context` 主体彻底拆成 `cache reader -> score core -> softmax/context core`，但已经把最外层 query 调度从单块串行逻辑改成了显式的 loader / compute / store 结构。
- 这样后续继续向 `score core` 与 `softmax/context core` 内部分裂时，不需要再重复改 query 级调度骨架。

### 本轮快速验证
- 使用 `QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context` 做前端入场验证。
- 结果：未出现新的前端 `Error:`，仍然可以稳定完成 `analyze` 并进入 `compile`。
- 早期 compile 曲线仍与上一轮相近：
  - `29.97s -> 6068716kB`
  - `59.95s -> 12079560kB`
  - `89.95s -> 14438856kB`
- 结论：这次 query 级三段化至少没有破坏已有 compile 入口；是否能改变后半段内存失控，还需要后续长跑观察。

### 后续验证
- 重新运行 context flow，观察：
  1. `Found design routine` 的数量是否明显下降。
  2. `compile` 峰值内存是否低于当前约 `15 GB`。
  3. 是否仍能稳定进入 `compile`，且不引入新的前端错误。

### 待提交范围
- `hls/prefill_only/qwen_prefill_attention_kernel.cpp`
- `hls/prefill_only/qwen_prefill_attention_context_stage_catapult.cpp`
- `script/run_catapult_prefill_attention_context.tcl`
- `docs/commit_rpt.md`