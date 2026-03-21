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

## 2026-03-18 新约束：DDR 读取口最大 256bit

### 约束解释
- 所有由 DDR/大数组搬运进入内部 channel 的数据，单拍最大位宽限制为 `256bit`。
- 对 `fp32` 数据，这意味着单 packet 最多 `8` 个元素。
- 对 `packed_w4_t`，这意味着单 packet 最多 `32` 个字节。

### 已完成检查
- 当前 `context` query 级 channel 已按该约束收紧：
  - 原先 `ContextQueryTaskPacket` 和 `ContextResultPacket` 都携带整 token，远超 `256bit`。
  - 现已改成：
    1. `ContextQueryMetaPacket` / `ContextResultMetaPacket` 负责索引元数据。
    2. `ContextFpWordPacket` 负责每拍 `8 x fp32 = 256bit` 的分词搬运。
- 这意味着 query loader / compute / store 三段之间，不再通过整 token 大包传输，而是按 `1536 / 8 = 192` 个 word 顺序搬运。

### 当前超限项盘点
- 仍需后续继续收敛的超限 `ac_channel` packet 主要集中在 `q_context_output` 和 `kv_cache` 路径：
  1. `qwen_prefill_attention_kv_cache_stage.cpp` 里的 `Kv*Packet` 仍需要按同样规则检查和收窄

### 本轮继续收敛：q_context_output 也压到 256bit
- `q_context_output` 路径里的 `HiddenProj*Packet` 原先全部超限：
  1. `HiddenProjFpTilePacket`：`64 x fp32 = 2048bit`
  2. `HiddenProjScaleTilePacket`：`64 x fp32 = 2048bit`
  3. `HiddenProjPartialTilePacket`：`64 x fp32 = 2048bit`
  4. `HiddenProjPackedWeightTilePacket`：`64 x 64 / 2 x 8bit = 16384bit`
- 本轮已将它们从 channel 侧替换为分词包：
  1. `HiddenProjFpWordPacket`：`8 x fp32 = 256bit`
  2. `HiddenProjPackedWeightWordPacket`：`32 x packed_w4_t = 256bit`
- 保留 tile array core 本地全宽数组不变，只把 channel 边界改成 256bit word 流；因此算法主体不变，约束只作用在搬运边界。

### 本轮快速验证
- 在 `HiddenProj` 256bit 分词改造后，使用 `QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context` 做快速前端验证。
- 结果：未引入新的前端 `Error:`，仍可完成 `analyze` 并进入 `compile`。
- 早期 compile 曲线为：
  - `29.95s -> 6265324kB`
  - `59.94s -> 12210632kB`
  - `89.93s -> 14438856kB`
- 结论：这次改动主要价值在于满足 `256bit` channel 约束、消除后续接口风险；对当前 `context` compile 的早期内存没有明显改善，后续仍应继续针对 `score core / softmax-context core` 的厚计算主体做分裂。

### 主线影响
- 从现在开始，新的 channel 化步骤必须默认以 `256bit` 为上限设计 packet。
- 后续继续推进时，优先把 `score/context` 主线内部保持在 `256bit` 分词搬运，再逐步回收 `q_context_output` 与 `kv_cache` 现有的大 tile packet。

### 约束下的快速验证
- 使用 `QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context` 对 256bit query channel 改造做快速验证。
- 结果：未引入新的前端 `Error:`，仍可完成 `analyze` 并进入 `compile`。
- 早期 compile 曲线相较上一轮略有下降：
  - `29.99s -> 5454380kB`，低于上一轮的 `6068716kB`
  - `59.97s -> 10965448kB`，低于上一轮的 `12079560kB`
  - `89.96s -> 14438856kB`，与上一轮接近
- 当前结论：`query` 级 256bit 分词至少没有恶化 compile 入口，并对早期内存有轻微改善；后半段是否继续失控，仍需长跑验证。

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

## 2026-03-18 继续收敛：kv_cache channel 边界压到 256bit

### 本轮动作
- `hls/prefill_only/qwen_prefill_attention_kv_cache_stage.cpp` 原先的 stream top 仍使用整 tile packet：
  1. `KvFpTilePacket`：`64 x fp32 = 2048bit`
  2. `KvPackedTilePairPacket`：`K/V` 双路打包后远超 `256bit`
  3. `KvScaleTilePairPacket`：`K/V` 双路 scale 整 tile
  4. `KvPartialTilePairPacket`：`K/V` 双路 partial sum 整 tile
- 本轮保持 `qwen_prefill_attention_kv_tile_array_core(...)` 不变，只把 stream top 边界改成 word stream：
  1. `KvFpWordPacket`：`8 x fp32 = 256bit`
  2. `KvPackedWordPacket`：`32 x packed_w4_t = 256bit`
  3. 原先 `K/V` 成对大包在 channel 边界拆成独立 `K` / `V` word channel
- 这样 `input`、`layernorm weight`、`packed weight`、`scale`、`partial sum` 几类搬运都满足新的 `256bit` 上限，tile 级本地数组只留在 core 内部。

### 当前判断
- 这一步的目标仍然是先消除接口位宽违规，不直接承诺会改善 `context` compile 内存。
- 但它补齐了 `kv_cache` 这条主路径上最明显的 oversized channel，避免后续在同类 stream top 上继续引入超宽包。

### 待验证
- 需要对 `QWEN_HLS_ENABLE_EXTRACT=0 make catapult_prefill_attention_kv_cache` 做快速入场验证，至少确认：
  1. 没有新的前端 `Error:`
  2. `analyze` 能完成
  3. `compile` 能启动

### 本轮快速验证
- 使用 `timeout 120s env QWEN_HLS_ENABLE_EXTRACT=0 make catapult_prefill_attention_kv_cache` 做有时限的快速入场验证，避免 `kv_cache` flow 在首次 compile 时无界运行。
- 结果：
  1. `work/tmp/catapult_prefill_attention_kv_cache_latest.log` 记录到 `Starting transformation 'analyze'`。
  2. 同一日志继续进入 `Starting transformation 'compile'`。
  3. 未观察到新的前端 `Error:`；输出里只看到与此前一致的 `CRD-1`、`CRD-68`、`CRD-111`、`CRD-541` 类告警。
- 结论：`kv_cache` 的 256bit word-stream 改造至少没有破坏 Catapult 前端入场，当前可以作为独立 checkpoint 提交。

## 2026-03-18 回到主线：context compute 再拆一层

### 本轮动作
- `compute_context_query_tasks(...)` 原先仍把整条 query 计算压在一段里：
  1. 读入 query
  2. 完整跑 `max score` pass
  3. 再完整跑 `softmax/value accumulate` pass
  4. 直接输出 context token
- 本轮把它拆成两个显式阶段：
  1. `compute_context_score_tasks(...)` 只负责 `max score` pass
  2. `compute_context_value_tasks(...)` 只负责 `softmax/value accumulate` 与结果写出
- 中间只通过 `256bit` word channel 传两类数据：
  1. 继续转发 query token 的 `ContextFpWordPacket`
  2. 新增 `max_score` 的 `ContextFpWordPacket` 流，按 `12` 个 head score 分成 `2` 个 word packet 传递

### 当前意义
- 这一步开始真正把 `context` 主体从“单个厚 query compute”切成 `score core` 和 `softmax/context core` 两段，而不只是做外围 loader / store 包装。
- 中间态没有引入新的超宽 `ac_channel`，仍保持在 `256bit` 上限之内。

### 待验证
- 需要重新执行 `QWEN_HLS_ENABLE_EXTRACT=0 make catapult_prefill_attention_context`，确认：
  1. 没有新的前端 `Error:`
  2. `analyze` 能完成
  3. `compile` 仍可进入
  4. 早期 compile 内存是否有可见变化

### 本轮快速验证
- 使用 `timeout 140s env QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context` 做带守卫的快速验证。
- 结果：
  1. `work/tmp/catapult_prefill_attention_context_latest.log` 记录到 `Completed transformation 'analyze'`，`analyze` 在 `8.77s` 完成。
  2. 同一日志继续进入 `Starting transformation 'compile'`。
  3. 未观察到新的前端 `Error:`；仍然只有既有的 `CRD-68`、`CRD-111` 等告警。
  4. 早期 compile 曲线为：
     - `29.95s -> 5346268kB`
     - `59.94s -> 10736328kB`
     - `89.93s -> 14438856kB`
- 对比上一轮 query-channel 256bit 改造后的快速验证：
  1. `29.99s -> 5454380kB` 降到 `5346268kB`
  2. `59.97s -> 10965448kB` 降到 `10736328kB`
  3. `89.96s -> 14438856kB` 基本持平
- 当前结论：把 `context` 主体继续拆成 `score core + softmax/value core` 对 compile 早期内存有轻微改善，但 90 秒附近仍回到此前量级，说明后续还需要继续压缩更深层的计算图，而不是停在 query 级两段化。

## 2026-03-18 继续压缩 score core：max-score 改成 K-only packet

### 本轮动作
- `process_context_max_score_key(...)` 之前虽然只需要 `K`，但仍复用了 `ContextKvTokenPacket` 路径，并把 `k_proj` 伪装成 `v_proj` 传入，等于在 `max score` pass 里仍构造了一份冗余的 `V` 侧数据通路。

## 2026-03-18 继续落实 stream top：context 顶层切到 channel top

### 本轮动作
- 在 `docs/project_plan.md` 中补充 `RTL 接口硬约束`：所有用于生成 RTL 的 `top/ccore` 默认应使用 `ac_channel` 数据口和 `<=256bit` 标量/寄存器配置口；任何超 `256bit` 的非 `ac_channel` 端口都应视为必须解释的问题。
- 将 `hls/prefill_only/qwen_prefill_attention_context_stage_catapult.cpp` 中的数组型 `context_stage_catapult` 真正降格为 loader/store wrapper：
  1. wrapper 负责从 `q_proj_buffer` 装载 query tile 并写入 `query_meta_chan/query_word_chan`
  2. `qwen_prefill_attention_context_query_tile_stream_catapult(...)` 作为新的 stream top 接收 channel + `k_cache/v_cache`
  3. wrapper 再通过 `store_context_result_packets(...)` 回写 `context_buffer`
- 同时修复了一次错误 patch 遗留的源码损坏：`qwen_prefill_attention_q_context_output_stage_catapult(...)` 原先被 stream-wrapper 代码打断，已恢复为“Q 投影/RoPE -> stream top -> output projection”的完整流程。
- 更新 `script/run_catapult_prefill_attention_context.tcl`，将 `top_function` 切到 `qwen_prefill_attention_context_query_tile_stream_catapult`。

### 本轮快速验证
- 使用 `timeout 150s env QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context` 做守卫验证。
- 结果：
  1. `analyze` 成功完成，日志记录为 `6.91s / 1532964kB peak`。
  2. `compile` 成功启动，没有新的前端 `Error:`。
  3. 新 top 已被识别为 `qwen_prefill_attention_context_query_tile_stream_catapult`。
- 早期 compile 曲线为：
  - `29.97s -> 6330860kB`
  - `59.96s -> 12014024kB`
  - `89.95s -> 14373320kB`
  - `119.94s -> 14181072kB`

### 当前结论
- 这次修改完成了用户要求的结构目标：
  1. Tcl top 已切到真正的 channel stream top
  2. `q_proj_buffer/context_buffer` 已外移到数组 wrapper
  3. stream top 的接口已收敛为 `ac_channel + 标量配置 + cache 指针`
- 但从 compile 曲线看，收益并不明显：
  1. 早期内存与上一轮“仅内部 stage stream 化”相比几乎持平
  2. `compile` 仍在约 `14.37GB` 附近回到同一平台
  3. 日志中可见 Catapult 继续对 `qwen_prefill_attention_context_query_tile_stream_catapult`、`prefill_attention_context_score_stream_stage_fp`、`prefill_attention_context_value_stream_stage_fp` 做 `Inlining routine ...`，说明仅把 top_function 切到 stream top 还不足以阻止 compile 图重新膨胀
- 因此当前更像是“接口形状整改完成，但 compile 图边界仍未被 Catapult 真正保留”。下一步应优先研究如何阻止这些 stage 在 compile 中被重新 inline，而不是再回到数组 top 上做外围包装。

### 后续文档核对与 Tcl 试验
- 继续查本地 Catapult 2026.1 安装目录中的 methodology 示例，重点核对：
  1. `CCORE_Stagewise_Enable`
  2. `mpccore`
  3. `ParallelCCORE`

## 2026-03-18 真正的边界重构：score/value child-top 去掉 k_proj/v_proj 裸指针

### 本轮动作
- 直接在 `hls/prefill_only/qwen_prefill_attention_context_stage_catapult.cpp` 内重构 `score/value` 两个 child-top 的层级边界：
  1. `prefill_attention_context_score_stream_stage_fp(...)` 不再接收 `k_proj`
  2. `prefill_attention_context_value_stream_stage_fp(...)` 不再接收 `k_proj/v_proj`
  3. 两个 child-top 的对外接口都收敛为 `ac_channel + scalar/config`
- 同时把原先落在 child-top 内部的 `K/V cache` 指针读取上提到 wrapper / primary-top 一侧：
  1. 新增 `stream_context_score_stage_inputs(...)`，按 query/head-group/key-tile 的真实消费顺序，把 `k_proj` 预读成 `K word stream`
  2. 新增 `stream_context_value_stage_inputs(...)`，把 `k_proj/v_proj` 预读成 `K word stream + V word stream`
  3. child-top 内部改为通过 `read_context_k_token_packet_words(...)` / `read_context_v_token_packet_words(...)` 从 channel 取数，而不是在综合边界上直接持有 memory pointer
- 保持数组/内存只停留在 wrapper 或 `qwen_prefill_attention_context_query_tile_stream_catapult(...)` 这一侧，符合前一轮文档审计得到的方向。

### 结果上发生的实质变化
- 这次不是简单地“把签名改薄”而已，而是把 `score/value` 的 key 遍历数据源改成了显式 producer-consumer：
  1. wrapper/top 负责按 child-top 的真实遍历顺序复制 query/meta/score word stream，并生成 `K/V` 输入流
  2. child-top 只保留 `query/meta/score/result` 与 `K/V word stream` 的消费逻辑
  3. 因而 `k_proj/v_proj` 不再跨越 `score/value child-top` 的综合边界
- 这使得当前 bottom-up flow 中最关键的两个 child-top，终于从“channel + raw memory pointer 混合边界”变成了“channel-only 边界”。

### 本轮快速验证
- 使用：`QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context`
- 结果：
  1. `prefill_attention_context_score_stream_stage_fp.v1` 的 `analyze` 正常完成，日志记录为 `3.87s / 1336356kB peak`
  2. 随后正常进入 `compile`
  3. `29.98s` 时 score child compile 内存约为 `6575520kB`
  4. 日志中已明确出现 `Inlining routine '<unnamed>::read_context_k_token_packet_words'`，说明 score child 当前消费的是 `K word channel`，而不是原来的 `k_proj` memory pointer
- 也就是说，这次边界重构已经被 Catapult 实际接受，不是停留在代码层的表面改动。

### 当前结论
- 这轮已经完成此前文档审计要求的关键一步：
  1. `score/value child-top` 不再把 `k_proj/v_proj` 暴露在函数边界上
  2. child-top 的接口已收敛到 `ac_channel + scalar/config`
- 但 compile 压力并没有因此立刻消失：
  1. score child 仍在 `compile` 早期迅速爬到约 `6.6GB`
  2. 说明当前主要矛盾已进一步收敛到 `score child` 自身内部计算图，而不再是“边界上仍挂着 memory pointer”这个旧问题
- 因此后续主线应切到：
  1. 继续观察 `score/value child` 在 channel-only 边界下的长期 compile 曲线
  2. 若平台仍高，再进一步拆解 child 内部的 `dot_product / score packet / exp-softmax` 路径，而不是回退到边界问题上反复试错

## 2026-03-18 文档回看与 warning 审计：当前核心问题不是“channel 数量不够”，而是层级边界建模错位

### 本地官方文档确认到的两条关键规则
- `Mgc_home/shared/pdfdocs/js/ccsDoc.json` 中关于 `ac_channel` shared memory 的限制明确写到：
  1. `C-style arrays mapped to memories are only supported for primary inputs/outputs`
  2. `Merging of C-style arrays onto a shared resource is not supported between blocks`
- 同一份文档的 datatype/interface 章节还明确写到：
  1. `Catapult doesn't have any frame of reference to reduce most variables declared on the top-level function interface`
  2. `For consistency, the number of bits on a port variable is never changed`

### 对照当前 context flow 的实际状态
- 当前 Tcl 流里，真正被设为 top 的共有三个边界：
  1. `prefill_attention_context_score_stream_stage_fp`
  2. `prefill_attention_context_value_stream_stage_fp`
  3. `qwen_prefill_attention_context_query_tile_stream_catapult`
- 这三个边界并没有全部收敛为 `ac_channel + 窄标量`：
  1. `prefill_attention_context_score_stream_stage_fp(...)` 仍带 `const catapult_fp_t* k_proj`
  2. `prefill_attention_context_value_stream_stage_fp(...)` 仍带 `const catapult_fp_t* k_proj` 和 `const catapult_fp_t* v_proj`
  3. `qwen_prefill_attention_context_query_tile_stream_catapult(...)` 仍带 `const catapult_fp_t* k_cache` 和 `const catapult_fp_t* v_cache`
- 也就是说，目前只把 `query/context` 载荷做了 channel 化，但 `K/V cache` 仍以裸数组/指针的形式穿过 top/child-top 边界。
- 从官方规则看，这种“分块层级之间仍以 C-style memory array 相连”的建模，本身就不符合其推荐的 shared-memory / hierarchical-block 风格；这比“有没有继续多加几个 channel wrapper”更接近当前 compile 图失控的根因。

### 对 warning 的重新分层
- 已确认不是主问题、但应先清理的 warning：
  1. Tcl 重复 include 路径导致的 `#1819-D`，本轮已通过收窄 `SearchPath/CompilerFlags` 去掉。
  2. 若干仓库文件缺结尾换行导致的 `CRD-1`，大部分已消除；目前只剩 `hls/catapult_shims/limits.h` 和 `hls/catapult_shims/cmath.h` 两处残留，属于日志卫生问题。
- 不是当前主攻方向的 warning：
  1. `ac_fixed.h` / `ac_std_float.h` 大量 `CRD-68`、`CRD-111`。
  2. 这些来自算法库模板实例化和类型转换链，量很大，但现阶段更像 compile 噪声放大器，不是层级边界错误本身。
- 需要保留关注、但不是“前端头文件问题”的 warning：
  1. `CIN-393 Original user directive 'block' overridden by 'top' directive`
  2. 这说明 Tcl 正在显式把 `ccore` helper 提升为 solution top；这是当前 bottom-up 流程的一部分，不是偶发错误。

### 当前主线判断
- 现在真正要解决的，不是继续盲目做“局部表达式瘦身”实验，而是先把 memory-bearing 边界重新建模：
  1. 让数组/内存只停留在 primary top 或 wrapper 一侧。
  2. score/value child-top 不再直接接 `k_proj/v_proj` 裸指针。
  3. top 与 child-top 之间改成显式 `ac_channel` 传输的 `K-only` / `KV` word-stream 或 tile-stream。
- 如果这一点不先做，Catapult 仍会在层级边界上同时处理“channel 网络 + 不可缩减的 memory pointer port”，compile 图很容易继续膨胀，之前出现的 `CIN-84 Unable to reduce array size for variable 'k_proj.d'` 也与这个方向一致。

### 后续动作收敛
- 下一轮不再先碰 `dot_product_128_fp`、`reduce_sum_128_fp` 这类局部算子。
- 优先按以下顺序推进：
  1. 先把 `score/value` child-top 的 `k_proj/v_proj` 指针接口移出层级边界。
  2. 只保留 `ac_channel + scalar/config` 作为 child-top 对外接口。
  3. 再观察 `compile` 是否仍然在 score child 上维持同样的内联膨胀和 `k_proj.d` 相关建模压力。
- 结论是：Catapult 没有单独的“function no-inline pragma”；较接近的官方机制只有两类：
  1. `directive set -STAGEWISE_ENABLE ccore`
  2. 在子 `ccore` 调用点使用 `-CCORE_FLOW both`
- 按文档先后做了两轮 Tcl 试验：
  1. 打开 `-CCORE_INOUT_MODE split_io_port` + `-STAGEWISE_ENABLE ccore`
  2. 再对子 stage 调用点增加 `-CCORE_FLOW both`
- 两轮结果一致：
  1. 指令都被 Catapult 接受
  2. `analyze` 仍稳定通过
  3. `compile` 仍正常进入
  4. 但 `CIN-14 Inlining routine` 依旧存在，`qwen_prefill_attention_context_query_tile_stream_catapult` 和 `prefill_attention_context_score_stream_stage_fp` 仍被 inline
  5. compile 内存仍在约 `14.37GB` 处回到同一平台
- 当前判断：
  1. 轻量 Tcl 开关不足以保住当前 score/value stage 边界
  2. 若要继续验证“是否能真正保边界”，必须转入 `mpccore` 风格的更强方案：bottom-up / unified CCORE flow
  3. 即先把 `score/value` 单独建成 CCORE library，再在 top flow 中用 `-MAP_TO_MODULE {[CCORE] ...}` 或全局 `-CCORE_FLOW bottomup` 绑定，而不是继续在单 solution 中做小范围 pragma/Tcl 微调

### 2026-03-18 bottom-up CCORE flow 首次跑通到 child compile
- 本轮先把 `script/run_catapult_prefill_attention_context.tcl` 改成真正的多 solution bottom-up 结构：
  1. `prefill_attention_context_score_stream_stage_fp` 先单独建 child solution
  2. `prefill_attention_context_value_stream_stage_fp` 计划作为第二个 child solution
  3. 顶层 `qwen_prefill_attention_context_query_tile_stream_catapult` 作为 top solution，后续再绑定 child CCORE library
- 最初的 blocker 不是 compile，而是 child solution 的 `go analyze`：
  1. Catapult 2026.1 在这个多 solution 流里直接拒绝 `-include limits.h` / `-include climits`
  2. 报错为 `go analyze: invalid compiler flag "-include"`
- 这一步之后按仓库已有 `docs/catapult_limits_header_fix.md` 的结论回退到“本地化头链 + 显式 `.h` shim”路线，做了两类修复：
  1. Tcl 侧：去掉 `-include limits.h`、`-include climits` 以及 `_GCC_LIMITS_H_` / `_LIBC_LIMITS_H_` 这类绕路宏；同时把 `design_files` 缩到真正的 translation unit，只保留 `qwen_prefill_attention_context_stage_catapult.cpp`
  2. 头文件侧：把无扩展名 shim（`cmath` / `climits` / `cstddef` / `cstdint` / `cstring` / `iostream` / `ostream` / `string`）统一收口到对应 `.h` shim；补齐 `hls/catapult_shims/cmath.h` 中被 EDG/libstdc++ 与 `ac_*` 头实际用到的数学符号；并把 `ac_int.h` / `ac_fixed.h` 中少数裸 `floor` / `frexp` 调用显式改成 `std::` 版本
- 修完后，bottom-up child score solution 已经首次越过 `analyze` 并进入 `compile`：
  1. `analyze` 完成记录为 `4.14s / 1336356kB peak`
  2. `compile` 在 30s 左右达到 `6706592kB`，后续平台约 `7103524kB`
  3. 180s 守卫窗口内未见新的前端 `Error:`，说明当前阻塞已经从头文件解析阶段前移到真正的 child compile
- 当前观测到的 compile 形态：
  1. 这次日志里出现了 `Synthesizing routine '<unnamed>::prefill_attention_context_score_stream_stage_fp'`
  2. 说明 child score stage 已经被当作独立 top 在编译，而不是还卡死在入口分析
  3. 但 child compile 内部仍然存在大量 `Inlining routine ...`，即 score child 自身内部 helper 仍被内联，这一步是合理现象；真正需要继续验证的是：后续 top solution 绑定 child CCORE library 后，top compile 是否还能避免把 score/value stage 再次整体 inline 回去
- 当前结论：
  1. 这轮修改已经把 bottom-up 流从“根本跑不起来”推进到“child score CCORE 可以独立 analyze + compile”
  2. 仅看 child score compile，内存平台约 `7.1GB`，明显低于此前单体 context top 快速验证时约 `14.37GB` 的平台值
  3. 下一步不应再回头折腾头文件或轻量 pragma，而应继续把 `score` child 跑完 `extract`，再验证 `value` child 与 top-level `[CCORE]` 绑定是否能真正保住 stage 边界

### 2026-03-18 long-run 复核：score child compile 不是 7.1GB 平台，而是延后爬升到约 19GB
- 对同一条 bottom-up flow 做更长时间观察后，先前 `180s` 窗口得到的“约 `7.1GB` 平台”结论需要修正：那只是 child score compile 的早期平台，不是最终平台。
- 从 `work/tmp/catapult_prefill_attention_context_monitor.log` 可见，`prefill_attention_context_score_stream_stage_fp` 在通过早期 `7.1GB` 平台后继续单调爬升：
  1. `18:07:36 -> 10380324kB`
  2. `18:08:33 -> 12411940kB`
  3. `18:09:36 -> 14378020kB`
  4. `18:10:33 -> 16475172kB`
  5. `18:11:35 -> 18572324kB`
  6. `18:16:06 -> 18965540kB`
- 同期 Catapult 日志仍停留在同一个 child solution 的 `compile`：
  1. `751.07s -> 18834468kB`
  2. `841.06s -> 18965540kB`
  3. `1231.04s -> 18965540kB`
  4. `1261.04s -> 18965540kB`
- 在这段长跑里没有出现新的前端 `Error:`，但也没有看到：
  1. `Completed transformation 'compile'`
  2. `go assembly`
  3. `go architect`
  4. `go extract`
  5. 切换到 `prefill_attention_context_value_stream_stage_fp`
- 结合日志尾部仍在持续刷新的 `Inlining routine ...`，当前更准确的判断是：
  1. bottom-up flow 已经成功把问题从“入口分析/头文件解析失败”推进到“score child 自身 compile 图过厚”
  2. 但它并没有把 compile 内存真正压在 `7GB` 量级，而是把大内存阶段推迟了几分钟后重新拉高到约 `19GB`
  3. 当前瓶颈已明确收敛到 `prefill_attention_context_score_stream_stage_fp` 自身，而不是 top-level 绑定或 value child
- 因此后续主线应调整为：优先继续削减 score child 内部 compile 图，而不是继续等待 top-level `[CCORE]` 绑定验证。尤其需要针对当前日志中最密集的内联热点继续收敛：
  1. `ac_std_float<32, 8>` 构造/`set_data`/`data_ac_int`
  2. `ccs_dw_fp_lib.h` 下的 `fp_mult` / `ccs_fp_mult`
  3. `dot_product_128_fp` 与 `compute_context_score_packet` 一带的 helper 展开

## 2026-03-18 方案 A 继续推进：score/value stage 深拆但未改变 compile 平台期

### 本轮动作
- 延续用户指定的方案 A，不再停留在外层 tile wrapper：
  1. 先把 `prefill_attention_context_score_*_stage_fp(...)` / `prefill_attention_context_value_stage_fp(...)` 改成真正直接执行 per-query 读入、计算、写出的 stage，而不是只包一层 `compute_context_*_tasks(...)`。
  2. 去掉 `value stage` 内部 `context_meta/context_word` 中间 channel，把结果直接写回 `context` tile。
  3. 进一步尝试把 `prefill_attention_context_query_max_score_fp(...)` / `prefill_attention_context_query_value_fp(...)` 提升为显式 `ccore`。
  4. 再进一步把 `head-group` 级别的 `prefill_attention_context_max_score_head_group_stage_fp(...)` / `prefill_attention_context_value_head_group_stage_fp(...)` 提升为更窄的 `ccore`，并把 `query_max_score/query_value` 退回 orchestration helper。

### 验证结果
- 三轮验证都保持同一结论：
  1. `analyze` 稳定通过，耗时约 `6.4s ~ 6.7s`。
  2. `compile` 稳定启动。
  3. 没有新的前端 `Error:`。
  4. 但 `150s` 守卫窗口内，`compile` 始终停留在 `score_query_tile_stage` / `max_score_head_group_stage` 相关综合展开，未进入新的稳定里程碑。
- 监控曲线基本不变，仍然在早期快速爬升到约 `14GB` 平台：
  - stage 直接执行版：
    - `~30s -> 6920684kB`
    - `~75s -> 13390280kB`
    - `~110s -> 14373320kB`
  - `query_max_score/query_value` 设为 `ccore`：
    - `~30s -> 6920684kB`
    - `~75s -> 13390280kB`
    - `~110s -> 14373320kB`
  - `head-group stage ccore`：
    - `~30s -> 7015356kB`
    - `~75s -> 13586888kB`
    - `~110s -> 14373320kB`

### 当前判断
- 这说明当前瓶颈已经不是“有没有把 score/value 命名成独立 stage”，而是：
  1. `context` 主路径依然是数组接口驱动的大调用图。
  2. `score` 计算主体内部仍然保留 `head-group -> key-tile -> key` 的厚展开。
  3. 在现有数组调用图里继续移动 `ccore` 边界，Catapult 仍会把这些阶段重新 inline 展开，内存平台期基本不变。
- 因此下一步不应继续做同类 `ccore` 位置微调，而应转入真正的 `loader/compute/store` 三段 channel 化：
  1. `cache reader / key-tile meta loader`
  2. `score compute core`
  3. `softmax/value context core`
- 在这一步完成前，本轮修改不具备形成稳定 checkpoint 的条件，因此先不提交。

## 2026-03-18 继续收敛：去掉内部 stage 的大数组 ccore 端口

### 本轮动作
- 针对“函数端口超过 `256bit` 且不是 `ac_channel`”这个约束，继续清理 `context` 主路径里最可疑的几层边界。
- 原先最重的内部 `ccore` 仍然带着大数组端口：
  1. `prefill_attention_context_score_query_tile_stage_fp(...)` 直接接 `q_proj_tile[kPrefillQueryCapacity][kHiddenSize]`
  2. `prefill_attention_context_value_stage_fp(...)` 直接接 `context[kPrefillQueryCapacity][kHiddenSize]`
  3. `prefill_attention_context_*_head_group_stage_fp(...)` 仍直接接整 `ContextQueryPacket` / `ContextTokenPacket`
- 本轮改成：
  1. 去掉这些内部 array-port stage 的 `ccore` 身份
  2. 新增真正的 channel/stream stage：
     - `prefill_attention_context_score_stream_stage_fp(...)`
     - `prefill_attention_context_value_stream_stage_fp(...)`
  3. 数组只留在 loader/store wrapper：
     - `stream_context_query_tasks_from_sequence(...)`
     - `stream_context_query_tasks_from_tile(...)`
     - `store_context_result_packets(...)`
  4. `score/value` 两个被 Catapult 识别的 stage 边界，现在只保留：标量、指针和 `ac_channel`，不再暴露整块 query/context 数组端口。

### 本轮验证
- 重新执行：`timeout 150s env QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context`
- 结果：
  1. `analyze` 在 `7.16s` 完成。
  2. `compile` 正常启动。
  3. 未出现新的前端 `Error:`。
  4. Catapult 日志里，旧的 `score_query_tile_stage_fp` 已不再是 `ccore`；新的 `score_stream_stage_fp` / `value_stream_stage_fp` 被识别为 design routine。
- 监控采样点为：
  1. `~52s -> 6593004kB`
  2. `~83s -> 12603848kB`
  3. `~110s -> 14373320kB`

### 当前结论
- 这一步说明“大数组 ccore 端口”确实是 compile 膨胀来源之一：
  1. 相比前一轮 `head-group stage ccore` 的约 `13.59GB`，本轮 `~83s` 已降到约 `12.60GB`
  2. 早期 compile 图明显变瘦，说明把内部 stage 边界改成 channel/stream 是正确方向
- 但 `~110s` 仍回到约 `14.37GB`，说明还有一个更大的数组边界没有拆掉：
  1. 当前文件里剩余的 `ccore` 只有 `score_stream_stage_fp`、`value_stream_stage_fp` 和 `qwen_prefill_attention_context_stage_catapult`
  2. 其中真正还带大数组口的，只剩 `qwen_prefill_attention_context_stage_catapult(...)` 顶层
- 因此下一步主线已经更明确：
  1. 不是再改内部 `score/value` stage
  2. 而是把 `qwen_prefill_attention_context_stage_catapult` 这个顶层数组口 `ccore` 继续下沉成真正的 stream top，或者把 `cache reader / store` 再外移一层
  3. 若不处理这个顶层大数组口，Catapult 仍会把内部 stream stage 重新 inline 回顶层大图，后半段内存平台仍难消失
- 本轮改成：
  1. 新增 `ContextKTokenPacket`
  2. 新增 `load_context_k_token_packet(...)`
  3. 为 `compute_context_score_packet(...)` 增加 `K-only` 重载
  4. `process_context_max_score_key(...)` 不再走 `ContextKvTokenPacket`

### 预期
- 目标是继续缩小 `score core` 内部数据结构和依赖图，避免 `max score` pass 带着不需要的 `V` 路径一起被编译展开。

### 本轮快速验证
- 使用 `timeout 140s env QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context` 再做一次带守卫的快速验证。
- 结果：
  1. `analyze` 仍然完成，日志记录为 `7.43s`。
  2. `compile` 仍然正常进入。
  3. 未观察到新的前端 `Error:`。
  4. 早期 compile 曲线为：
     - `29.98s -> 6068716kB`
     - `59.96s -> 11882952kB`
     - `89.95s -> 14438856kB`
     - `119.94s -> 14438856kB`

### 当前结论
- 这次 `K-only` 化没有破坏 compile 入口，但对早期内存没有带来正收益，反而比上一轮 `score/value` 两段化略高：
  1. `29.95s` 从约 `5.35GB` 回到约 `6.07GB`
  2. `59.94s` 从约 `10.74GB` 回到约 `11.88GB`
  3. `89.9s` 后仍回到 `14.44GB`
- 因此目前看，`score core` 的 `K/V` 包结构不是 compile 内存的主要矛盾；下一步仍应回到更高层的循环图切分，优先考虑 `head-group` / `key-tile` 级边界，而不是继续在单 key helper 上做局部瘦身。

## 2026-03-18 继续压缩循环图：key-tile 元数据分阶段

### 本轮动作
- 继续沿着 `head-group / key-tile` 边界做显式分阶段，把 `max score` 和 `value accumulate` 两个 pass 里原先直接展开的 `key_tile` 循环提取成元数据驱动的任务流：
  1. 新增 `ContextKeyTileMetaPacket`
  2. 新增 `init_context_key_tile_meta_packet(...)`
  3. 新增 `count_context_key_tiles(...)`
  4. 新增 `stream_context_key_tile_meta_packets(...)`
  5. 新增 `compute_context_max_score_tile_tasks(...)`
  6. 新增 `compute_context_value_tile_tasks(...)`
- `compute_context_max_score_head_state_packet(...)` 与 `compute_context_value_head_state_packet(...)` 改为先生成 `key_tile` 元数据流，再分别消费该流完成 tile 级 pass。

### 首次验证与修复
- 第一次快速验证没有进入 `compile`，而是在 `analyze` 前端失败。
- 失败原因不是算法逻辑，而是 Catapult EDG 对类型声明顺序更严格：
  1. `ContextKeyTileMetaPacket` 初版仍定义在使用它的函数签名之后。
  2. `work/tmp/catapult_prefill_attention_context_latest.log` 报出 `CRD-20`：
     - `qwen_prefill_attention_context_stage_catapult.cpp(995)`
     - `qwen_prefill_attention_context_stage_catapult.cpp(1000)`
     - `qwen_prefill_attention_context_stage_catapult.cpp(1020)`
     - `qwen_prefill_attention_context_stage_catapult.cpp(1027)`
- 修复方式：把 `ContextHeadStatePacket` 与 `ContextKeyTileMetaPacket` 一并前移到 packet 定义区，放到所有相关函数签名之前。

### 修复后快速验证
- 重新执行 `timeout 150s env QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context`。
- 结果：
  1. `analyze` 已恢复通过，日志记录为 `6.95s`。
  2. 随后正常进入 `Starting transformation 'compile'`。
  3. 本次退出原因为外层 `timeout 150s`，不是新的前端 `Error:`。
- 监控采样点显示这轮已回到有效 compile 入口：
  1. 约 `16s -> 1532964kB`
  2. 约 `52s -> 6461932kB`
  3. 约 `78s -> 12538312kB`
  4. 约 `109s -> 14438856kB`
  5. 直到 `150s` 超时前，进程树 RSS 仍维持在约 `13.4GB ~ 13.8GB`

### 当前结论
- 这一步的直接价值是把 `key_tile` 循环正式提升为显式 stage 边界，并确认 Catapult 前端可以接受该结构。
- 从 `analyze -> compile` 入场表现看，它没有比上一轮更差，也没有立即拉低 `90s+` 区间的内存平台。
- 因此当前主矛盾进一步收敛到：仅仅把 `key_tile` 变成元数据驱动 stage 还不够，后续仍需要继续拆 `softmax / value accumulate` 内部的厚状态更新，或者进一步减少单个 stage 中携带的 head-state 体积。

## 2026-03-18 继续压缩 value pass：softmax weight 与 value accumulate 分段

### 本轮动作
- 在 `value` pass 内继续把“算 score/exp + 读 V + 更新 denom/accum”这段厚逻辑拆成两个显式子阶段：
  1. `stream_context_value_key_tasks(...)` 负责为每个 `key` 生成 `exp(score-max)` 权重流，并同步发出 `V` word 流。
  2. `accumulate_context_value_key_tasks(...)` 负责消费这两路 `256bit` word stream，更新 `denom` 与 `accum`。
- 新增/调整的关键组件：
  1. `kContextKvWordCount`
  2. `ContextVTokenPacket`
  3. `accumulate_context_weighted_value_packet(...)`
  4. `stream_context_v_token_packet_words(...)`
  5. `read_context_v_token_packet_words(...)`
  6. `stream_context_value_key_packet_words(...)`
  7. `accumulate_context_value_key_packet_words(...)`
- `process_context_value_tile(...)` 现在不再直接在单循环内做全部工作，而是通过两路 `ac_channel<ContextFpWordPacket>` 显式连接这两个子阶段。

### 首次验证与修复
- 第一次快速验证仍然没有进入 `compile`，但失败原因依旧是 Catapult EDG 的声明顺序，而不是新拆分逻辑本身。
- 新增的 value-stage helper 被放在 `ContextFpWordPacket` 与相关 word helper 之前，导致 `analyze` 报出新的 `CRD-20`：
  1. `ContextFpWordPacket` 未定义
  2. `load_context_fp_word_packet(...)` 未定义
  3. `store_context_fp_word_packet(...)` 未定义
  4. `stream_context_score_packet_words(...)` / `read_context_score_packet_words(...)` 未定义
- 修复方式：
  1. 把 `ContextFpWordPacket` 定义前移到 packet 定义区。
  2. 为上述 word helper 增加前置声明，确保 EDG 在新 value-stage 函数体处可见。

### 修复后快速验证
- 重新执行 `timeout 150s env QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context`。
- 结果：
  1. `analyze` 在 `6.81s` 完成。
  2. 后续正常进入 `Starting transformation 'compile'`。
  3. 本次退出原因仍然是外层 `timeout 150s`，不是新的前端 `Error:`。
- 监控采样点为：
  1. 约 `15s -> 1532964kB`
  2. 约 `51s -> 6134252kB`
  3. 约 `77s -> 12276168kB`
  4. 约 `106s -> 14504392kB`
  5. `150s` 前进程树 RSS 约在 `13.3GB ~ 13.8GB` 区间

### 当前结论
- 这次拆分已经把 `value` pass 里最厚的一段显式拆成“weight 生成”和“value 累加”两层，且 Catapult 前端已经接受该结构。
- 与上一轮 `key_tile` 分阶段相比，早期内存略有下降：
  1. `~52s` 从约 `6.46GB` 降到约 `6.13GB`
  2. `~78s` 从约 `12.54GB` 降到约 `12.28GB`
- 但 `~106s` 仍回到约 `14.50GB`，说明 compile 图的主要厚度还留在更深层的状态更新或后续阶段，当前拆分只能改善早期膨胀，尚未改变 `90s+` 平台。

## 2026-03-18 继续压缩 value head-state：移除冗余 max_score 携带

### 本轮动作
- 在 `value` pass 的 head-state 里，`max_score` 原先被重复携带在 `ContextHeadStatePacket` 中，但它实际上已经在 `score -> value` 边界通过独立数组/word stream 传入。
- 本轮把该状态包收窄为 `ContextValueHeadStatePacket`，仅保留：
  1. `denom`
  2. `accum`
- 主要调整点：
  1. `init_context_value_head_state_packet(...)`
  2. `compute_context_value_head_state_packet(...)`
  3. `store_context_head_state_packet(...)`
  4. `process_context_head_group(...)`
  5. `process_context_value_head_group(...)`
  6. `prefill_attention_context_query_value_fp(...)`
- 同时修复了在尝试该改动过程中被局部破坏的 query stream orchestration 区域，恢复了：
  1. `compute_context_score_tasks(...)`
  2. `compute_context_value_tasks(...)`
  3. `prefill_attention_context_block_stream_fp(...)`
  4. `prefill_attention_context_query_tile_stream_fp(...)`

### 修复后快速验证
- 重新执行 `timeout 150s env QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context`。
- 结果：
  1. `analyze` 在 `6.93s` 完成。
  2. 后续正常进入 `Starting transformation 'compile'`。
  3. 本次退出原因为外层 `timeout 150s`，不是新的前端 `Error:`。
- 监控采样点为：
  1. 约 `16s -> 1532964kB`
  2. 约 `52s -> 6265324kB`
  3. 约 `78s -> 12145096kB`
  4. 约 `110s -> 14438856kB`
  5. `150s` 前进程树 RSS 维持在约 `13.3GB ~ 13.8GB`

### 当前结论
- 这一步确认了：`max_score` 不需要继续跟随 value head-state 一起搬运，Catapult 前端也接受这种更窄的状态表达。
- 与上一轮 value 子阶段拆分相比：
  1. `~51s` 从约 `6.13GB` 回到约 `6.27GB`
  2. `~77s` 从约 `12.28GB` 降到约 `12.15GB`
  3. `~106s` 从约 `14.50GB` 回到约 `14.44GB`
- 当前判断：移除冗余 `max_score` 携带对 compile 图后半段有轻微收敛，但幅度仍然有限；下一步更可能需要继续压缩 `value` 路径内部状态更新的耦合，或者进一步拆分 `head-group` 级 orchestrator，而不是继续在单个 packet 字段上做细粒度修补。

## 2026-03-18 方案 A：context stage 改成 query-tile wrapper/core 形态

### 本轮动作
- 按 Catapult 推荐的“wrapper 与固定尺寸 compute core 分离”思路，继续收窄 `context` top 的职责。
- 新增两个 query-tile wrapper helper：
  1. `load_context_query_tile_from_sequence(...)`
  2. `store_context_query_tile_to_sequence(...)`
- `qwen_prefill_attention_context_stage_catapult(...)` 不再让核心直接从 full-sequence `q_proj_buffer` 读取 query，而是改成：
  1. 外层按 `query_tile` 装载 `q_proj_tile`
  2. 调用固定尺寸 core：`prefill_attention_context_query_tile_fp(...)`
  3. 再把 `context_tile` 写回 `context_buffer`
- 同样把 `qwen_prefill_attention_context_output_stage_catapult(...)` 也切到相同的 tile-core 组织方式，避免同一文件内保留两种不同的 `context` 入口形态。

### 本轮意图
- 这一步不是再改 `score/value` 算法本体，而是把 compute core 从“直接感知 full sequence 数组”进一步推进到“只处理固定 query tile”的形态。
- 目标是缩小 top/orchestrator 在 compile 看到的数组视角和循环耦合，向 `loader -> core -> store` 的标准 HLS 数据流建模继续靠拢。

### 快速验证
- 执行：`timeout 150s env QWEN_HLS_ENABLE_EXTRACT=0 QWEN_HLS_MEMORY_POLL_SEC=5 make catapult_prefill_attention_context`
- 结果：
  1. `analyze` 在 `7.55s` 完成。
  2. 后续正常进入 `Starting transformation 'compile'`。
  3. 未出现新的前端 `Error:`；本次仍由外层 `timeout 150s` 截断。
- 监控采样点为：
  1. 约 `15s -> 1532964kB`
  2. 约 `52s -> 6068716kB`
  3. 约 `83s -> 11882952kB`
  4. 约 `109s -> 14504392kB`
  5. `150s` 前进程树 RSS 约在 `13.4GB ~ 13.8GB`

### 当前结论
- 方案 A 的结构方向本身是成立的：Catapult 前端接受“query-tile wrapper + fixed-tile core”这种更接近硬件分层的组织方式。
- 但在当前 `150s` 快速窗口内，没有观察到立刻的 compile-memory 收敛；相较上一轮 head-state 瘦身：
  1. `~52s` 从约 `6.27GB` 回到约 `6.07GB`
  2. `~77s` 从约 `12.15GB` 回到约 `11.88GB`
  3. `~106s` 又上升到约 `14.50GB`
- 当前判断：仅把 query 输入视角从 full-sequence 数组收窄到 tile-core 还不够，compile 图的主要厚度仍在 `score/value` 内部以及 `head-group` 级 orchestration；后续若继续沿方案 A 推进，应进一步把 `score core` / `value core` 提升成更明确的 stage top，而不只是外层 tile wrapper。

## 2026-03-20 full-context 正式验证：结构问题已清空，当前阻塞收敛到 value-stage 调度反馈环

### 本轮目标
- 用户要求从现有 `score-only` 路径升级到完整 `context` top 的正式 Catapult 验证，再在此基础上继续推进 full prefill top。
- 本轮实际验证入口仍通过环境变量覆盖，不直接修改默认 Tcl top：
  1. `QWEN_CONTEXT_TOP_FUNCTION=qwen_prefill_attention_context_query_tile_stream_catapult`
  2. `QWEN_CONTEXT_SOLUTION_NAME=prefill_attention_context_query_tile_stream_catapult`

### 本轮已完成收敛
- `hls/prefill_only/qwen_prefill_attention_context_stage_catapult.cpp` 内，full-context 路径已经从结构性失败推进到真正的调度问题：
  1. 清掉了 channel-based child stage 上不合法的 `ccore` 用法。
  2. 为需要保留层级的 helper/stage 显式补齐 `block` 边界。
  3. 去掉了导致 `HIER-6` 的非静态局部 `ac_channel`，改成跨层级可接受的 `static ac_channel`。
  4. 清理了 `seq_len`、`query_count`、`tile_config.query_heads_parallel` 一带的 `MEM-94` / `MEM-71` 资源映射问题。
  5. 将 full-context 形式化路径专门收敛到当前综合默认的 `query_heads_parallel=2` 头组规模，避免顶层继续背负不必要的控制扇出。

### 当前正式验证进度
- 最新 full-context 运行已经稳定越过以下阶段：
  1. `compile`
  2. `libraries`
  3. `assembly`
  4. `memories`
  5. `cluster`
  6. `architect`
  7. 并进入 `allocate`
- 对应日志：`work/tmp/catapult_formal_attention_context_full_20260320.log`
- 最新一轮代表性指标：
  1. `memories` 完成时：`Total ops = 31043, Real ops = 736, Vars = 4274`
  2. `architect` 完成时：`Total ops = 33997, Real ops = 1510, Vars = 5037`
  3. 峰值内存约 `2249688kB`

### 本轮为调度收敛做的最小代码动作
- 针对 value-stage 反馈环，做了两项已经验证有效但尚不足以闭合调度的收敛动作：
  1. 将 `ATTN_CONTEXT_VALUE_TILE_LOOP` 的 `#pragma hls_pipeline_init_interval` 从 `2` 放宽到 `4`。
  2. 将 `ContextValueHeadStatePacket` 从“按全部 attention heads 分配”收窄为“只按当前综合头组大小 `kContextHeadGroupSize` 分配”，以减少 `denom/accum` 状态体积和对应 RAM 压力。

### 当前唯一主阻塞
- full-context 现在不再卡在 hierarchy 或 memory mapping，而是卡在 `allocate` 阶段的真实调度反馈环：
  1. `SCHD-3`
  2. `SCHD-20`
- 当前失败分区为：
  1. `/<unnamed>::qwen_prefill_attention_context_query_tile_stream_catapult/<unnamed>::prefill_attention_context_value_stream_stage_fp/core`
- 日志显示的关键路径已经收敛到同一条反馈链：
  1. `head_state_packet.accum` RAM read
  2. `ccs_dw_fp_mac<23,8,0>()`
  3. 写回同一 `head_state_packet.accum` RAM
- 最新 trace 中该资源已收敛为较小的 `256 x 32` RAM，说明状态体积压缩已经生效；但 recurrence 仍未被打断。

### 当前结论
- 到这一轮为止，full-context top 的结构性门槛已经基本清空，问题已成功收敛为一个局部、可定位的 value-stage 调度反馈环。
- 因为还没有越过 `allocate -> extract`，所以当前不应：
  1. 直接把 `script/run_catapult_prefill_attention_context.tcl` 的默认 top 从 `score-only` 切到 full-context。
  2. 提前启动 full prefill top 的 RTL 生成。
- 下一步应继续围绕 `head_state_packet.accum` 的存储/访问形态做根因修复，优先目标是打断 `RAM read -> DW FP MAC -> RAM write` 的 recurrence，而不是再做同类外层结构改造。

## 2026-03-21 value-stage 根因修复：快速 `SCHD-3` 已清除，当前卡点转为 allocate 长时间搜索

### 本轮动作
- 继续只动 `hls/prefill_only/qwen_prefill_attention_context_stage_catapult.cpp`，围绕 value-stage 的 recurrence 做根因修复，不再扩大脚本或其他 stage 的修改面。
- 已验证的关键收敛动作包括：
  1. 将 value 累加长期状态从 `ContextValueHeadStatePacket` 热路径中拆成显式 `denom/accum` 数组。
  2. 将“每个 key 都直接写回长期 `head_state_accum`”改成“tile 内局部 `tile_accum/tile_denom` 归并，tile 结束后只 merge 一次到长期状态”。
  3. 将 `ATTN_CONTEXT_VALUE_TILE_LOOP` 的 `II` 从 `4` 继续放宽到 `6`。
  4. 重写 `dot_product_128_fp(...)`，去掉 `lane_products[128]` 临时数组，改成显式 `8-lane chunk` 归约树，清除新的临时 RAM 反馈环。

### 阶段性效果
- 这些修改后，full-context flow 不再在 `allocate` 初期快速报同类 `SCHD-3`。
- 最新完整日志：`work/tmp/catapult_formal_attention_context_full_20260320.log`
- 最新监控日志：`work/tmp/catapult_formal_attention_context_full_20260320_monitor.log`
- 代表性指标已经稳定在较低内存平台：
  1. `assembly` 完成：`Total ops = 3580, Real ops = 274, Vars = 584`
  2. `architect` 完成：`Total ops = 34306, Real ops = 1518, Vars = 5297`
  3. `allocate` 阶段长时间运行时，日志内存稳定在约 `2577368kB`，峰值约 `2608500kB`

### 当前真实状态
- 当前已经不再是“立即失败的 schedule 闭环”，而是：
  1. `allocate` 可以稳定进入并持续运行数百秒以上。
  2. 在手工中断前，没有再出现此前 value-stage 的快速 `SCHD-3` 终止。
  3. 日志可见多个关键 loop 已经被 `Prescheduled`，说明 Catapult 已经开始对修复后的 value-stage 进行长期调度搜索，而不是卡死在同一条反馈环上。

### 当前判断
- 根因修复方向是正确的：
  1. 原先导致快速失败的 `head_state_accum` / `tile_accum` / `lane_products` RAM 反馈链已被逐层拔掉。
  2. 当前阻塞已经从“错误型 schedule failure”转成“设计图仍偏厚，allocate 搜索时间过长”。
- 因此下一步主线不再是继续修同类 `SCHD-3`，而是进一步压缩 stage 内部设计图规模，重点应放在：
  1. 将 `ContextScorePacket` 等局部状态从全 `kNumAttentionHeads` 宽度继续收窄到实际 `head-group` 宽度。
  2. 减少 `prefill_attention_context_value_stream_stage_fp/core` 内仍保留的宽 packet 与 orchestration 扇出。
  3. 优先降低 `assembly`/`architect` 的 `Real ops`，避免 `allocate` 在当前规模上做超长时间搜索。