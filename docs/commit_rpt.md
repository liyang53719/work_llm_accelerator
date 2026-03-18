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