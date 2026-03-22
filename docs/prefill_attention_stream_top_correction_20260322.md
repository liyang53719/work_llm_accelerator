# Prefill Attention Stream Top 方向纠正记录

## 1. 背景

本次新增的 `qwen_prefill_attention_stream_top_catapult` 已经把顶层数据口改成了 `ac_channel`，并把单包宽度控制在 `256bit` 内。

但进一步检查后确认，这个版本仍然只是“接口形式正确了一步”，并没有满足片上 SRAM 预算与真正可综合的数据流边界要求。

核心问题不是顶层口型，而是 top 内部仍然采用了：

- channel boundary
- local full buffers
- 再调用旧版整层数组 kernel

这种写法会把原来“顶层大数组口不合理”的问题，转化成“顶层内部整层缓存不合理”的问题。

## 2. 这次纠正了什么

### 2.1 顶层运行时参数收缩

已经确认以下参数不应该继续作为顶层运行时输入存在：

- `attention_seq_tile`
- `attention_query_tile`
- `attention_key_tile`
- `attention_hidden_proj_tile`
- `attention_kv_proj_tile`
- `attention_head_dim_tile`
- `attention_query_heads_parallel`
- `attention_kv_heads_parallel`

原因：

- 这些量本质上决定的是综合后的微结构。
- 它们影响 tile 划分、并行度、bank 组织和 block 粒度。
- 这类参数应在综合前固定，而不是作为每次调用都变化的控制口。
- 因此它们不属于 `DirectInput` 语义，更不应作为最终 RTL top 的可变配置。

当前顶层已经收缩为只保留：

- `seq_len`
- `rms_eps`

其中 `seq_len` 仍属于真正的调用期控制；`rms_eps` 是否继续保留为运行时值，可在后续统一决定，但它也不应再与微结构参数混在一起。

### 2.2 `DirectInput` 不再滥用到过渡顶层

此前该 stream top 上给 `seq_len`、`rms_eps` 等参数加过 `DirectInput` pragma，但在当前函数形态下并未正确绑定，log 中出现了 `CIN-389`。

这说明：

- 当前这个过渡 top 还不是最终的稳定 compute block 形态。
- `DirectInput` 的绑定更适合真正稳定的 stage/block 级接口。
- 在该 top 上继续保留无效 pragma 只会制造噪声，误导后续日志判断。

因此，本轮已将这一层上的无效 `DirectInput` pragma 去掉。

### 2.3 微结构参数的正确归属

后续应把上述 tile / parallel 参数改成“固定宏参数”，而不是运行时口。

建议固定为以下综合期参数：

- `PREFILL_ATTENTION_SEQ_TILE`
- `PREFILL_ATTENTION_QUERY_TILE`
- `PREFILL_ATTENTION_KEY_TILE`
- `PREFILL_ATTENTION_HIDDEN_PROJ_TILE`
- `PREFILL_ATTENTION_KV_PROJ_TILE`
- `PREFILL_ATTENTION_HEAD_DIM_TILE`
- `PREFILL_ATTENTION_QUERY_HEADS_PARALLEL`
- `PREFILL_ATTENTION_KV_HEADS_PARALLEL`

推荐做法：

1. 用专门的配置头统一定义这些宏。
2. 在 Catapult 编译命令或统一配置头中固定它们。
3. 在源码中通过 `constexpr` 或 `static_assert` 把这些宏收敛到固定 tile config。
4. 不再把它们暴露到 RTL 接口上。

换句话说：

- `DirectInput` 只绑定真正“每次调用才变化”的控制量。
- 微结构参数属于“综合前固定”的编译期常量。

## 3. Local Full Buffers 容量核算

当前 stream top 内部仍然分配了整层 attention 所需的大量本地数组。

按当前容量上界计算：

- `kHiddenSize = 1536`
- `kNumKeyValueHeads = 2`
- `kHeadDim = 128`
- `kPrefillCatapultSeqCapacity = 128`
- `kPrefillCatapultKvWidth = 256`
- FP32 按 `4 Bytes`
- `packed_w4_t` 按 `1 Byte`

### 3.1 容量口径说明

这里需要区分两个不同的量：

- 逻辑矩阵规模：即真实权重元素个数，例如 `1536 x 1536 = 2304 Ki` 个 INT4 权值。
- 打包后存储字节：当前代码使用 `packed_w4_t = std::uint8_t`，每个 byte 存两个 INT4，因此存储字节数是逻辑元素数的一半。

也就是说：

- `1536 x 1536` 的 Q/O 权重，逻辑规模确实是 `2304 Ki` 个 INT4 权值。
- 但当前代码建模下的物理存储量是 `1152 KiB`。

当前代码相关依据：

- `packed_w4_t = std::uint8_t`
- 权重解码统一通过 `packed_weights[flat_index / 2]` 与高/低 nibble 访问

因此这次修正结论是：

- 之前文档把“逻辑规模”和“打包后存储字节”混写了。
- 代码当前按“INT4 packed storage”建模，本身没有在这一点上写错。

### 3.2 各数组大小

| 本地数组 | 逻辑规模 | 当前代码下存储量 | 说明 |
| --- | ---: | ---: | --- |
| `input_sequence` | 192 Ki FP32 | 768 KiB | 整个 seq tile 输入激活 |
| `input_layernorm_weight` | 1.5 Ki FP32 | 6 KiB | layernorm weight |
| `q_packed_weights` | 2304 Ki INT4 | 1152 KiB | Q 投影整块权重 |
| `k_packed_weights` | 384 Ki INT4 | 192 KiB | K 投影整块权重 |
| `v_packed_weights` | 384 Ki INT4 | 192 KiB | V 投影整块权重 |
| `o_packed_weights` | 2304 Ki INT4 | 1152 KiB | O 投影整块权重 |
| `q_bias` | 1.5 Ki FP32 | 6 KiB | Q bias |
| `k_bias` | 0.25 Ki FP32 | 1 KiB | K bias |
| `v_bias` | 0.25 Ki FP32 | 1 KiB | V bias |
| `q_scales` | 1.5 Ki FP32 | 6 KiB | Q scales |
| `k_scales` | 0.25 Ki FP32 | 1 KiB | K scales |
| `v_scales` | 0.25 Ki FP32 | 1 KiB | V scales |
| `o_scales` | 1.5 Ki FP32 | 6 KiB | O scales |
| `k_cache` | 32 Ki FP32 | 128 KiB | 当前 seq tile 的 K cache |
| `v_cache` | 32 Ki FP32 | 128 KiB | 当前 seq tile 的 V cache |
| `output_sequence` | 192 Ki FP32 | 768 KiB | 整个 seq tile 输出 |

### 3.2 总量

总静态本地存储约为：

- `4616192 Bytes`
- `4508 KiB`
- 约 `4.40 MiB`

## 4. 为什么这个数字直接否定了当前方案

项目当前已明确：

- 总 SRAM 上限最多 `2 MB`
- 实际目标要尽量向 `1 MB` 收敛

而当前单个 stream top 就已经静态占用了约 `4.40 MiB` 本地数组，问题包括：

1. 已经远超 `2 MB` 总预算。
2. 还没有计入 Catapult 可能为调度、banking、复制、FIFO 引入的额外资源。
3. 这类整层数组会继续推高 compile/assembly 图规模。
4. 它会掩盖真正应该做的事情，即：把“整层缓存”改成“tile 级工作集 + 外部 DDR 分批搬运”。

因此结论非常明确：

- `channel boundary + local full buffers` 只能作为极短期接口试验。
- 不能作为最终 prefill attention RTL 方案。

## 5. 现阶段正确的方向

后续必须从“整层缓存”切换到“工作集缓存”。

正确方向应为：

1. 顶层只保留 `ac_channel` 数据边界。
2. 微结构参数改为综合期固定宏参数。
3. 片上只缓存小而高复用的常驻量，或当前 tile 的高复用工作集。
4. 对放不下的大对象改成按 tile / panel / batch 从 DDR 读取。
5. 优先减少“被多次复读的大对象”的 DDR 读取次数，而不是盲目追求所有数据全量驻留。

## 6. 后续落地约束

后续任何新的 attention stream top，都应满足以下约束：

### 6.1 接口约束

- 数据口使用 `ac_channel<packet>`。
- 单个数据包不超过 `256bit`。
- 顶层不再暴露可变 tile / parallel 参数。

### 6.2 结构约束

- 不允许整层 `input_sequence` 本地化。
- 不允许整层 `q_packed_weights` / `o_packed_weights` 本地化。
- 不允许整层 `output_sequence` 本地化。
- 不允许为“沿 seq 全复用”的中间结果分配整层缓冲，只允许 tile 级或 panel 级缓冲。

### 6.3 预算约束

- 设计必须先过 `2 MB` 总 SRAM 上限。
- 方案评估时必须同时给出 `1 MB` 目标收敛路径。

## 7. 当前结论

本次 stream top 的主要价值是：

- 证明了顶层接口可以向 `ac_channel + <=256bit packet` 迁移。
- 同时也清楚证明了：如果内部仍沿用整层数组 kernel，那么即使接口形式变对，整体架构仍然不对。

因此，下一阶段不应继续围绕“整层缓存版 stream top”做 QoR 微调，而应转向：

- 拆分真正的 tile 级 stream core
- 固定微结构宏参数
- 基于 SRAM 预算重构片上工作集
