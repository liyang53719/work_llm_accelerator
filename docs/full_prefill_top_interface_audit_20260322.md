# Full Prefill 顶层接口审计与重构清单

## 1. 结论

- 当前 full prefill Catapult flow 的 `top_function` 仍然是大数组 memory top，不符合本工程已经明确写入文档的 `RTL 接口硬约束`。
- 现在的问题不是某几个端口位宽偶尔超限，而是综合边界的建模层级整体偏向 `host/orchestrator + DDR window` 语义，而不是 `compute block + ac_channel + DirectInput` 语义。
- 当前 full prefill flow 能继续向前推进到 `assembly`，说明头文件与局部建模合法性问题正在被逐步清除；但如果顶层边界不改，后续仍会在 compile/assembly/architect 阶段持续承担不必要的图膨胀与 memory-port 建模压力。

## 2. 当前约束基线

### 2.1 项目内约束

`docs/project_plan.md` 已明确写入：

- 所有用于 Catapult 生成 RTL 的 `top` / `ccore` 函数接口，默认必须优先落实为：
  1. `ac_channel<...>` 数据口
  2. `<=256bit` 的标量/寄存器配置口
- 大数组口如整 tile array、整 token array、整 cache window，默认不应直接挂在 `top` / `ccore` 边界上。

### 2.2 Catapult 本地文档约束

本地 Catapult 2026.1 文档中已确认以下规则：

- `C-style arrays mapped to memories are only supported for primary inputs/outputs`
- `Merging of C-style arrays onto a shared resource is not supported between blocks`
- `Catapult doesn't have any frame of reference to reduce most variables declared on the top-level function interface`
- `For consistency, the number of bits on a port variable is never changed`

这几条规则与本工程此前出现的 `CIN-84`、compile 图膨胀、child-top 混合 channel/pointer 边界问题完全一致。

## 3. 当前 full prefill 调用链

### 3.1 现状调用链

- `qwen_prefill_top_catapult(...)`
  - 当前 Catapult Tcl 的 `top_function`
  - 对外暴露完整 DDR/SRAM 大数组口
  - 同时做参数校验、地址换算、窗口范围检查、kernel 调度
- `qwen_prefill_top_core(...)`
  - 接收 `PrefillLayerDescriptor + PrefillTopLevelPorts`
  - 派生各个权重/scale/cache 指针
  - 串行调用 attention kernel 与 MLP kernel
- `qwen_prefill_attention_kernel(...)`
  - 当前仍以大数组/指针接口为主
- `qwen_prefill_mlp_kernel(...)`
  - 当前仍以大数组/指针接口为主

### 3.2 关键观察

- `PrefillTopLevelPorts` 已经把“所有大数组资源”聚合成一个 wrapper 视图。
- 这说明代码抽象层已经暗含了“外层保留 memory 语义、内层收敛 compute 语义”的方向。
- 但 Catapult 目前仍把 `qwen_prefill_top_catapult` 当最终 top，因此没有真正沿着这个抽象切到 stream/compute 边界。

## 4. 当前不符合约束的具体点

### 4.1 top_function 选在了错误层级

当前 Tcl：

- `hls/prefill_only/run_catapult_prefill_attention.tcl`
  - `set top_function llm_accel::qwen_prefill_top_catapult`

问题：

- 这个函数本质上是系统级 wrapper，不是单职责 compute block。
- 它承担了：
  1. 配置合法性检查
  2. memory window 边界检查
  3. base address 到 typed pointer 的派生
  4. attention / mlp 两大子核的串接
- 这与 Catapult 文档推荐的“稳定、固定尺寸、单职责”的 top 语义相反。

### 4.2 顶层接口仍然是 DDR/SRAM 大数组

当前 `qwen_prefill_top_catapult(...)` 直接暴露：

- `weight_ddr[...]`
- `scale_ddr[...]`
- `kv_cache_ddr[...]`
- `activation_ddr[...]`
- `weight_sram[...]`
- `kv_sram[...]`
- `partial_sum_sram[...]`
- `softmax_sram[...]`
- `control_sram[...]`

问题：

- 这些端口虽然被 Tcl 绑成了 AXI/memory resource，但接口抽象仍然是“大块 window”语义。
- 从工程约束看，这类端口应只停留在 wrapper 或 primary-memory top 一侧，而不是直接出现在要优化的 compute top 上。

### 4.3 child kernel 仍保留大量数组边界

`qwen_prefill_attention_kernel.h` 当前仍大量暴露：

- `q_proj_buffer[...]`
- `k_cache[...]`
- `v_cache[...]`
- `context_buffer[...]`
- 以及多个二维/一维 array stage 接口

问题：

- 即使顶层换成 wrapper，若 child-top 之间仍以数组相连，仍然不符合 Catapult 对块间 channel/shared-memory 的推荐风格。
- 这与此前 context flow 总结中“只把外层做成 channel，但 child-top 仍残留 memory pointer/array 边界”的问题是同一类根因。

### 4.4 当前 Tcl 绑定放大了 memory-top 语义

当前 full prefill Tcl 在 `go assembly` 后直接做：

- `weight_ddr` -> `amba.ccs_axi4_slave_mem DATA_WIDTH=8`
- `scale_ddr` -> `amba.ccs_axi4_slave_mem DATA_WIDTH=32`
- `kv_cache_ddr` -> `amba.ccs_axi4_slave_mem DATA_WIDTH=32`
- `activation_ddr` -> `amba.ccs_axi4_slave_mem DATA_WIDTH=32`

问题：

- 从“总线物理位宽”看，它们没有超过 `256bit`。
- 但从“综合边界建模”看，这一步是在把 monolithic wrapper 进一步固化为 memory top，而不是推动 compute top 收敛为 channel-only。

### 4.5 当前 solution 有边界混杂信号

从当前 `messages.txt` 可见：

- `Multiple tops detected`
- `Original pragma 'top' overridden by 'block'`

这不是当前最主要的错误，但说明现有源码里已经同时存在多个潜在 top / block 入口，而当前 solution 仍未形成一条清晰的自顶向下接口策略。

## 5. 当前哪些部分是对的

### 5.1 已有 stream-top 模式可以直接复用

当前仓库里已经存在符合目标风格的局部 top：

- `qwen_prefill_attention_context_query_tile_stream_catapult(...)`
- `qwen_prefill_attention_q_context_output_tile_stream_catapult(...)`
- `qwen_prefill_attention_kv_cache_stage.cpp` 中的 tile stream top

这些函数的共同特征：

- 数据口全部使用 `ac_channel<packet>`
- 配置口使用 `DirectInput`
- 算法核心与 loader/store 边界相对清晰

这说明目标风格在本工程里已经不是“理论建议”，而是已有成功模式。

### 5.2 `PrefillTopLevelPorts` 已经提供了 wrapper 侧抽象

`PrefillTopLevelPorts` 可以继续保留，但应重新定位：

- 它属于 host/wrapper/memory-facing 层
- 不应继续直接对应最终 Catapult compute top 的接口签名

## 6. 推荐的 full prefill 重构层次

### 6.1 层次划分

建议将 full prefill 重新明确分为四层：

1. `dispatch / host wrapper` 层

- 继续保留 `PrefillLayerDescriptor`
- 继续处理 layer dispatch、地址解析、窗口合法性检查
- 不作为主优化目标，也不作为主要 compute RTL top

2. `memory loader / packer` 层

- 从 `weight_ddr` / `scale_ddr` / `kv_cache_ddr` / `activation_ddr` 中读取数据
- 打成 `<=256bit` 的 word packet / tile packet
- 将 memory 语义转换为 channel 语义

3. `compute stream core` 层

- 作为真正的 Catapult top 或 child-top
- 接口统一为：
  - `ac_channel<packet>` 数据口
  - `DirectInput` 配置口
- 不直接接触 DDR window 或裸数组 cache

4. `store / unpack` 层

- 接收结果 channel
- 回写 activation / kv cache / partial sum

### 6.2 full prefill 的建议模块化边界

建议不要继续把 attention 和 mlp 都塞在一个 monolithic top 中。

优先拆成：

1. `prefill_attention_memory_wrapper`

- memory-facing
- 负责：输入激活、layernorm weight、Q/K/V/O weight/scales、KV cache 的打包和回写

2. `prefill_attention_stream_core`

- channel-facing
- 内部由现有可复用 stream stage 串接

3. `prefill_mlp_memory_wrapper`

- memory-facing
- 负责 residual / norm / gate-up-down weights/scales 的打包和回写

4. `prefill_mlp_stream_core`

- channel-facing
- 作为新的 MLP compute top

5. 可选的更外层 `prefill_layer_orchestrator`

- 若后续确实需要 system-level RTL shell，再在这一层组合 attention wrapper 和 mlp wrapper
- 但不应把它作为当前主要 QoR 优化目标

### 6.3 `catapult_prefill` 弃用 / 降格清单

这一节的目的不是删除现有 `catapult_prefill` 路径，而是明确：哪些对象应冻结为 wrapper 语义，不再继续作为主线 QoR、timing 和 legality 的优化对象。

1. 需要明确降格为 wrapper 的函数

- `llm_accel::qwen_prefill_top_catapult(...)`
  - 新定位：`memory-facing wrapper`
  - 保留职责：descriptor 校验、memory window 校验、base address 到 typed pointer 的换算、串接 attention / mlp 路径
  - 不再承担：full prefill 主优化 top、主线 RTL timing/QoR 收敛入口
- `llm_accel::qwen_prefill_top_catapult_fine(...)`
  - 新定位：`debug / decomposition wrapper`
  - 保留职责：粗细粒度 stage 串接与交叉验证
  - 不再承担：完整 block RTL 主入口
- `llm_accel::qwen_prefill_top_core(...)`
  - 新定位：`wrapper-side orchestration core`
  - 保留职责：基于 `PrefillLayerDescriptor + PrefillTopLevelPorts` 做资源派生与 kernel 串接
  - 不再承担：未来 compute-core 的接口定义来源

2. 应冻结在 wrapper 层、不得继续向 compute core 传播的接口语义

- DDR window 大数组口：
  - `weight_ddr[...]`
  - `scale_ddr[...]`
  - `kv_cache_ddr[...]`
  - `activation_ddr[...]`
- SRAM scratch 大数组口：
  - `weight_sram[...]`
  - `kv_sram[...]`
  - `partial_sum_sram[...]`
  - `softmax_sram[...]`
  - `control_sram[...]`
- 不再作为最终 compute top 运行时口的微结构参数：
  - `attention_seq_tile`
  - `attention_query_tile`
  - `attention_key_tile`
  - `attention_hidden_proj_tile`
  - `attention_kv_proj_tile`
  - `attention_head_dim_tile`
  - `attention_query_heads_parallel`
  - `attention_kv_heads_parallel`
  - `mlp_seq_tile`
  - `mlp_hidden_tile`
  - `mlp_ff_tile`

3. 当前应视为“冻结区”的逻辑

- `valid_synth_tile_args(...)` 这类综合期结构参数校验
- base address 到 pointer 的换算逻辑
- memory window range check
- monolithic top 上的 AXI memory resource 绑定前提

4. 仍应保留、不视为废弃的对象

- `PrefillLayerDescriptor`
- `PrefillTopLevelPorts`
- `qwen_prefill_top_wrapper(...)`
- `qwen_prefill_dispatch_layers(...)`

5. 执行判定规则

- 如果某项修改主要是在修 `qwen_prefill_top_catapult(...)` 的大数组端口、`qwen_prefill_top_catapult_fine(...)` 的 monolithic 调度，或 wrapper 顶层上的 AXI memory resource 绑定细节，就必须先判断它是在帮助未来 compute-core 收敛，还是只是在延长旧 memory-top 主线的生命周期。
- 若答案偏向后者，则该修改默认降级为次优先级，不再作为当前 block 主线推进目标。

## 7. 迁移优先级

### 第一优先级

先让当前 full prefill flow 的主优化 top 不再是 `qwen_prefill_top_catapult`。

目标：

- 将 `qwen_prefill_top_catapult` 明确降格为 wrapper
- 新增一个真正的 `prefill_attention_stream_core` 作为当前主线 RTL top

理由：

- attention 是当前 compile/assembly 压力主源
- 仓库内已经存在可复用的 channel-only stage 模式
- 先把 attention 主线收敛，收益最大、风险最小

### 第二优先级

补齐 MLP 的 stream-core 化。

当前 `qwen_prefill_mlp_kernel` 仍是数组接口，建议：

- 参考现有 attention tile stream top 的模式
- 为 MLP 引入 `channel + DirectInput` 的 tile compute top

### 第三优先级

最后再决定是否要做 full layer orchestration top 的 RTL 集成。

也就是说，顺序应为：

1. 先把 `attention` 收敛成 channel-only compute top
2. 再把 `mlp` 收敛成 channel-only compute top
3. 最后才考虑是否需要一个更外层的 `layer orchestrator` RTL top

而不是反过来先把整层都塞进一个 top 再逐个补丁修 legalize 问题。

## 8. Tcl 设置建议

### 8.1 当前不建议继续维持的设置

- 继续以 `llm_accel::qwen_prefill_top_catapult` 作为 `top_function`
- 在这个 monolithic top 上直接绑定四个 AXI memory resource

### 8.2 建议的新 flow 组织方式

建议拆成两类 Tcl：

1. `compute-core` flow

- `top_function` 指向 channel-only stream core
- 主要用于 QoR、compile/assembly/architect 调优

2. `memory-wrapper/system-integration` flow

- `top_function` 指向 memory-facing wrapper
- 只在 compute core 稳定后再做系统级接口联调

这样可以避免所有问题都在一个 solution 里互相叠加。

### 8.3 对现有 `catapult_prefill` flow 的直接处置建议

当前 `make catapult_prefill` 不建议再承担“追第一版完整 block RTL”的主线目标，建议直接按以下口径处理：

1. 短期

- 保留 `make catapult_prefill`，但明确它只是 wrapper / integration 参考流。
- 不再以它推进到某个 Catapult 阶段，作为 block 主线是否收敛的主要判断标准。

2. 中期

- 新增或组装 attention compute-core flow。
- 为 MLP 新增 compute-core flow。
- 只有这两个 compute core 稳定后，才重新定义 full-block integration flow。

3. 长期

- 若新的 full-block integration flow 成熟，当前 monolithic `catapult_prefill` 可以进一步降为历史兼容入口，或仅保留给 memory 接口联调。

## 9. 当前可直接执行的改造顺序

### Step 1

冻结 `qwen_prefill_top_catapult` 为 wrapper 角色，不再把它作为主优化 top。

补充要求：

- 不再围绕 `qwen_prefill_top_catapult` 扩展新的 compute-side 特性。
- 若必须修改它，修改范围仅限于：
  - wrapper 语义澄清
  - descriptor / memory window 口径对齐
  - 与新 compute core 的连接适配

### Step 2

基于现有 attention stream-stage，新增或组装一个 `full attention stream core`：

- 输入：activation / norm / weight / scale / kv 数据的 word/tile channel
- 输出：attention residual / updated kv 的 channel
- 配置：`seq_len`、tile 参数、并行度等 DirectInput

### Step 3

把 current memory arrays 留在 loader/store wrapper，一律不再穿过 compute core 边界。

### Step 4

为 MLP 补一套相同风格的 stream compute top。

补充要求：

- 这一阶段应被视为“拿到完整 block RTL”的主线缺口。
- 默认不要再新建面向 memory-top 的 monolithic MLP 顶层；优先直接做 `prefill_mlp_stream_core` 或同等语义的 channel-only compute top。

### Step 5

等 attention / mlp 两个 compute core 都稳定后，再评估是否需要 layer-level integration RTL top。

### Step 6

只有在 attention core 与 mlp core 都稳定后，才重新定义“完整 block RTL”：

- wrapper 负责 memory-facing pack/unpack 与 descriptor 解释
- block 级 compute 由 attention core + mlp core 组合
- 此时再决定新的 full-block integration flow 是否替代当前 `catapult_prefill`

## 10. 一句话判断标准

后续任何一个用于 Catapult 主线收敛的 `top/ccore`，都先问两个问题：

1. 这个边界上是否还残留了整块 DDR/cache/activation 数组口？
2. 这个函数是否仍然同时承担了地址解析、边界检查、tile 拆分和完整计算？

只要其中任意一个答案是“是”，它大概率就还不是应该被当作主优化目标的 compute top。