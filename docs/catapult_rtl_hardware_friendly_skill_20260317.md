# Catapult 生成 RTL 硬件友好风格 Skill

## Skill 名称

Catapult HLS 分层建模与 RTL 友好接口风格

## 适用范围

- 需要把单个 stage 单独综合成 RTL 的 Catapult HLS 设计。
- 设计中同时存在 host/TB 级 DDR 大窗口访问和 tile 级计算内核。
- 需要把大数组顶层改成流接口、共享缓冲和寄存器配置接口。

## 本次重点浏览的 Catapult shared 文档与例子

- `shared/include/ac_channel.h`
- `shared/examples/methodology/DirectOutputUsage/fir_channel.cpp`
- `shared/examples/methodology/DirectOutputUsage/fir_channel_pragma.cpp`
- `shared/examples/methodology/Memory_Stagewise_Enable/memory_stagewise_enable.pdf`
- `shared/examples/methodology/Memory_Stagewise_Enable/src/ac_channel_swen.cpp`
- `shared/examples/methodology/Memory_Stagewise_Enable/src/ac_shared_swen.cpp`
- `shared/examples/methodology/ChannelBroadcastAndDemux/ChannelBroadcastAndDemux.pdf`
- `shared/training/walkthroughs/imageproc/slidingwindow/README.txt`
- `shared/training/walkthroughs/imageproc/slidingwindow/EdgeDetect_Hierarchy.h`
- `shared/training/walkthroughs/imageproc/slidingwindow/EdgeDetect_SinglePort_Programable.h`
- `shared/examples/ac_ipl/ac_filter_2d/ac_filter2d.tcl`
- `shared/examples/docs/directives/map_to_module/readme.txt`
- `shared/examples/docs/directives/map_to_module/pragma.cpp`
- `shared/examples/docs/modeling/cpp/coeffs/reg_directinput.h`

## 从文档提炼出的核心理念

### 1. 顶层要表达真实硬件边界，而不是软件调用边界

- 若函数参数还是完整 sequence、大块权重矩阵、完整 cache 窗口，这个函数本质上还是 host/TB 壳，不是 RTL 计算核。
- Catapult 推荐把真正的综合 top 建成“稳定、固定尺寸、单职责”的硬件块。
- DDR 地址解析、窗口越界检查、tile 切分、数据搬运，应该留在 host wrapper 或上一级 orchestrator。

### 2. 块间通信优先用 channel，而不是裸数组接口

- `ac_channel` 是 Catapult 推荐的层次化流接口。
- 层次块之间如果是生产者/消费者关系，应用 channel 表达数据流，而不是让下一级再去感知上一级的大数组布局。
- `ChannelBroadcastAndDemux` 文档进一步说明，channel 组能被 Catapult 识别并做共享寄存器、广播和 demux 优化。

### 3. 共享缓冲要显式建模成 shared memory，而不是隐式复用数组

- `Memory Stagewise Enable` 和 `ac_channel_swen.cpp` / `ac_shared_swen.cpp` 说明：块间共享缓冲要么是 `ac_channel<ac_array<...>>`，要么是 `ac_shared` 加 `ac_sync`。
- 这样 Catapult 才能识别 wait controller、skid buffer、stagewise enable 与共享存储仲裁边界。
- 直接把同一个大数组在多个 ccore 中切片传递，往往既不利于接口收缩，也不利于 memory architecture 优化。

### 4. 可编程常量和控制标量应该映射成 DirectInput

- `reg_directinput.h` 和 `map_to_module/readme.txt` 说明，单个常量或小控制寄存器应映射到 `[DirectInput]`。
- 这样 Catapult 不会像普通 `ccs_in` 一样在第一拍寄存输入，而是把相关逻辑尽量后移到真正使用的时刻。
- 适合做 DirectInput 的对象：`width/height/bypass` 这类标量配置、tile 有效长度、选择位、控制寄存器。
- 不适合做 DirectInput 的对象：大数组权重、大 tile activation、大块 cache 数据。

### 5. 滑窗/2D 卷积例子强调“先拆缓冲块和 PE，再拼顶层”

- `EdgeDetect` walkthrough 明确展示了一条演化链：
  算法版 -> bit-accurate -> synthesizable -> memory architecture -> hierarchy -> single-port -> programmable -> circular buffer。
- `ac_filter_2d` 例子进一步把 `lineBuffer`、`filterPE`、`vldComp` 先独立综合，再在 `peArray` 和最终 top 中复用这些 ccore。
- 这说明 Catapult 的推荐风格不是把所有行为写进一个顶层，而是把“缓冲”“算子”“valid/控制”拆成独立模块，再由上层组装。

## 反模式清单

### 反模式 1

把完整 `seq_len x hidden_size` 激活、完整权重矩阵、完整 cache 窗口同时挂在一个 stage top 上。

问题：

- 顶层不是硬件块，而是软件壳。
- 接口过大，assembly/architect 复杂度极高。
- 容易触发接口缩减、资源推断、调度时间爆炸等问题。

### 反模式 2

在顶层中一边做 tile 现场切分，一边做完整运算，一边再承担参数合法性检查。

问题：

- host 语义与 compute kernel 语义混杂。
- 不利于单独提取可复用 ccore。
- 使 top 既不像 wrapper，也不像 kernel。

### 反模式 3

把应该是配置寄存器的标量和大数据口同样建模成普通输入。

问题：

- 调度过早锁定。
- 额外输入寄存器与不必要的时序压力。

### 反模式 4

在同一层循环上同时施加全展开和 pipeline 约束。

问题：

- 会出现 `Cannot pipeline a fully unrolled loop (LOOP-21)`。
- 即使工具继续运行，也往往是在错误状态下浪费时间和内存。

## 本工程当前问题判断

### `qwen_prefill_attention_kv_cache_stage` 当前问题

- 旧的 stage 级接口仍然偏向 host/TB 语义，承载了 sequence、整权重、整 cache 的视角。
- 新增的 `kv tile` 虽然把计算维度缩小到了 `64 x 64`，但仍然在 top 内部使用数组块接口，而不是 channel 流接口。
- 这导致 top 仍然不是 Catapult 文档里最推荐的“分层块间流式”风格。

### 当前日志现象说明什么

- 旧 run 中的 `LOOP-21` 说明调度约束和展开约束冲突。
- 即便修掉这个点，如果顶层建模方式仍然是“单 top 同时承担搬运和计算”，编译/assembly 依然可能很重。

## 本工程建议的顶层重构方向

### 分 4 层建模

1. `host/tb wrapper` 层

- 处理 DDR 地址、窗口边界、tile 切分、软件可见 descriptor。
- 不作为要单独抽取 RTL 的 compute top。

2. `tile loader / packer` 层

- 从外部存储窗口中读取一个 tile。
- 打包成 channel payload，例如 input tile、weight tile、scale tile、partial-sum tile。
- 输出到下一级 compute core。

3. `compute core` 层

- 真正给 Catapult 设 top 的模块。
- 接口全部使用 `ac_channel<packet>`。
- 单值配置如 `inv_rms`、`lane_extent`、`out_extent` 用 `DirectInput`。

4. `post-process / store` 层

- 接收计算结果 channel。
- 写回 tile partial sum、cache 或下一级 block。

### `kv_cache` 路径的具体接口建议

建议综合 top 改成如下风格：

- `ac_channel<KvFpTilePacket>` 输入 token tile
- `ac_channel<KvFpTilePacket>` 输入 layernorm weight tile
- `ac_channel<KvPackedTilePairPacket>` 输入 K/V 权重 tile
- `ac_channel<KvScaleTilePairPacket>` 输入 K/V scale tile
- `ac_channel<KvPartialTilePairPacket>` 输入 partial sum tile
- `DirectInput`：`inv_rms`
- `DirectInput`：`lane_extent`
- `DirectInput`：`out_extent`
- `ac_channel<KvPartialTilePairPacket>` 输出更新后的 partial sum tile

这样综合 top 就只代表一个稳定的 tile MAC/RMSNorm 子核，不再携带 sequence 或 DDR 语义。

## 本工程下一轮改造计划

### 第一阶段

- 用 `ac_channel` 顶层替换当前数组型 tile top。
- 保留数组型 tile 计算函数作为内部 helper。
- Tcl 的 `top_function` 指向新的 stream top。

### 第二阶段

- 增加 `tile loader / store` 包装块，把 sequence / weight / cache 的大数组访问从 compute top 移出去。
- 将 `k_bias`、`v_bias`、`inv_rms`、`lane_extent`、`out_extent` 中适合寄存器化的量拆成 DirectInput 或小寄存器包。

### 第三阶段

- 对 `q_context_output` 路径按同样风格重构。
- score/context 路径进一步参考 sliding window / hierarchy 风格，拆成 line-buffer-like cache reader、score core、softmax/context core。

## 执行检查表

- 顶层是否只表达单一硬件块职责。
- 顶层是否已经去掉完整 DDR window 数组口。
- 块间数据是否优先通过 `ac_channel` 传递。
- 共享状态是否显式建模成 `ac_shared` 或 channel-backed shared memory。
- 单值配置是否映射成 `DirectInput`。
- 每个 pipeline loop 是否避免和全展开约束冲突。
- 每个 ccore 是否有稳定、固定、可复用的接口边界。