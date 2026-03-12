# Layer Dispatch Design

## 目标

把当前 `layer0` 参考路径收敛成“单层数学金标 + 多层调度”，而不是继续复制 `layer1`、`layer2` 的专用实现。

## 核心原则

- 单层 kernel 或 reference wrapper 只负责“当前层”的计算。
- 28 层复用通过 `layer_id`、DDR 基址和 stride 计算完成。
- 不为每一层综合一套 RTL；综合的是单层 kernel 或少量 top-level 变体。

## 调度边界

### Host 或 firmware 负责

- 维护 layer loop。
- 根据 `layer_id` 计算当前层的权重和 scale 基址。
- 根据 `layer_id` 和 `seq_len` 计算 KV cache 读写位置。
- 对 decode 和 prefill 生成不同 descriptor。

### 单层 kernel 负责

- 消费当前层 descriptor。
- 从 DDR 搬运当前 tile 所需的权重和 KV working set 到 SRAM。
- 完成 attention、MLP、norm、residual 的片上调度。
- 把输出张量和更新后的 KV 写回 DDR。

## 为什么不再写 layer1 专用实现

- Qwen2.5-1.5B 的 28 层拓扑同构，差异主要是参数值和 KV 地址，不是执行结构。
- 当前 `layer0` 路径的价值在于固定数学口径，不在于成为 28 个专用实现的模板复制源。
- 如果继续复制 layer1，会把验证便利性误当成架构收敛，后续还要再拆回 descriptor 驱动模式。

## 计划中的接口

- `DecodeLayerDescriptor`
  - `layer_id`
  - `past_seq_len`
  - `input_token_addr`
  - `output_token_addr`
  - `layer_weights_base_addr`
  - `layer_scales_base_addr`
  - `k_cache_base_addr`
  - `v_cache_base_addr`
  - `scratch_base_addr`
- `PrefillLayerDescriptor`
  - 在 decode descriptor 基础上增加 `seq_len`、`tile_m`

当前已落下首版代码骨架：

- `hls/common/llm_layer_dispatch.h`
- `hls/common/llm_memory_layout.h`
- `hls/decode_only/qwen_decode_top_wrapper.h`
- `hls/decode_only/qwen_decode_top_wrapper.cpp`
- `hls/prefill_only/qwen_prefill_top_wrapper.h`
- `hls/prefill_only/qwen_prefill_top_wrapper.cpp`
- `python/layer_descriptor_builder.py`
- `verification/validate_layer_dispatch_layout.py`

## 验证分层

1. 单层数学验证：继续使用 `layer0` prefill/decode reference case。
2. 多层调度验证：新增 all-layer dispatch validator，按 layer loop 调单层 reference wrapper。
3. RTL 接口验证：检查 descriptor 到 DDR 地址映射是否与模型层序一致。

## 下一步

1. 为 decode 增加 top-level wrapper 草图，固定 descriptor + DDR/SRAM 端口边界。
2. 为 prefill 定义对应的 blocked attention descriptor。
3. 在 full-model validator 之外增加多层 dispatch validator，避免继续扩展 layer-specific C ABI。