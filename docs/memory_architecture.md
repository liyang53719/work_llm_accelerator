# Memory Architecture

## 目标

在 1 MB SRAM 约束下，明确 Qwen2.5-1.5B prefill/decode 加速器的 DDR 与片上 SRAM 职责边界，并为后续 Catapult top-level 接口提供固定口径。

## 总体原则

- 整模型 INT4 权重常驻 DDR。
- 长上下文 KV cache 常驻 DDR。
- 1 MB SRAM 只保存当前 tile 需要的工作集。
- decode 与 prefill 共享同一套地址空间原则，但工作集切分不同。
- attention 路径统一按 `32 x 64` 乘法器阵列组织 micro-tile；SRAM 分区、搬运节拍和局部 buffer 大小都以这个固定阵列为基准，而不是按可变阵列估算。

## 片上 SRAM 建议分区

| 区域 | 容量 | 作用 |
| --- | ---: | --- |
| weight ping-pong | 256 KB | 当前 layer/tile 的权重分块搬运 |
| KV working set | 256 KB | 当前 tile 依赖的 K/V 片段 |
| partial sums | 128 KB | matmul / reduction 累加缓存 |
| softmax scratch | 128 KB | max / sum / exp / normalize 临时区 |
| control scratch | 256 KB | norm 中间量、descriptor decode、边界处理 |

总计：1024 KB。

## `32 x 64` attention 阵列对 SRAM 的直接含义

- attention projection 的单个 Psum micro-tile 固定为 `32 x 64 x 4 Bytes = 8 KB`。
- attention score 的单个 score tile 固定为 `32 x 64`；若用 FP32 累加或缓存，对应 `8 KB` scratch。
- `head_dim = 128`，所以 `Q @ K^T` 和 `P @ V` 的 reduction 主路径固定拆成两个 `64` 宽 pass，不能再假设单次吞掉完整 128 维。
- 对 `Q/K/V/O projection`，weight buffer 的最小有效搬运粒度应围绕 `64` 个输出列或 `64` 个 reduction 列设计。
- 对 `seq_len` 方向，主路径以 32 行为一个 query/token tile；`11/65` 等长度依然通过尾块处理解决。

这意味着 memory layout 的目标不是“支持任意 attention 阵列形状”，而是为固定 `32 x 64` 节拍提供稳定的 ping-pong 和 Psum 轮转空间。

## DDR 侧数据布局

### 权重区

- 按 layer 顺序排布。
- 每层内部再按 `q/k/v/o/gate/up/down` 和 norm/scales 分段。
- decode 与 prefill 共用同一权重地址布局。

### KV cache 区

- 按 `layer -> token -> kv_head -> head_dim` 线性排布。
- decode 每步只搬当前层所需的历史 token 窗口。
- prefill 以 block 为单位读写，避免一次性把整段 KV 驻留片上。
- 在固定 `32 x 64` attention 阵列下，建议历史 token 窗口优先按 64 个 key/value token 为一批搬运，以直接匹配 score/context 的 `T_k=64` 主节拍。

### activation 区

- 输入 token 或 sequence block 常驻 DDR。
- 每层输出回写 DDR，供下一层继续读取，除非后续证明 layer fusion 值得做。
- 若后续采用 K/V-first + Q 直通 score/context 的调度，则 Q 不应作为完整 activation 落 DDR，而应在局部 SRAM / 寄存器路径内完成 RoPE 与后续 score 计算。

## 为什么当前 reference 接口不是最终接口

- 现在的 reference wrapper 直接把多个权重张量和缓存数组作为函数参数传入，是为了快速做 host 侧数学对齐。
- 这种接口不对应 1 MB SRAM 约束，也不对应最终 RTL 的 AXI master 接口。
- 最终综合对象应该是“descriptor + AXI 端口 + 局部 SRAM scratch”，不是“超长裸参数表”。

## 计划中的 RTL-facing 接口

### 控制侧

- `layer_id`
- `past_seq_len` 或 `seq_len`
- `tile_m`
- `input/output` DDR base
- `weight/scales` DDR base
- `k/v cache` DDR base
- `scratch` base 或局部 SRAM 分区选择

### 存储侧

- AXI master read：权重、scale、KV、输入激活
- AXI master write：输出激活、更新后的 KV

## 实现顺序

1. 用 descriptor 固定多层复用边界。
2. 用 memory layout 头文件固定 DDR 地址口径和 SRAM 分区预算。
3. 再把 decode/prefill kernel 从 reference wrapper 迁移到可综合的 tile 化实现。
4. Catapult top-level 只接 descriptor 和 AXI 端口，不再暴露 layer-specific 参数列表。