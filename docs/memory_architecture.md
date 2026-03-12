# Memory Architecture

## 目标

在 1 MB SRAM 约束下，明确 Qwen2.5-1.5B prefill/decode 加速器的 DDR 与片上 SRAM 职责边界，并为后续 Catapult top-level 接口提供固定口径。

## 总体原则

- 整模型 INT4 权重常驻 DDR。
- 长上下文 KV cache 常驻 DDR。
- 1 MB SRAM 只保存当前 tile 需要的工作集。
- decode 与 prefill 共享同一套地址空间原则，但工作集切分不同。

## 片上 SRAM 建议分区

| 区域 | 容量 | 作用 |
| --- | ---: | --- |
| weight ping-pong | 256 KB | 当前 layer/tile 的权重分块搬运 |
| KV working set | 256 KB | 当前 tile 依赖的 K/V 片段 |
| partial sums | 128 KB | matmul / reduction 累加缓存 |
| softmax scratch | 128 KB | max / sum / exp / normalize 临时区 |
| control scratch | 256 KB | norm 中间量、descriptor decode、边界处理 |

总计：1024 KB。

## DDR 侧数据布局

### 权重区

- 按 layer 顺序排布。
- 每层内部再按 `q/k/v/o/gate/up/down` 和 norm/scales 分段。
- decode 与 prefill 共用同一权重地址布局。

### KV cache 区

- 按 `layer -> token -> kv_head -> head_dim` 线性排布。
- decode 每步只搬当前层所需的历史 token 窗口。
- prefill 以 block 为单位读写，避免一次性把整段 KV 驻留片上。

### activation 区

- 输入 token 或 sequence block 常驻 DDR。
- 每层输出回写 DDR，供下一层继续读取，除非后续证明 layer fusion 值得做。

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