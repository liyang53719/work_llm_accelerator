# Performance Model

该目录保存 `work_llm_accelerator` 的系统级预算脚本与说明，用于在进入 HLS 设计前回答以下问题：

- Decode 的瓶颈主要在 MAC、DDR 还是 KV 访问。
- Prefill 在 blocked attention 下的 MAC、KV 和 SRAM 压力如何随 tile 改变。
- 在固定 `1 MB SRAM`、`2K MAC`、`1 GHz` 约束下，距离 `10 token/s` 目标还有多少差距。

## 当前文件

- `estimate_qwen_prefill_decode.py`
  - 联合预算 Qwen2.5-1.5B 的 decode、prefill 和 mixed workload
  - 支持调整 prompt 长度、decode 上下文长度、prefill tile 大小、权重位宽、KV 位宽和目标 token/s
  - 输出既包含总量，也包含归一化到 token 的近似口径

## 使用方式

在仓库根目录下运行：

```bash
python work_llm_accelerator/performance_model/estimate_qwen_prefill_decode.py
```

也可以指定参数，例如：

```bash
python work_llm_accelerator/performance_model/estimate_qwen_prefill_decode.py \
  --prompt-len 512 \
  --decode-context-len 2048 \
  --prefill-tile-m 64 \
  --target-decode-tok-s 10 \
  --target-prefill-tok-s 200
```

## 口径说明

- 这是系统级估算器，不是 HLS 周期精确模型。
- Decode 默认假设每生成一个 token，需要重新流过一遍完整模型权重。
- Prefill 默认假设每个 tile-M 块可复用一次权重，因此给出按 `prefill_tile_m` 摊薄后的等效权重字节数。
- KV 访问分为 decode 的全历史读取与 prefill 的写回，并给出可选的 blocked attention 粗估读流量。
- 后续如果建立更细的 banking、burst efficiency 和 kernel schedule 模型，应在此目录继续扩展而不是覆盖当前基线。