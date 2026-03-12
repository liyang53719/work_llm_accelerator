# Prefill-Only HLS

该目录承载 Qwen2.5-1.5B 的 prefill-only HLS 骨架，目标是先稳定 blocked attention 与 MLP 的接口、tile 形状和 Catapult 入口，再逐步收敛到可综合实现。

## 当前范围

- `qwen_prefill_attention_kernel.*`
  - prefill attention 主入口
  - 接口中显式保留 `tile_m`、`seq_len` 和 KV write-back 相关参数
- `qwen_prefill_mlp_kernel.*`
  - prefill 阶段的 MLP 主入口
- `run_catapult_prefill_attention.tcl`
- `run_catapult_prefill_mlp.tcl`

## 当前原则

- 先把接口和验证边界固定下来，再决定内部 banking、pipeline 和 tile 细节。
- 不把 decode-only 的 `M=1` 假设带入 prefill。
- 一旦 HLS 骨架建立，优先通过 `verification/` 中的整网 reference 校验去约束接口行为。

## 下一步

1. 定义 blocked attention 的内部 buffer 组织。
2. 建立 prefill reference case 生成脚本。
3. 将 HLS 入口与整网验证框架中的 prefill backend 对齐。