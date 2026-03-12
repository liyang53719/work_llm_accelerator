# Decode-Only HLS

该目录承载 Qwen2.5-1.5B 的 decode-only HLS 骨架，目标是先固定单 token attention/MLP 的接口与验证边界，再逐步收敛时延、banking 和 Catapult 设计点。

## 当前范围

- `qwen_decode_attention_kernel.*`
  - decode attention 主入口
  - 显式保留 `past_seq_len` 和 KV cache 接口
  - 当前已从 passthrough stub 推进到第一版 tile-aware 实现，包含 INT4 解包、RMSNorm、RoPE、分块投影、KV 追加和两遍 softmax 归约
- `qwen_decode_mlp_kernel.*`
  - decode MLP 主入口
  - 当前已从 passthrough stub 推进到第一版 tile-aware 实现，包含 post-attention RMSNorm、gate/up/down 和 residual add
- `run_catapult_decode_attention.tcl`
- `run_catapult_decode_mlp.tcl`

## 当前原则

- decode 路线单独优化，不继承 prefill 的 tile-M 假设。
- attention 与 MLP 分离，便于先查清单层时延和供数瓶颈。
- 整网正确性验证必须先于大规模 design sweep。
- 当前 attention/MLP kernel 已具备 smoke test，attention 还补了带非零历史 KV 的 regression，top wrapper 也有单独 regression，但仍未纳入更精确的量化口径和 Catapult directives。

## 下一步

1. 把更精确的量化口径和 reference 路径差异分析并入 attention/MLP kernel 验证。
2. 继续扩大 kernel-level regression，对更多真实 prompt 和更长历史 cache 做覆盖。
3. 再引入更细的 banking、pipeline 和 Catapult directives。