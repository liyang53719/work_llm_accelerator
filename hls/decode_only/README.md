# Decode-Only HLS

该目录承载 Qwen2.5-1.5B 的 decode-only HLS 骨架，目标是先固定单 token attention/MLP 的接口与验证边界，再逐步收敛时延、banking 和 Catapult 设计点。

## 当前范围

- `qwen_decode_attention_kernel.*`
  - decode attention 主入口
  - 显式保留 `past_seq_len` 和 KV cache 接口
  - 当前已从 passthrough stub 推进到第一版 tile-aware 实现，包含 INT4 解包、RMSNorm、RoPE、分块投影、KV 追加和两遍 softmax 归约
- `qwen_decode_mlp_kernel.*`
  - decode MLP 主入口
- `run_catapult_decode_attention.tcl`
- `run_catapult_decode_mlp.tcl`

## 当前原则

- decode 路线单独优化，不继承 prefill 的 tile-M 假设。
- attention 与 MLP 分离，便于先查清单层时延和供数瓶颈。
- 整网正确性验证必须先于大规模 design sweep。
- 当前 attention kernel 已具备最小 smoke test 和 top-wrapper regression，但仍未纳入更精确的量化口径和 Catapult directives。

## 下一步

1. 把更精确的量化口径和 reference 路径差异分析并入 decode attention kernel 验证。
2. 让更大范围的 kernel-level regression 覆盖多 token、多 head 和非零历史 cache 场景。
3. 再引入更细的 banking、pipeline 和 Catapult directives。