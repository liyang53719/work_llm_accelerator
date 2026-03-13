# Prefill Tile 切分约束表

## 1. 固定模型维度

| 项目 | 数值 | 说明 |
| --- | ---: | --- |
| hidden_size | 1536 | 输入隐藏维、attention 输出维、MLP 输出维 |
| intermediate_size | 8960 | MLP gate/up 输出维、down 输入维 |
| num_hidden_layers | 28 | 当前表只针对 layer 维度无关的单层切分约束 |
| num_attention_heads | 12 | Q / O 路径按 12 个 query heads 展开 |
| num_key_value_heads | 2 | K / V 路径按 2 个 KV heads 展开 |
| num_groups | 6 | 每个 KV head 对应 6 个 query heads |
| head_dim | 128 | `1536 / 12 = 128` |
| kv_width | 256 | `2 * 128 = 256` |
| max_position_embeddings | 32768 | 模型支持的最大位置长度 |

## 2. 当前已覆盖的序列长度

| 类别 | seq_len |
| --- | --- |
| 真实 case | 11, 65 |
| 随机 case | 8, 16, 32, 64, 128, 256, 512, 1024 |

结论：

- `8..1024` 都是模型允许的合法输入长度，但不是模型最大长度覆盖。
- 对 tile 设计而言，不能只支持 2 的幂长度，因为真实 case 已经包含 `11` 和 `65`。
- 因此无论选择什么 tile，硬件侧都需要支持尾块处理，或者明确采用 padding 策略。

## 3. Prefill Attention 维度约束

| 模块 | 张量 / 计算 | 形状 | 硬约束 | 推荐 tile 方向 |
| --- | --- | --- | --- | --- |
| Input RMSNorm | `X -> X_ln` | `[S, 1536]` | hidden 轴若不做尾块，tile 必须整除 1536 | `T_h` 取 128, 256, 384, 512, 768 |
| Q proj | `[S, 1536] x [1536, 1536]` | 输出 `[S, 1536]` | 输入 / 输出 hidden 轴都受 1536 约束 | `T_s x T_h x T_h` |
| K proj | `[S, 1536] x [1536, 256]` | 输出 `[S, 256]` | 输出轴若不做尾块，tile 必须整除 256 | `T_s x T_h x T_kv` |
| V proj | `[S, 1536] x [1536, 256]` | 输出 `[S, 256]` | 同上 | `T_s x T_h x T_kv` |
| RoPE(Q) | `[S, 12, 128]` | `[S, 12, 128]` | head_dim 最自然粒度是 128；若切更细，必须保持偶数 pair | `T_sq x T_hq x T_hd` |
| RoPE(K) | `[S, 2, 128]` | `[S, 2, 128]` | KV head 轴只允许 1 或 2 的整分块 | `T_sk x T_hkv x T_hd` |
| Score | `Q @ K^T` | 每 head 为 `[S, S]` | 因果 mask 要求 `T_q`、`T_k` 支持上三角裁剪 | `T_q x T_k` |
| Softmax/Context | `P @ V` | 每 head 为 `[S, 128]` | 若不物化全分数矩阵，可用 streaming softmax 降 SRAM | `T_q x T_k x T_hd` |
| O proj | `[S, 1536] x [1536, 1536]` | 输出 `[S, 1536]` | 同 Q proj | `T_s x T_h x T_h` |

### Attention 切分硬规则

| 维度 | 约束 |
| --- | --- |
| `T_h` | 若希望无尾块，必须整除 1536 |
| `T_kv` | 若希望无尾块，必须整除 256 |
| `T_hd` | 若希望无尾块，必须整除 128；RoPE 还要求 2-way pair 对齐 |
| `T_hq` | 建议整除 12；若按 GQA 共享 K/V，最好按 6 或 12 分组 |
| `T_hkv` | 建议取 1 或 2 |
| `T_q, T_k` | 必须支持 `11, 65` 这样的非整倍数序列尾块 |

### Attention 推荐候选

| 轴 | 推荐候选 | 原因 |
| --- | --- | --- |
| `T_h` | 128, 256, 384, 512, 768 | 都整除 1536，且与 `head_dim=128` 对齐较好 |
| `T_kv` | 128, 256 | 都整除 256，避免 KV 输出尾块 |
| `T_hd` | 64, 128 | 64 便于更细粒度并行，128 可避免 head_dim 方向分块 |
| `T_hq` | 6, 12 | 直接匹配 GQA 的 `12/2 = 6` 组关系 |
| `T_q/T_k` | 32, 64, 128, 256 | 对当前测试长度集比较自然，且便于功耗 / SRAM 折中 |

### Attention 额外注意点

- 不建议物化重复后的 `K/V` 为 `[S, 12, 128]` 再存储，最好保持原始 `2` 个 KV heads，按 group 广播使用。
- 当 `seq_len` 提升到 `256/512/1024` 后，分数矩阵压力明显上升，`T_q x T_k` 将是 prefill SRAM 的主导项。
- 若 score 保持 FP32 累加，单块 score buffer 约为 `T_q * T_k * 4` 字节。

## 4. Prefill MLP 维度约束

| 模块 | 张量 / 计算 | 形状 | 硬约束 | 推荐 tile 方向 |
| --- | --- | --- | --- | --- |
| Post-Attn RMSNorm | `X_res -> X_ln` | `[S, 1536]` | `T_h` 若无尾块需整除 1536 | `T_s x T_h` |
| Gate proj | `[S, 1536] x [1536, 8960]` | 输出 `[S, 8960]` | 中间轴若不做尾块，tile 必须整除 8960 | `T_s x T_h x T_ff` |
| Up proj | `[S, 1536] x [1536, 8960]` | 输出 `[S, 8960]` | 同上 | `T_s x T_h x T_ff` |
| SiLU + Mul | `[S, 8960]` | `[S, 8960]` | Gate / Up 两路 tile 必须同形 | `T_s x T_ff` |
| Down proj | `[S, 8960] x [8960, 1536]` | 输出 `[S, 1536]` | 输入 tile 与 `T_ff` 对齐；输出 tile 与 `T_h` 对齐 | `T_s x T_ff x T_h` |

### MLP 切分硬规则

| 维度 | 约束 |
| --- | --- |
| `T_h` | 若希望无尾块，必须整除 1536 |
| `T_ff` | 若希望无尾块，必须整除 8960 |
| `T_s` | 仍需支持 `11, 65` 的尾块 |
| Gate/Up | 两路必须共享同一个 `T_ff`，否则逐点乘阶段会增加重排复杂度 |
| Down | 输入 tile 必须与 Gate/Up 输出 tile 完全匹配 |

### MLP 推荐候选

| 轴 | 推荐候选 | 原因 |
| --- | --- | --- |
| `T_h` | 128, 256, 384, 512, 768 | 与 hidden_size=1536 对齐 |
| `T_ff` | 128, 256, 640, 1280 | 都整除 8960；同时对 int4 打包和 DDR burst 更友好 |
| `T_s` | 32, 64, 128 | 与当前已测试长度兼容，SRAM 压力可控 |

### MLP 不推荐候选

| 候选 | 问题 |
| --- | --- |
| `T_ff = 512` | 不能整除 8960，会引入固定尾块 |
| `T_ff = 1024` | 不能整除 8960，会引入固定尾块 |
| `T_ff = 8960` | 虽然无尾块，但对片上缓存压力过大，通常不现实 |

## 5. 面向 RTL 的约束总结

| 类别 | 结论 |
| --- | --- |
| 长度兼容性 | 必须支持 `11/65` 这类非整倍数长度，不能只针对 2 的幂长度设计 |
| Attention 优先约束 | `head_dim=128`、`num_heads=12`、`num_kv_heads=2`、`num_groups=6` 是最硬的结构约束 |
| MLP 优先约束 | `intermediate_size=8960` 决定 `T_ff` 不能随便取 512/1024 这类常见值 |
| SRAM 主压力 | prefill attention 的 score/context 路径会随 `seq_len` 增长最快 |
| 推荐起步方案 | Attention 先看 `T_q/T_k=64 or 128`，`T_h=128 or 256`，`T_kv=128 or 256`；MLP 先看 `T_ff=256 or 640` |
| 设计原则 | 先优先选择“整除主维度 + 仅在 seq_len 上留尾块”的方案，避免同时在 hidden / ffn / seq 三个轴上都留尾块 |

## 6. 与当前误差包络的关系

- 当前扩展统计已经覆盖 `8,16,32,64,128,256,512,1024`。
- 从误差包络看，`256` 以后 attention-only 与 full-quant 的最坏值明显抬升，因此 tile 讨论可以开始，但不能把当前结果当成最终 signoff。
- 更稳妥的顺序是：先按本表约束筛掉明显不合适的 tile 候选，再在这些候选上继续做数值和带宽评估。