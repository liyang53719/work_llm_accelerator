# qwen_prefill_attention_context_stage_catapult.cpp 函数接口清单（2026-03-18，已更新）

目标文件：`hls/prefill_only/qwen_prefill_attention_context_stage_catapult.cpp`

统计口径：按当前文件做保守的函数签名抽取后分四类：

- `channel-only`：签名包含 `ac_channel`，且不再暴露裸指针 `*` 或数组 `[...]`。
- `mixed`：签名同时包含 `ac_channel` 与裸指针/数组。
- `pointer-array`：签名不含 `ac_channel`，但仍含裸指针/数组。
- `scalar-ref-only`：纯标量/引用，不属于本轮 channel 边界改造对象。

## 1. 最新统计

| 分类 | 数量 | 状态 |
| --- | ---: | --- |
| channel-only | 40 | 已满足本轮边界目标 |
| mixed | 0 | 已清零 |
| pointer-array | 0 | 已清零 |
| scalar-ref-only | 58 | 不纳入本轮 channel 边界改造 |
| 总计 | 98 | 保守抽取结果 |

标记规则：

- `[x]`：当前已经满足目标。
- `[ ]`：当前仍未满足目标。
- `[-]`：不纳入本轮 channel 边界改造。

本轮新增变化：

1. `process_context_score_key`、`process_context_max_score_key`、`load_context_kv_token_packet` 三个无调用旧函数已删除。
2. context score/value 主链路与 hidden-proj 内所有 `mixed` helper 已全部收敛为 `packet/reference` 接口，`mixed` 已清零。
3. `stream_context_query_tasks_from_sequence`、`stream_context_query_tasks_from_tile`、`store_context_result_packets` 已删除并内联到外层 pointer-array wrapper。
4. context 内部的 query/result load-store helper、KV token loader、FP word pack/unpack helper 已继续删除并内联到唯一调用点。
5. hidden-proj 内一批纯数组搬运 helper 与纯转发 wrapper 已继续删除并内联到调用点。
6. catapult 专用实现文件里无活跃调用链的 stage 入口已继续删除，包括 `q/v/k projection`、`kv/qkv projection`、`q/k rope`、`rope apply`、`qkv rope`、`context_output`、`output_projection`、`context_stage` 等纯导出层定义。
7. `project_hidden_token_tilewise_fp` 已改为 template/reference 形式，不再暴露裸指针/数组签名。
8. `qwen_prefill_attention_kv_cache_stage_catapult`、`qwen_prefill_attention_q_context_output_stage_catapult`、`qwen_prefill_attention_kernel_catapult` 已改成 `CatapultTensorView` / `CatapultConstTensorView` 轻量 view 接口，并同步更新调用点。

## 2. 已满足目标：channel-only（40）

- [x] `stream_context_v_token_packet_words`
- [x] `stream_context_k_token_packet_words`
- [x] `read_context_k_token_packet_words`
- [x] `read_context_v_token_packet_words`
- [x] `stream_context_value_key_packet_words`
- [x] `accumulate_context_value_key_packet_words`
- [x] `stream_context_value_key_tasks`
- [x] `accumulate_context_value_key_tasks`
- [x] `process_context_max_score_tile`
- [x] `process_context_value_tile`
- [x] `compute_context_max_score_tile_tasks`
- [x] `compute_context_value_tile_tasks`
- [x] `stream_context_key_tile_meta_packets`
- [x] `stream_context_query_packet_words`
- [x] `read_context_query_packet_words`
- [x] `stream_context_result_packet_words`
- [x] `read_context_result_packet_words`
- [x] `stream_context_score_packet_words`
- [x] `read_context_score_packet_words`
- [x] `compute_context_max_score_head_state_packet`
- [x] `compute_context_value_head_state_packet`
- [x] `prefill_attention_context_max_score_head_group_stage_fp`
- [x] `prefill_attention_context_query_max_score_fp`
- [x] `prefill_attention_context_value_head_group_stage_fp`
- [x] `prefill_attention_context_query_value_fp`
- [x] `stream_context_query_packet_word_channel`
- [x] `stream_context_score_word_channel`
- [x] `stream_context_score_stage_key_words_for_query`
- [x] `stream_context_value_stage_kv_words_for_query`
- [x] `stream_context_score_stage_inputs`
- [x] `stream_context_value_stage_inputs`
- [x] `prefill_attention_context_score_stream_stage_fp`
- [x] `prefill_attention_context_value_stream_stage_fp`
- [x] `qwen_prefill_attention_context_query_tile_stream_catapult`
- [x] `read_hidden_proj_fp_tile_packet`
- [x] `read_hidden_proj_fp_tile_words`
- [x] `write_hidden_proj_fp_tile_packet`
- [x] `write_hidden_proj_fp_tile_words`
- [x] `read_hidden_proj_packed_weight_tile_packet`
- [x] `qwen_prefill_attention_q_context_output_tile_stream_catapult`

## 3. 仍需改造：mixed（0）

`mixed` 组已清零。

## 4. 仍需改造：pointer-array（0）

`pointer-array` 组已清零。

## 5. 不纳入本轮 channel 边界改造：scalar-ref-only（58）

- [-] `min_int`
- [-] `max_int`
- [-] `approx_sqrt`
- [-] `wrap_angle`
- [-] `approx_exp`
- [-] `decode_int4_weight`
- [-] `fp_const`
- [-] `fp_const_int`
- [-] `fp_zero`
- [-] `fp_one`
- [-] `fp_add_op`
- [-] `fp_sub_op`
- [-] `fp_mul_op`
- [-] `fp_div_op`
- [-] `fp_mac_op`
- [-] `fp_sqrt_op`
- [-] `fp_eq_op`
- [-] `fp_lt_op`
- [-] `fp_gt_op`
- [-] `fp_le_op`
- [-] `approx_exp_fp`
- [-] `approx_rsqrt_fp`
- [-] `approx_reciprocal_fp`
- [-] `wrap_angle_fp`
- [-] `decode_int4_weight_fp`
- [-] `compute_context_score_packet`
- [-] `compute_context_score_packet`（K packet 重载）
- [-] `update_context_max_score_packet`
- [-] `accumulate_context_weighted_value_packet`
- [-] `count_context_key_tiles`
- [-] `init_context_max_score_packet`
- [-] `init_context_value_head_state_packet`
- [-] `init_context_query_meta_packet`
- [-] `init_context_result_meta_packet`
- [-] `init_context_key_tile_meta_packet`
- [-] `store_context_head_state_packet`

## 6. 当前结论

1. 现在这份表对应当前代码真实状态：`mixed` 与 `pointer-array` 都已清零。
2. 本轮 `channel-only` 保持 40，不再新增裸指针/数组边界。
3. 这轮保守统计下，`pointer-array` 已从 69 逐步压到 0。
4. 当前剩余内容只属于 `scalar-ref-only` 或 `channel-only`，已经脱离本轮边界清理目标。
5. 后续如果继续做 Catapult 建模优化，应转向 top wrapper 建模、operator binding 或性能/资源方向，而不是继续在这份文件里找裸数组边界。