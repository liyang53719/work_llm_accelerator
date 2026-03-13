#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

int qwen_prefill_stub_forward(
    const float* input_sequence,
    int seq_len,
    int tile_m,
    float* output_sequence);

int qwen_prefill_attention_smoke_forward(
    const float* input_sequence,
    int seq_len,
    int tile_m,
    const float* input_layernorm_weight,
    const std::uint8_t* q_packed_weights,
    const std::uint8_t* k_packed_weights,
    const std::uint8_t* v_packed_weights,
    const std::uint8_t* o_packed_weights,
    const float* q_bias,
    const float* k_bias,
    const float* v_bias,
    const float* q_scales,
    const float* k_scales,
    const float* v_scales,
    const float* o_scales,
    float* k_cache,
    float* v_cache,
    float* output_sequence);

int qwen_prefill_mlp_smoke_forward(
    const float* attention_residual_sequence,
    int seq_len,
    int tile_m,
    const float* post_attention_layernorm_weight,
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* down_packed_weights,
    const float* gate_scales,
    const float* up_scales,
    const float* down_scales,
    float* output_sequence);

int qwen_prefill_layer_smoke_forward(
    const float* input_sequence,
    int seq_len,
    int tile_m,
    const float* input_layernorm_weight,
    const float* post_attention_layernorm_weight,
    const std::uint8_t* q_packed_weights,
    const std::uint8_t* k_packed_weights,
    const std::uint8_t* v_packed_weights,
    const std::uint8_t* o_packed_weights,
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* down_packed_weights,
    const float* q_bias,
    const float* k_bias,
    const float* v_bias,
    const float* q_scales,
    const float* k_scales,
    const float* v_scales,
    const float* o_scales,
    const float* gate_scales,
    const float* up_scales,
    const float* down_scales,
    float* k_cache,
    float* v_cache,
    float* output_sequence);

int qwen_prefill_top_smoke_forward(
    int layer_id,
    int seq_len,
    int tile_m,
    std::uint64_t input_sequence_addr,
    std::uint64_t output_sequence_addr,
    std::uint64_t layer_weights_base_addr,
    std::uint64_t layer_scales_base_addr,
    std::uint64_t k_cache_base_addr,
    std::uint64_t v_cache_base_addr,
    const std::uint8_t* weight_ddr,
    const float* scale_ddr,
    float* kv_cache_ddr,
    float* activation_ddr,
    std::uint8_t* weight_sram,
    float* kv_sram,
    std::int32_t* partial_sum_sram,
    float* softmax_sram,
    float* control_sram);

int qwen_prefill_layer0_reference_forward(
    const float* input_sequence,
    int seq_len,
    const float* input_layernorm_weight,
    const float* q_weight,
    const float* q_bias,
    const float* k_weight,
    const float* k_bias,
    const float* v_weight,
    const float* v_bias,
    const float* o_weight,
    const float* o_bias,
    const float* post_attention_layernorm_weight,
    const float* gate_weight,
    const float* gate_bias,
    const float* up_weight,
    const float* up_bias,
    const float* down_weight,
    const float* down_bias,
    float rms_eps,
    float* output_sequence);

int qwen_prefill_layer0_reference_forward_with_cache(
    const float* input_sequence,
    int seq_len,
    const float* input_layernorm_weight,
    const float* q_weight,
    const float* q_bias,
    const float* k_weight,
    const float* k_bias,
    const float* v_weight,
    const float* v_bias,
    const float* o_weight,
    const float* o_bias,
    const float* post_attention_layernorm_weight,
    const float* gate_weight,
    const float* gate_bias,
    const float* up_weight,
    const float* up_bias,
    const float* down_weight,
    const float* down_bias,
    float rms_eps,
    float* output_sequence,
    float* k_cache,
    float* v_cache);

#ifdef __cplusplus
}
#endif