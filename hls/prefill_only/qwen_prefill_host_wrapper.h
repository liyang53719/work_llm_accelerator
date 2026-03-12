#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int qwen_prefill_stub_forward(
    const float* input_sequence,
    int seq_len,
    int tile_m,
    float* output_sequence);

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