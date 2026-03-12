#pragma once

#include "../common/llm_accel_types.h"
#include "../common/qwen2_model_config.h"

namespace llm_accel {

KernelStatus qwen_decode_attention_kernel(
    const scalar_t* input_token,
    int past_seq_len,
    const packed_w4_t* q_packed_weights,
    const packed_w4_t* k_packed_weights,
    const packed_w4_t* v_packed_weights,
    const packed_w4_t* o_packed_weights,
    const scalar_t* q_scales,
    const scalar_t* k_scales,
    const scalar_t* v_scales,
    const scalar_t* o_scales,
    scalar_t* k_cache,
    scalar_t* v_cache,
    scalar_t* output_token);

}  // namespace llm_accel