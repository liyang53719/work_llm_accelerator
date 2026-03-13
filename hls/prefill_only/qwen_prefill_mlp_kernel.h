#pragma once

#include "../common/llm_accel_types.h"
#include "../common/qwen2_model_config.h"

namespace llm_accel {

KernelStatus qwen_prefill_mlp_kernel(
    const scalar_t* attention_residual,
    int seq_len,
    int tile_m,
    const scalar_t* post_attention_layernorm_weight,
    scalar_t rms_eps,
    const packed_w4_t* gate_packed_weights,
    const packed_w4_t* up_packed_weights,
    const packed_w4_t* down_packed_weights,
    const scalar_t* gate_scales,
    const scalar_t* up_scales,
    const scalar_t* down_scales,
    scalar_t* output_sequence);

}  // namespace llm_accel