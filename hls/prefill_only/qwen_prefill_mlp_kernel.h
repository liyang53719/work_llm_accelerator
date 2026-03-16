#pragma once

#include "../common/llm_accel_types.h"
#include "../common/qwen2_model_config.h"
#include "qwen_catapult_fp.h"

namespace llm_accel {

KernelStatus qwen_prefill_mlp_kernel(
    const scalar_t* attention_residual,
    int seq_len,
    const PrefillMLPTileConfig& tile_config,
    const scalar_t* post_attention_layernorm_weight,
    scalar_t rms_eps,
    const packed_w4_t* gate_packed_weights,
    const packed_w4_t* up_packed_weights,
    const packed_w4_t* down_packed_weights,
    const scalar_t* gate_scales,
    const scalar_t* up_scales,
    const scalar_t* down_scales,
    scalar_t* output_sequence);

KernelStatus qwen_prefill_mlp_kernel_catapult(
    const prefill_catapult_fp_t attention_residual[kPrefillCatapultSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillMLPTileConfig& tile_config,
    const prefill_catapult_fp_t post_attention_layernorm_weight[kHiddenSize],
    prefill_catapult_fp_t rms_eps,
    const packed_w4_t gate_packed_weights[kIntermediateSize * kHiddenSize / 2],
    const packed_w4_t up_packed_weights[kIntermediateSize * kHiddenSize / 2],
    const packed_w4_t down_packed_weights[kIntermediateSize * kHiddenSize / 2],
    const prefill_catapult_fp_t gate_scales[kIntermediateSize],
    const prefill_catapult_fp_t up_scales[kIntermediateSize],
    const prefill_catapult_fp_t down_scales[kHiddenSize],
    prefill_catapult_fp_t output_sequence[kPrefillCatapultSeqCapacity * kHiddenSize]);

}  // namespace llm_accel