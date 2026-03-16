#pragma once

#include "../common/llm_accel_types.h"
#include "../common/qwen2_model_config.h"
#include "qwen_catapult_fp.h"

namespace llm_accel {

KernelStatus qwen_prefill_attention_kernel(
    const scalar_t* input_sequence,
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const scalar_t* input_layernorm_weight,
    scalar_t rms_eps,
    const packed_w4_t* q_packed_weights,
    const packed_w4_t* k_packed_weights,
    const packed_w4_t* v_packed_weights,
    const packed_w4_t* o_packed_weights,
    const scalar_t* q_bias,
    const scalar_t* k_bias,
    const scalar_t* v_bias,
    const scalar_t* q_scales,
    const scalar_t* k_scales,
    const scalar_t* v_scales,
    const scalar_t* o_scales,
    scalar_t* k_cache,
    scalar_t* v_cache,
    scalar_t* output_sequence);

KernelStatus qwen_prefill_attention_kernel_catapult(
    const prefill_catapult_fp_t input_sequence[kPrefillCatapultSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const prefill_catapult_fp_t input_layernorm_weight[kHiddenSize],
    prefill_catapult_fp_t rms_eps,
    const packed_w4_t q_packed_weights[kHiddenSize * kHiddenSize / 2],
    const packed_w4_t k_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const packed_w4_t v_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const packed_w4_t o_packed_weights[kHiddenSize * kHiddenSize / 2],
    const prefill_catapult_fp_t q_bias[kHiddenSize],
    const prefill_catapult_fp_t k_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t q_scales[kHiddenSize],
    const prefill_catapult_fp_t k_scales[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_scales[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t o_scales[kHiddenSize],
    prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth],
    prefill_catapult_fp_t v_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth],
    prefill_catapult_fp_t output_sequence[kPrefillCatapultSeqCapacity * kHiddenSize]);

void qwen_prefill_attention_qkv_rope_stage_catapult(
    const prefill_catapult_fp_t input_sequence[kPrefillCatapultSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const prefill_catapult_fp_t input_layernorm_weight[kHiddenSize],
    prefill_catapult_fp_t rms_eps,
    const packed_w4_t q_packed_weights[kHiddenSize * kHiddenSize / 2],
    const packed_w4_t k_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const packed_w4_t v_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const prefill_catapult_fp_t q_bias[kHiddenSize],
    const prefill_catapult_fp_t k_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t q_scales[kHiddenSize],
    const prefill_catapult_fp_t k_scales[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_scales[kPrefillCatapultKvWidth],
    prefill_catapult_fp_t q_proj_buffer[kPrefillCatapultSeqCapacity][kHiddenSize],
    prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth],
    prefill_catapult_fp_t v_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth]);

void qwen_prefill_attention_input_norm_stage_catapult(
    const prefill_catapult_fp_t input_sequence[kPrefillCatapultSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const prefill_catapult_fp_t input_layernorm_weight[kHiddenSize],
    prefill_catapult_fp_t rms_eps,
    prefill_catapult_fp_t normalized_sequence[kPrefillCatapultSeqCapacity][kHiddenSize]);

void qwen_prefill_attention_q_projection_stage_catapult(
    const prefill_catapult_fp_t normalized_sequence[kPrefillCatapultSeqCapacity][kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const packed_w4_t q_packed_weights[kHiddenSize * kHiddenSize / 2],
    const prefill_catapult_fp_t q_bias[kHiddenSize],
    const prefill_catapult_fp_t q_scales[kHiddenSize],
    prefill_catapult_fp_t q_proj_buffer[kPrefillCatapultSeqCapacity][kHiddenSize]);

void qwen_prefill_attention_k_projection_stage_catapult(
    const prefill_catapult_fp_t normalized_sequence[kPrefillCatapultSeqCapacity][kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const packed_w4_t k_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const prefill_catapult_fp_t k_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t k_scales[kPrefillCatapultKvWidth],
    prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth]);

void qwen_prefill_attention_v_projection_stage_catapult(
    const prefill_catapult_fp_t normalized_sequence[kPrefillCatapultSeqCapacity][kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const packed_w4_t v_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const prefill_catapult_fp_t v_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_scales[kPrefillCatapultKvWidth],
    prefill_catapult_fp_t v_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth]);

void qwen_prefill_attention_kv_projection_stage_catapult(
    const prefill_catapult_fp_t normalized_sequence[kPrefillCatapultSeqCapacity][kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const packed_w4_t k_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const packed_w4_t v_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const prefill_catapult_fp_t k_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t k_scales[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_scales[kPrefillCatapultKvWidth],
    prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth],
    prefill_catapult_fp_t v_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth]);

void qwen_prefill_attention_q_rope_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    prefill_catapult_fp_t q_proj_buffer[kPrefillCatapultSeqCapacity][kHiddenSize]);

void qwen_prefill_attention_k_rope_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth]);

void qwen_prefill_attention_qkv_projection_stage_catapult(
    const prefill_catapult_fp_t input_sequence[kPrefillCatapultSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const prefill_catapult_fp_t input_layernorm_weight[kHiddenSize],
    prefill_catapult_fp_t rms_eps,
    const packed_w4_t q_packed_weights[kHiddenSize * kHiddenSize / 2],
    const packed_w4_t k_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const packed_w4_t v_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const prefill_catapult_fp_t q_bias[kHiddenSize],
    const prefill_catapult_fp_t k_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t q_scales[kHiddenSize],
    const prefill_catapult_fp_t k_scales[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_scales[kPrefillCatapultKvWidth],
    prefill_catapult_fp_t q_proj_buffer[kPrefillCatapultSeqCapacity][kHiddenSize],
    prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth],
    prefill_catapult_fp_t v_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth]);

void qwen_prefill_attention_rope_apply_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    prefill_catapult_fp_t q_proj_buffer[kPrefillCatapultSeqCapacity][kHiddenSize],
    prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth]);

void qwen_prefill_attention_context_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const prefill_catapult_fp_t q_proj_buffer[kPrefillCatapultSeqCapacity][kHiddenSize],
    const prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth],
    prefill_catapult_fp_t context_buffer[kPrefillCatapultSeqCapacity][kHiddenSize]);

void qwen_prefill_attention_context_output_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const prefill_catapult_fp_t q_proj_buffer[kPrefillCatapultSeqCapacity][kHiddenSize],
    const prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth],
    const packed_w4_t o_packed_weights[kHiddenSize * kHiddenSize / 2],
    const prefill_catapult_fp_t o_scales[kHiddenSize],
    prefill_catapult_fp_t output_sequence[kPrefillCatapultSeqCapacity * kHiddenSize]);

void qwen_prefill_attention_output_projection_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const prefill_catapult_fp_t context_buffer[kPrefillCatapultSeqCapacity][kHiddenSize],
    const packed_w4_t o_packed_weights[kHiddenSize * kHiddenSize / 2],
    const prefill_catapult_fp_t o_scales[kHiddenSize],
    prefill_catapult_fp_t output_sequence[kPrefillCatapultSeqCapacity * kHiddenSize]);

}  // namespace llm_accel