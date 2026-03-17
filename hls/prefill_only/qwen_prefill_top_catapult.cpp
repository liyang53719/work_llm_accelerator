#include "qwen_prefill_top_catapult.h"

#include "qwen_prefill_attention_kernel.h"
#include "qwen_prefill_mlp_kernel.h"

namespace llm_accel {
namespace {

constexpr PrefillAttentionTileConfig kSynthAttentionTileConfig = {
    kDefaultPrefillSeqTile,
    kDefaultPrefillAttentionQueryTile,
    kDefaultPrefillAttentionKeyTile,
    kDefaultPrefillAttentionHiddenProjTile,
    kDefaultPrefillAttentionKvProjTile,
    kDefaultPrefillAttentionHeadDimTile,
    kDefaultPrefillAttentionQueryHeadsParallel,
    kDefaultPrefillAttentionKvHeadsParallel,
};

constexpr PrefillMLPTileConfig kSynthMlpTileConfig = {
    kDefaultPrefillSeqTile,
    kDefaultPrefillMLPHiddenTile,
    kDefaultPrefillMLPFFTile,
};

bool valid_synth_tile_args(
    int attention_seq_tile,
    int attention_query_tile,
    int attention_key_tile,
    int attention_hidden_proj_tile,
    int attention_kv_proj_tile,
    int attention_head_dim_tile,
    int attention_query_heads_parallel,
    int attention_kv_heads_parallel,
    int mlp_seq_tile,
    int mlp_hidden_tile,
    int mlp_ff_tile) {
  return attention_seq_tile == kSynthAttentionTileConfig.seq &&
         attention_query_tile == kSynthAttentionTileConfig.query &&
         attention_key_tile == kSynthAttentionTileConfig.key &&
         attention_hidden_proj_tile == kSynthAttentionTileConfig.hidden_proj &&
         attention_kv_proj_tile == kSynthAttentionTileConfig.kv_proj &&
         attention_head_dim_tile == kSynthAttentionTileConfig.head_dim &&
         attention_query_heads_parallel == kSynthAttentionTileConfig.query_heads_parallel &&
         attention_kv_heads_parallel == kSynthAttentionTileConfig.kv_heads_parallel &&
         mlp_seq_tile == kSynthMlpTileConfig.seq &&
         mlp_hidden_tile == kSynthMlpTileConfig.hidden &&
         mlp_ff_tile == kSynthMlpTileConfig.ff;
}

const packed_w4_t* weight_ptr(const packed_w4_t* base, std::uint64_t byte_offset) {
  return base + byte_offset;
}

const prefill_catapult_fp_t* fp_ptr(const prefill_catapult_fp_t* base, std::uint64_t byte_offset) {
  return base + byte_offset / sizeof(scalar_t);
}

prefill_catapult_fp_t* fp_ptr(prefill_catapult_fp_t* base, std::uint64_t byte_offset) {
  return base + byte_offset / sizeof(scalar_t);
}

}  // namespace

#pragma hls_design top
KernelStatus qwen_prefill_top_catapult(
    int layer_id,
    int seq_len,
    int attention_seq_tile,
    int attention_query_tile,
    int attention_key_tile,
    int attention_hidden_proj_tile,
    int attention_kv_proj_tile,
    int attention_head_dim_tile,
    int attention_query_heads_parallel,
    int attention_kv_heads_parallel,
    int mlp_seq_tile,
    int mlp_hidden_tile,
    int mlp_ff_tile,
    std::uint64_t input_sequence_addr,
    std::uint64_t output_sequence_addr,
    std::uint64_t layer_weights_base_addr,
    std::uint64_t layer_scales_base_addr,
    std::uint64_t k_cache_base_addr,
    std::uint64_t v_cache_base_addr,
    std::uint64_t scratch_base_addr,
    const packed_w4_t weight_ddr[kPrefillWeightWindowBytes],
    const prefill_catapult_fp_t scale_ddr[kPrefillScaleWindowElements],
    prefill_catapult_fp_t kv_cache_ddr[kPrefillKvCacheWindowElements],
    prefill_catapult_fp_t activation_ddr[kPrefillActivationWindowElements],
    packed_w4_t weight_sram[kPrefillWeightSramElements],
    prefill_catapult_fp_t kv_sram[kPrefillKvSramElements],
    acc_t partial_sum_sram[kPrefillPartialSumSramElements],
    prefill_catapult_fp_t softmax_sram[kPrefillSoftmaxSramElements],
    prefill_catapult_fp_t control_sram[kPrefillControlSramElements]) {
  if ((scratch_base_addr % sizeof(scalar_t)) != 0U) {
    return {false, kPrefillInvalidMemoryWindowError};
  }
  if (!valid_layer_id(layer_id) || seq_len <= 0 || seq_len > kPrefillSequenceCapacity) {
    return {false, kPrefillInvalidDescriptorError};
  }
  if (!valid_synth_tile_args(
          attention_seq_tile,
          attention_query_tile,
          attention_key_tile,
          attention_hidden_proj_tile,
          attention_kv_proj_tile,
          attention_head_dim_tile,
          attention_query_heads_parallel,
          attention_kv_heads_parallel,
          mlp_seq_tile,
          mlp_hidden_tile,
          mlp_ff_tile)) {
    return {false, kPrefillInvalidDescriptorError};
  }

  const std::uint64_t scratch_index = scratch_base_addr / sizeof(scalar_t);
  if (scratch_index > static_cast<std::uint64_t>(kPrefillActivationWindowElements) ||
      scratch_index + static_cast<std::uint64_t>(seq_len) * static_cast<std::uint64_t>(kHiddenSize) >
          static_cast<std::uint64_t>(kPrefillActivationWindowElements)) {
    return {false, kPrefillInvalidMemoryWindowError};
  }

  const PrefillTileConfig tile_config{
      kSynthAttentionTileConfig,
      kSynthMlpTileConfig,
  };
  if (!valid_prefill_tile_config(tile_config)) {
    return {false, kPrefillInvalidDescriptorError};
  }

  const LayerParameterLayout layout = default_layer_parameter_layout();
  const prefill_catapult_fp_t* input_sequence = fp_ptr(activation_ddr, input_sequence_addr);
  prefill_catapult_fp_t* output_sequence = fp_ptr(activation_ddr, output_sequence_addr);
  prefill_catapult_fp_t* attention_scratch = activation_ddr + scratch_index;
  prefill_catapult_fp_t* k_cache = fp_ptr(kv_cache_ddr, k_cache_base_addr);
  prefill_catapult_fp_t* v_cache = fp_ptr(kv_cache_ddr, v_cache_base_addr);
  const prefill_catapult_fp_t* input_layernorm_weight =
      fp_ptr(scale_ddr, layer_scales_base_addr + layout.input_layernorm_weight_offset_bytes);
  const prefill_catapult_fp_t* post_attention_layernorm_weight =
      fp_ptr(scale_ddr, layer_scales_base_addr + layout.post_attention_layernorm_weight_offset_bytes);

  const packed_w4_t* q_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.q_weight_offset_bytes);
  const packed_w4_t* k_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.k_weight_offset_bytes);
  const packed_w4_t* v_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.v_weight_offset_bytes);
  const packed_w4_t* o_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.o_weight_offset_bytes);
  const packed_w4_t* gate_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.gate_weight_offset_bytes);
  const packed_w4_t* up_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.up_weight_offset_bytes);
  const packed_w4_t* down_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.down_weight_offset_bytes);

  const prefill_catapult_fp_t* q_bias = fp_ptr(scale_ddr, layer_scales_base_addr + layout.q_bias_offset_bytes);
  const prefill_catapult_fp_t* k_bias = fp_ptr(scale_ddr, layer_scales_base_addr + layout.k_bias_offset_bytes);
  const prefill_catapult_fp_t* v_bias = fp_ptr(scale_ddr, layer_scales_base_addr + layout.v_bias_offset_bytes);
  const prefill_catapult_fp_t* q_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.q_scale_offset_bytes);
  const prefill_catapult_fp_t* k_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.k_scale_offset_bytes);
  const prefill_catapult_fp_t* v_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.v_scale_offset_bytes);
  const prefill_catapult_fp_t* o_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.o_scale_offset_bytes);
  const prefill_catapult_fp_t* gate_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.gate_scale_offset_bytes);
  const prefill_catapult_fp_t* up_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.up_scale_offset_bytes);
  const prefill_catapult_fp_t* down_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.down_scale_offset_bytes);

  const KernelStatus attention_status = qwen_prefill_attention_kernel_catapult(
      input_sequence,
      seq_len,
      tile_config.attention,
      input_layernorm_weight,
      prefill_catapult_fp_t(kRmsNormEps),
      q_weights,
      k_weights,
      v_weights,
      o_weights,
      q_bias,
      k_bias,
      v_bias,
      q_scales,
      k_scales,
      v_scales,
      o_scales,
      k_cache,
      v_cache,
      attention_scratch);
  if (!attention_status.ok) {
    return attention_status;
  }

  return qwen_prefill_mlp_kernel_catapult(
      attention_scratch,
      seq_len,
      tile_config.mlp,
      post_attention_layernorm_weight,
      prefill_catapult_fp_t(kRmsNormEps),
      gate_weights,
      up_weights,
      down_weights,
      gate_scales,
      up_scales,
      down_scales,
      output_sequence);
}

#pragma hls_design ccore
#pragma hls_ccore_type sequential
KernelStatus qwen_prefill_top_catapult_fine(
    int layer_id,
    int seq_len,
    int attention_seq_tile,
    int attention_query_tile,
    int attention_key_tile,
    int attention_hidden_proj_tile,
    int attention_kv_proj_tile,
    int attention_head_dim_tile,
    int attention_query_heads_parallel,
    int attention_kv_heads_parallel,
    int mlp_seq_tile,
    int mlp_hidden_tile,
    int mlp_ff_tile,
    std::uint64_t input_sequence_addr,
    std::uint64_t output_sequence_addr,
    std::uint64_t layer_weights_base_addr,
    std::uint64_t layer_scales_base_addr,
    std::uint64_t k_cache_base_addr,
    std::uint64_t v_cache_base_addr,
    std::uint64_t scratch_base_addr,
    const packed_w4_t weight_ddr[kPrefillWeightWindowBytes],
    const prefill_catapult_fp_t scale_ddr[kPrefillScaleWindowElements],
    prefill_catapult_fp_t kv_cache_ddr[kPrefillKvCacheWindowElements],
    prefill_catapult_fp_t activation_ddr[kPrefillActivationWindowElements],
    packed_w4_t weight_sram[kPrefillWeightSramElements],
    prefill_catapult_fp_t kv_sram[kPrefillKvSramElements],
    acc_t partial_sum_sram[kPrefillPartialSumSramElements],
    prefill_catapult_fp_t softmax_sram[kPrefillSoftmaxSramElements],
    prefill_catapult_fp_t control_sram[kPrefillControlSramElements]) {
  if ((scratch_base_addr % sizeof(scalar_t)) != 0U) {
    return {false, kPrefillInvalidMemoryWindowError};
  }
  if (!valid_layer_id(layer_id) || seq_len <= 0 || seq_len > kPrefillSequenceCapacity) {
    return {false, kPrefillInvalidDescriptorError};
  }
  if (!valid_synth_tile_args(
          attention_seq_tile,
          attention_query_tile,
          attention_key_tile,
          attention_hidden_proj_tile,
          attention_kv_proj_tile,
          attention_head_dim_tile,
          attention_query_heads_parallel,
          attention_kv_heads_parallel,
          mlp_seq_tile,
          mlp_hidden_tile,
          mlp_ff_tile)) {
    return {false, kPrefillInvalidDescriptorError};
  }

  const std::uint64_t scratch_index = scratch_base_addr / sizeof(scalar_t);
  if (scratch_index > static_cast<std::uint64_t>(kPrefillActivationWindowElements) ||
      scratch_index + static_cast<std::uint64_t>(seq_len) * static_cast<std::uint64_t>(kHiddenSize) >
          static_cast<std::uint64_t>(kPrefillActivationWindowElements)) {
    return {false, kPrefillInvalidMemoryWindowError};
  }

  const PrefillTileConfig tile_config{
      kSynthAttentionTileConfig,
      kSynthMlpTileConfig,
  };
  if (!valid_prefill_tile_config(tile_config)) {
    return {false, kPrefillInvalidDescriptorError};
  }

  const LayerParameterLayout layout = default_layer_parameter_layout();
  const prefill_catapult_fp_t* input_sequence = fp_ptr(activation_ddr, input_sequence_addr);
  prefill_catapult_fp_t* output_sequence = fp_ptr(activation_ddr, output_sequence_addr);
  prefill_catapult_fp_t* attention_scratch = activation_ddr + scratch_index;
  prefill_catapult_fp_t* k_cache = fp_ptr(kv_cache_ddr, k_cache_base_addr);
  prefill_catapult_fp_t* v_cache = fp_ptr(kv_cache_ddr, v_cache_base_addr);
  const prefill_catapult_fp_t* input_layernorm_weight =
      fp_ptr(scale_ddr, layer_scales_base_addr + layout.input_layernorm_weight_offset_bytes);
  const prefill_catapult_fp_t* post_attention_layernorm_weight =
      fp_ptr(scale_ddr, layer_scales_base_addr + layout.post_attention_layernorm_weight_offset_bytes);

  const packed_w4_t* q_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.q_weight_offset_bytes);
  const packed_w4_t* k_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.k_weight_offset_bytes);
  const packed_w4_t* v_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.v_weight_offset_bytes);
  const packed_w4_t* o_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.o_weight_offset_bytes);
  const packed_w4_t* gate_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.gate_weight_offset_bytes);
  const packed_w4_t* up_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.up_weight_offset_bytes);
  const packed_w4_t* down_weights = weight_ptr(weight_ddr, layer_weights_base_addr + layout.down_weight_offset_bytes);

  const prefill_catapult_fp_t* q_bias = fp_ptr(scale_ddr, layer_scales_base_addr + layout.q_bias_offset_bytes);
  const prefill_catapult_fp_t* k_bias = fp_ptr(scale_ddr, layer_scales_base_addr + layout.k_bias_offset_bytes);
  const prefill_catapult_fp_t* v_bias = fp_ptr(scale_ddr, layer_scales_base_addr + layout.v_bias_offset_bytes);
  const prefill_catapult_fp_t* q_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.q_scale_offset_bytes);
  const prefill_catapult_fp_t* k_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.k_scale_offset_bytes);
  const prefill_catapult_fp_t* v_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.v_scale_offset_bytes);
  const prefill_catapult_fp_t* o_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.o_scale_offset_bytes);
  const prefill_catapult_fp_t* gate_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.gate_scale_offset_bytes);
  const prefill_catapult_fp_t* up_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.up_scale_offset_bytes);
  const prefill_catapult_fp_t* down_scales = fp_ptr(scale_ddr, layer_scales_base_addr + layout.down_scale_offset_bytes);

    qwen_prefill_attention_kv_cache_stage_catapult(
      input_sequence,
      seq_len,
      tile_config.attention,
      input_layernorm_weight,
      prefill_catapult_fp_t(kRmsNormEps),
      k_weights,
      v_weights,
      k_bias,
      v_bias,
      k_scales,
      v_scales,
      k_cache,
      v_cache);
    qwen_prefill_attention_q_context_output_stage_catapult(
      input_sequence,
      seq_len,
      tile_config.attention,
      input_layernorm_weight,
      prefill_catapult_fp_t(kRmsNormEps),
      q_weights,
      q_bias,
      q_scales,
      k_cache,
      v_cache,
      o_weights,
      o_scales,
      attention_scratch);

  (void)weight_sram;
  (void)kv_sram;
  (void)partial_sum_sram;
  (void)softmax_sram;
  (void)control_sram;

  return qwen_prefill_mlp_kernel_catapult(
      attention_scratch,
      seq_len,
      tile_config.mlp,
      post_attention_layernorm_weight,
      prefill_catapult_fp_t(kRmsNormEps),
      gate_weights,
      up_weights,
      down_weights,
      gate_scales,
      up_scales,
      down_scales,
      output_sequence);
}

}  // namespace llm_accel