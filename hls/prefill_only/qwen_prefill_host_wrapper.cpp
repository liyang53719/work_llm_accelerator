#include "qwen_prefill_host_wrapper.h"

#include <cstdint>
#include <vector>

#include "qwen_prefill_attention_kernel.h"
#include "qwen_prefill_mlp_kernel.h"
#include "qwen_prefill_top_wrapper.h"

using namespace llm_accel;

namespace {

PrefillAttentionTileConfig make_attention_tile_config(
    int seq_tile,
    int query_tile,
    int key_tile,
    int hidden_proj_tile,
    int kv_proj_tile,
    int head_dim_tile,
    int query_heads_parallel,
    int kv_heads_parallel) {
  return {
      seq_tile,
      query_tile,
      key_tile,
      hidden_proj_tile,
      kv_proj_tile,
      head_dim_tile,
      query_heads_parallel,
      kv_heads_parallel,
  };
}

PrefillMLPTileConfig make_mlp_tile_config(int seq_tile, int hidden_tile, int ff_tile) {
  return {
      seq_tile,
      hidden_tile,
      ff_tile,
  };
}

PrefillTileConfig make_prefill_tile_config(
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
  return {
      make_attention_tile_config(
          attention_seq_tile,
          attention_query_tile,
          attention_key_tile,
          attention_hidden_proj_tile,
          attention_kv_proj_tile,
          attention_head_dim_tile,
          attention_query_heads_parallel,
          attention_kv_heads_parallel),
      make_mlp_tile_config(mlp_seq_tile, mlp_hidden_tile, mlp_ff_tile),
  };
}

}  // namespace

extern "C" int qwen_prefill_stub_forward(
    const float* input_sequence,
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
    float* output_sequence) {
  if (input_sequence == nullptr || output_sequence == nullptr || seq_len <= 0) {
    return 1;
  }

  const PrefillTileConfig tile_config = make_prefill_tile_config(
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
      mlp_ff_tile);

  std::vector<float> attention_output(seq_len * kHiddenSize, 0.0f);
  std::vector<float> k_cache(seq_len * kNumKeyValueHeads * kHeadDim, 0.0f);
  std::vector<float> v_cache(seq_len * kNumKeyValueHeads * kHeadDim, 0.0f);
    std::vector<float> input_layernorm_weight(kHiddenSize, 1.0f);
    std::vector<float> post_attention_layernorm_weight(kHiddenSize, 1.0f);
    std::vector<packed_w4_t> q_weights(static_cast<std::size_t>(kHiddenSize) * kHiddenSize / 2, 0);
    std::vector<packed_w4_t> k_weights(static_cast<std::size_t>(kHiddenSize) * kNumKeyValueHeads * kHeadDim / 2, 0);
    std::vector<packed_w4_t> v_weights(static_cast<std::size_t>(kHiddenSize) * kNumKeyValueHeads * kHeadDim / 2, 0);
    std::vector<packed_w4_t> o_weights(static_cast<std::size_t>(kHiddenSize) * kHiddenSize / 2, 0);
    std::vector<packed_w4_t> gate_weights(static_cast<std::size_t>(kIntermediateSize) * kHiddenSize / 2, 0);
    std::vector<packed_w4_t> up_weights(static_cast<std::size_t>(kIntermediateSize) * kHiddenSize / 2, 0);
    std::vector<packed_w4_t> down_weights(static_cast<std::size_t>(kIntermediateSize) * kHiddenSize / 2, 0);
    std::vector<float> q_bias(kHiddenSize, 0.0f);
    std::vector<float> k_bias(kNumKeyValueHeads * kHeadDim, 0.0f);
    std::vector<float> v_bias(kNumKeyValueHeads * kHeadDim, 0.0f);
    std::vector<float> q_scales(kHiddenSize, 1.0f);
    std::vector<float> k_scales(kNumKeyValueHeads * kHeadDim, 1.0f);
    std::vector<float> v_scales(kNumKeyValueHeads * kHeadDim, 1.0f);
    std::vector<float> o_scales(kHiddenSize, 1.0f);
    std::vector<float> gate_scales(kIntermediateSize, 1.0f);
    std::vector<float> up_scales(kIntermediateSize, 1.0f);
    std::vector<float> down_scales(kHiddenSize, 1.0f);

  KernelStatus attention_status = qwen_prefill_attention_kernel(
      input_sequence,
      seq_len,
      tile_config.attention,
      input_layernorm_weight.data(),
      kRmsNormEps,
      q_weights.data(),
      k_weights.data(),
      v_weights.data(),
      o_weights.data(),
      q_bias.data(),
      k_bias.data(),
      v_bias.data(),
      q_scales.data(),
      k_scales.data(),
      v_scales.data(),
      o_scales.data(),
      k_cache.data(),
      v_cache.data(),
      attention_output.data());
  if (!attention_status.ok) {
    return attention_status.error_code;
  }

  KernelStatus mlp_status = qwen_prefill_mlp_kernel(
      attention_output.data(),
      seq_len,
      tile_config.mlp,
      post_attention_layernorm_weight.data(),
      kRmsNormEps,
      gate_weights.data(),
      up_weights.data(),
      down_weights.data(),
      gate_scales.data(),
      up_scales.data(),
      down_scales.data(),
      output_sequence);
  return mlp_status.ok ? 0 : mlp_status.error_code;
}

extern "C" int qwen_prefill_attention_smoke_forward(
    const float* input_sequence,
    int seq_len,
  int attention_seq_tile,
  int attention_query_tile,
  int attention_key_tile,
  int attention_hidden_proj_tile,
  int attention_kv_proj_tile,
  int attention_head_dim_tile,
  int attention_query_heads_parallel,
  int attention_kv_heads_parallel,
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
    float* output_sequence) {
  const PrefillAttentionTileConfig tile_config = make_attention_tile_config(
      attention_seq_tile,
      attention_query_tile,
      attention_key_tile,
      attention_hidden_proj_tile,
      attention_kv_proj_tile,
      attention_head_dim_tile,
      attention_query_heads_parallel,
      attention_kv_heads_parallel);
  KernelStatus status = qwen_prefill_attention_kernel(
      input_sequence,
      seq_len,
      tile_config,
      input_layernorm_weight,
      kRmsNormEps,
      q_packed_weights,
      k_packed_weights,
      v_packed_weights,
      o_packed_weights,
      q_bias,
      k_bias,
      v_bias,
      q_scales,
      k_scales,
      v_scales,
      o_scales,
      k_cache,
      v_cache,
      output_sequence);
  return status.ok ? 0 : status.error_code;
}

extern "C" int qwen_prefill_mlp_smoke_forward(
    const float* attention_residual_sequence,
    int seq_len,
  int mlp_seq_tile,
  int mlp_hidden_tile,
  int mlp_ff_tile,
    const float* post_attention_layernorm_weight,
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* down_packed_weights,
    const float* gate_scales,
    const float* up_scales,
    const float* down_scales,
    float* output_sequence) {
  const PrefillMLPTileConfig tile_config = make_mlp_tile_config(mlp_seq_tile, mlp_hidden_tile, mlp_ff_tile);
  KernelStatus status = qwen_prefill_mlp_kernel(
      attention_residual_sequence,
      seq_len,
      tile_config,
      post_attention_layernorm_weight,
      kRmsNormEps,
      gate_packed_weights,
      up_packed_weights,
      down_packed_weights,
      gate_scales,
      up_scales,
      down_scales,
      output_sequence);
  return status.ok ? 0 : status.error_code;
}

extern "C" int qwen_prefill_layer_smoke_forward(
    const float* input_sequence,
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
    float* output_sequence) {
  const PrefillTileConfig tile_config = make_prefill_tile_config(
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
      mlp_ff_tile);
  std::vector<float> attention_output(static_cast<std::size_t>(seq_len) * kHiddenSize, 0.0f);
  KernelStatus attention_status = qwen_prefill_attention_kernel(
      input_sequence,
      seq_len,
      tile_config.attention,
      input_layernorm_weight,
      kRmsNormEps,
      q_packed_weights,
      k_packed_weights,
      v_packed_weights,
      o_packed_weights,
      q_bias,
      k_bias,
      v_bias,
      q_scales,
      k_scales,
      v_scales,
      o_scales,
      k_cache,
      v_cache,
      attention_output.data());
  if (!attention_status.ok) {
    return attention_status.error_code;
  }

  KernelStatus mlp_status = qwen_prefill_mlp_kernel(
      attention_output.data(),
      seq_len,
      tile_config.mlp,
      post_attention_layernorm_weight,
      kRmsNormEps,
      gate_packed_weights,
      up_packed_weights,
      down_packed_weights,
      gate_scales,
      up_scales,
      down_scales,
      output_sequence);
  return mlp_status.ok ? 0 : mlp_status.error_code;
}

extern "C" int qwen_prefill_top_smoke_forward(
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
    const std::uint8_t* weight_ddr,
    const float* scale_ddr,
    float* kv_cache_ddr,
    float* activation_ddr,
    std::uint8_t* weight_sram,
    float* kv_sram,
    std::int32_t* partial_sum_sram,
    float* softmax_sram,
    float* control_sram) {
  PrefillLayerDescriptor descriptor{
      layer_id,
      seq_len,
      make_prefill_tile_config(
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
          mlp_ff_tile),
      input_sequence_addr,
      output_sequence_addr,
      layer_weights_base_addr,
      layer_scales_base_addr,
      k_cache_base_addr,
      v_cache_base_addr,
      0,
  };
  PrefillTopLevelPorts ports{
      weight_ddr,
      scale_ddr,
      kv_cache_ddr,
      activation_ddr,
      weight_sram,
      kv_sram,
      partial_sum_sram,
      softmax_sram,
      control_sram,
  };
  KernelStatus status = qwen_prefill_top_wrapper(descriptor, ports);
  return status.ok ? 0 : status.error_code;
}