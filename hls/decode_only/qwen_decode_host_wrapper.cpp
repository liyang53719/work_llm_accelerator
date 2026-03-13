#include "qwen_decode_host_wrapper.h"

#include <array>
#include <vector>

#include "qwen_decode_attention_kernel.h"
#include "qwen_decode_mlp_kernel.h"
#include "qwen_decode_top_wrapper.h"

using namespace llm_accel;

extern "C" int qwen_decode_stub_forward(
    const float* input_token,
    int past_seq_len,
    float* output_token) {
  if (input_token == nullptr || output_token == nullptr || past_seq_len < 0) {
    return 1;
  }

  std::vector<float> attention_output(kHiddenSize, 0.0f);
  std::vector<float> k_cache((past_seq_len + 1) * kNumKeyValueHeads * kHeadDim, 0.0f);
  std::vector<float> v_cache((past_seq_len + 1) * kNumKeyValueHeads * kHeadDim, 0.0f);
    std::vector<float> input_layernorm_weight(kHiddenSize, 1.0f);
    std::vector<packed_w4_t> q_weights(static_cast<std::size_t>(kHiddenSize) * kHiddenSize / 2, 0);
    std::vector<packed_w4_t> k_weights(static_cast<std::size_t>(kHiddenSize) * kNumKeyValueHeads * kHeadDim / 2, 0);
    std::vector<packed_w4_t> v_weights(static_cast<std::size_t>(kHiddenSize) * kNumKeyValueHeads * kHeadDim / 2, 0);
    std::vector<packed_w4_t> o_weights(static_cast<std::size_t>(kHiddenSize) * kHiddenSize / 2, 0);
    std::vector<float> q_bias(kHiddenSize, 0.0f);
    std::vector<float> k_bias(kNumKeyValueHeads * kHeadDim, 0.0f);
    std::vector<float> v_bias(kNumKeyValueHeads * kHeadDim, 0.0f);
    std::vector<float> q_scales(kHiddenSize, 1.0f);
    std::vector<float> k_scales(kNumKeyValueHeads * kHeadDim, 1.0f);
    std::vector<float> v_scales(kNumKeyValueHeads * kHeadDim, 1.0f);
    std::vector<float> o_scales(kHiddenSize, 1.0f);

  KernelStatus attention_status = qwen_decode_attention_kernel(
      input_token,
      past_seq_len,
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

  KernelStatus mlp_status = qwen_decode_mlp_kernel(
      attention_output.data(),
      input_layernorm_weight.data(),
      kRmsNormEps,
      q_weights.data(),
      q_weights.data(),
      o_weights.data(),
      q_scales.data(),
      q_scales.data(),
      o_scales.data(),
      output_token);
  return mlp_status.ok ? 0 : mlp_status.error_code;
}

extern "C" int qwen_decode_attention_smoke_forward(
    const float* input_token,
    int past_seq_len,
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
    float* output_token) {
  KernelStatus status = qwen_decode_attention_kernel(
      input_token,
      past_seq_len,
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
      output_token);
  return status.ok ? 0 : status.error_code;
}

extern "C" int qwen_decode_top_smoke_forward(
    int layer_id,
    int past_seq_len,
    std::uint64_t input_token_addr,
    std::uint64_t output_token_addr,
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
  DecodeLayerDescriptor descriptor{
      layer_id,
      past_seq_len,
      input_token_addr,
      output_token_addr,
      layer_weights_base_addr,
      layer_scales_base_addr,
      k_cache_base_addr,
      v_cache_base_addr,
      0,
  };
  DecodeTopLevelPorts ports{
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
  KernelStatus status = qwen_decode_top_wrapper(descriptor, ports);
  return status.ok ? 0 : status.error_code;
}

extern "C" int qwen_decode_layer_smoke_forward(
    const float* input_token,
    int past_seq_len,
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
    float* output_token) {
  std::vector<float> attention_output(kHiddenSize, 0.0f);
  KernelStatus attention_status = qwen_decode_attention_kernel(
      input_token,
      past_seq_len,
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

  KernelStatus mlp_status = qwen_decode_mlp_kernel(
      attention_output.data(),
      post_attention_layernorm_weight,
      kRmsNormEps,
      gate_packed_weights,
      up_packed_weights,
      down_packed_weights,
      gate_scales,
      up_scales,
      down_scales,
      output_token);
  return mlp_status.ok ? 0 : mlp_status.error_code;
}

extern "C" int qwen_decode_mlp_smoke_forward(
    const float* attention_residual_token,
    const float* post_attention_layernorm_weight,
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* down_packed_weights,
    const float* gate_scales,
    const float* up_scales,
    const float* down_scales,
    float* output_token) {
  KernelStatus status = qwen_decode_mlp_kernel(
      attention_residual_token,
      post_attention_layernorm_weight,
      kRmsNormEps,
      gate_packed_weights,
      up_packed_weights,
      down_packed_weights,
      gate_scales,
      up_scales,
      down_scales,
      output_token);
  return status.ok ? 0 : status.error_code;
}