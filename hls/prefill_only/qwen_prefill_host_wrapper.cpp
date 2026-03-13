#include "qwen_prefill_host_wrapper.h"

#include <cstdint>
#include <vector>

#include "qwen_prefill_attention_kernel.h"
#include "qwen_prefill_mlp_kernel.h"

using namespace llm_accel;

extern "C" int qwen_prefill_stub_forward(
    const float* input_sequence,
    int seq_len,
    int tile_m,
    float* output_sequence) {
  if (input_sequence == nullptr || output_sequence == nullptr || seq_len <= 0 || tile_m <= 0) {
    return 1;
  }

  std::vector<float> attention_output(seq_len * kHiddenSize, 0.0f);
  std::vector<float> k_cache(seq_len * kNumKeyValueHeads * kHeadDim, 0.0f);
  std::vector<float> v_cache(seq_len * kNumKeyValueHeads * kHeadDim, 0.0f);
    std::vector<float> input_layernorm_weight(kHiddenSize, 1.0f);
    std::vector<packed_w4_t> q_weights(static_cast<std::size_t>(kHiddenSize) * kHiddenSize / 2, 0);
    std::vector<packed_w4_t> k_weights(static_cast<std::size_t>(kHiddenSize) * kNumKeyValueHeads * kHeadDim / 2, 0);
    std::vector<packed_w4_t> v_weights(static_cast<std::size_t>(kHiddenSize) * kNumKeyValueHeads * kHeadDim / 2, 0);
    std::vector<packed_w4_t> o_weights(static_cast<std::size_t>(kHiddenSize) * kHiddenSize / 2, 0);
    std::vector<float> q_scales(kHiddenSize, 1.0f);
    std::vector<float> k_scales(kNumKeyValueHeads * kHeadDim, 1.0f);
    std::vector<float> v_scales(kNumKeyValueHeads * kHeadDim, 1.0f);
    std::vector<float> o_scales(kHiddenSize, 1.0f);

  KernelStatus attention_status = qwen_prefill_attention_kernel(
      input_sequence,
      seq_len,
      tile_m,
      input_layernorm_weight.data(),
      kRmsNormEps,
      q_weights.data(),
      k_weights.data(),
      v_weights.data(),
      o_weights.data(),
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
      tile_m,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      output_sequence);
  return mlp_status.ok ? 0 : mlp_status.error_code;
}

extern "C" int qwen_prefill_attention_smoke_forward(
    const float* input_sequence,
    int seq_len,
    int tile_m,
    const float* input_layernorm_weight,
    const std::uint8_t* q_packed_weights,
    const std::uint8_t* k_packed_weights,
    const std::uint8_t* v_packed_weights,
    const std::uint8_t* o_packed_weights,
    const float* q_scales,
    const float* k_scales,
    const float* v_scales,
    const float* o_scales,
    float* k_cache,
    float* v_cache,
    float* output_sequence) {
  KernelStatus status = qwen_prefill_attention_kernel(
      input_sequence,
      seq_len,
      tile_m,
      input_layernorm_weight,
      kRmsNormEps,
      q_packed_weights,
      k_packed_weights,
      v_packed_weights,
      o_packed_weights,
      q_scales,
      k_scales,
      v_scales,
      o_scales,
      k_cache,
      v_cache,
      output_sequence);
  return status.ok ? 0 : status.error_code;
}