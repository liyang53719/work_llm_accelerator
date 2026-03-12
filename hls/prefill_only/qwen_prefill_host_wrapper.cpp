#include "qwen_prefill_host_wrapper.h"

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

  KernelStatus attention_status = qwen_prefill_attention_kernel(
      input_sequence,
      seq_len,
      tile_m,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
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