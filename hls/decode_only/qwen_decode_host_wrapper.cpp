#include "qwen_decode_host_wrapper.h"

#include <vector>

#include "qwen_decode_attention_kernel.h"
#include "qwen_decode_mlp_kernel.h"

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

  KernelStatus attention_status = qwen_decode_attention_kernel(
      input_token,
      past_seq_len,
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

  KernelStatus mlp_status = qwen_decode_mlp_kernel(
      attention_output.data(),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      output_token);
  return mlp_status.ok ? 0 : mlp_status.error_code;
}