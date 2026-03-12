#include "qwen_decode_attention_kernel.h"

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
    scalar_t* output_token) {
  if (input_token == nullptr || output_token == nullptr || past_seq_len < 0) {
    return {false, 1};
  }

  (void)q_packed_weights;
  (void)k_packed_weights;
  (void)v_packed_weights;
  (void)o_packed_weights;
  (void)q_scales;
  (void)k_scales;
  (void)v_scales;
  (void)o_scales;
  (void)k_cache;
  (void)v_cache;

  for (int index = 0; index < kHiddenSize; ++index) {
    output_token[index] = input_token[index];
  }

  return {true, 0};
}

}  // namespace llm_accel