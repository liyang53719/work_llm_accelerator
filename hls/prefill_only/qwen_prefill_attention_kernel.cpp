#include "qwen_prefill_attention_kernel.h"

namespace llm_accel {

KernelStatus qwen_prefill_attention_kernel(
    const scalar_t* input_sequence,
    int seq_len,
    int tile_m,
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
    scalar_t* output_sequence) {
  if (input_sequence == nullptr || output_sequence == nullptr || seq_len <= 0 || tile_m <= 0) {
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

  const int element_count = seq_len * kHiddenSize;
  for (int index = 0; index < element_count; ++index) {
    output_sequence[index] = input_sequence[index];
  }

  return {true, 0};
}

}  // namespace llm_accel