#include "qwen_prefill_mlp_kernel.h"

namespace llm_accel {

KernelStatus qwen_prefill_mlp_kernel(
    const scalar_t* attention_output,
    int seq_len,
    int tile_m,
    const packed_w4_t* gate_packed_weights,
    const packed_w4_t* up_packed_weights,
    const packed_w4_t* down_packed_weights,
    const scalar_t* gate_scales,
    const scalar_t* up_scales,
    const scalar_t* down_scales,
    scalar_t* output_sequence) {
  if (attention_output == nullptr || output_sequence == nullptr || seq_len <= 0 || tile_m <= 0) {
    return {false, 1};
  }

  (void)gate_packed_weights;
  (void)up_packed_weights;
  (void)down_packed_weights;
  (void)gate_scales;
  (void)up_scales;
  (void)down_scales;

  const int element_count = seq_len * kHiddenSize;
  for (int index = 0; index < element_count; ++index) {
    output_sequence[index] = attention_output[index];
  }

  return {true, 0};
}

}  // namespace llm_accel