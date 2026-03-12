#include "qwen_decode_mlp_kernel.h"

namespace llm_accel {

KernelStatus qwen_decode_mlp_kernel(
    const scalar_t* attention_output_token,
    const packed_w4_t* gate_packed_weights,
    const packed_w4_t* up_packed_weights,
    const packed_w4_t* down_packed_weights,
    const scalar_t* gate_scales,
    const scalar_t* up_scales,
    const scalar_t* down_scales,
    scalar_t* output_token) {
  if (attention_output_token == nullptr || output_token == nullptr) {
    return {false, 1};
  }

  (void)gate_packed_weights;
  (void)up_packed_weights;
  (void)down_packed_weights;
  (void)gate_scales;
  (void)up_scales;
  (void)down_scales;

  for (int index = 0; index < kHiddenSize; ++index) {
    output_token[index] = attention_output_token[index];
  }

  return {true, 0};
}

}  // namespace llm_accel