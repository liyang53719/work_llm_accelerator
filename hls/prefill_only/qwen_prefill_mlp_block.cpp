#include "qwen_prefill_mlp_block.h"

namespace llm_accel {

KernelStatus qwen_prefill_mlp_block(
    int seq_len,
    const PrefillMLPTileConfig& tile_config,
    const PrefillMlpBlockIO& io) {
  if (io.attention_residual == nullptr || io.post_attention_layernorm_weight == nullptr ||
      io.gate_packed_weights == nullptr || io.up_packed_weights == nullptr || io.down_packed_weights == nullptr ||
      io.gate_scales == nullptr || io.up_scales == nullptr || io.down_scales == nullptr ||
      io.output_sequence == nullptr) {
    return {false, 21};
  }

  return qwen_prefill_mlp_kernel(
      io.attention_residual,
      seq_len,
      tile_config,
      io.post_attention_layernorm_weight,
      io.rms_eps,
      io.gate_packed_weights,
      io.up_packed_weights,
      io.down_packed_weights,
      io.gate_scales,
      io.up_scales,
      io.down_scales,
      io.output_sequence);
}

}  // namespace llm_accel