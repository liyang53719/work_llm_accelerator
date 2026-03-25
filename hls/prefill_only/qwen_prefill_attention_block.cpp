#include "qwen_prefill_attention_block.h"

namespace llm_accel {

KernelStatus qwen_prefill_attention_block(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const PrefillAttentionBlockIO& io) {
  if (io.input_sequence == nullptr || io.input_layernorm_weight == nullptr || io.q_packed_weights == nullptr ||
      io.k_packed_weights == nullptr || io.v_packed_weights == nullptr || io.o_packed_weights == nullptr ||
      io.q_bias == nullptr || io.k_bias == nullptr || io.v_bias == nullptr || io.q_scales == nullptr ||
      io.k_scales == nullptr || io.v_scales == nullptr || io.o_scales == nullptr || io.k_cache == nullptr ||
      io.v_cache == nullptr || io.attention_output == nullptr) {
    return {false, 21};
  }

  return qwen_prefill_attention_kernel(
      io.input_sequence,
      seq_len,
      tile_config,
      io.input_layernorm_weight,
      io.rms_eps,
      io.q_packed_weights,
      io.k_packed_weights,
      io.v_packed_weights,
      io.o_packed_weights,
      io.q_bias,
      io.k_bias,
      io.v_bias,
      io.q_scales,
      io.k_scales,
      io.v_scales,
      io.o_scales,
      io.k_cache,
      io.v_cache,
      io.attention_output);
}

}  // namespace llm_accel