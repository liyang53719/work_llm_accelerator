#pragma once

#include "qwen_prefill_mlp_kernel.h"

namespace llm_accel {

struct PrefillMlpBlockIO {
  const scalar_t* attention_residual;
  const scalar_t* post_attention_layernorm_weight;
  scalar_t rms_eps;
  const packed_w4_t* gate_packed_weights;
  const packed_w4_t* up_packed_weights;
  const packed_w4_t* down_packed_weights;
  const scalar_t* gate_scales;
  const scalar_t* up_scales;
  const scalar_t* down_scales;
  scalar_t* output_sequence;
};

KernelStatus qwen_prefill_mlp_block(
    int seq_len,
    const PrefillMLPTileConfig& tile_config,
    const PrefillMlpBlockIO& io);

}  // namespace llm_accel