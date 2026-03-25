#pragma once

#include "qwen_prefill_attention_kernel.h"

namespace llm_accel {

struct PrefillAttentionBlockIO {
  const scalar_t* input_sequence;
  const scalar_t* input_layernorm_weight;
  scalar_t rms_eps;
  const packed_w4_t* q_packed_weights;
  const packed_w4_t* k_packed_weights;
  const packed_w4_t* v_packed_weights;
  const packed_w4_t* o_packed_weights;
  const scalar_t* q_bias;
  const scalar_t* k_bias;
  const scalar_t* v_bias;
  const scalar_t* q_scales;
  const scalar_t* k_scales;
  const scalar_t* v_scales;
  const scalar_t* o_scales;
  scalar_t* k_cache;
  scalar_t* v_cache;
  scalar_t* attention_output;
};

KernelStatus qwen_prefill_attention_block(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const PrefillAttentionBlockIO& io);

}  // namespace llm_accel