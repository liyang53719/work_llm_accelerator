#pragma once

#include <cstdint>

#include "llm_accel_types.h"
#include "qwen2_model_config.h"

namespace llm_accel {

enum class WorkloadKind : std::uint8_t {
  kDecode = 0,
  kPrefill = 1,
};

struct DecodeLayerDescriptor {
  int layer_id;
  int past_seq_len;
  std::uint64_t input_token_addr;
  std::uint64_t output_token_addr;
  std::uint64_t layer_weights_base_addr;
  std::uint64_t layer_scales_base_addr;
  std::uint64_t k_cache_base_addr;
  std::uint64_t v_cache_base_addr;
  std::uint64_t scratch_base_addr;
};

struct PrefillLayerDescriptor {
  int layer_id;
  int seq_len;
  PrefillTileConfig tile_config;
  std::uint64_t input_sequence_addr;
  std::uint64_t output_sequence_addr;
  std::uint64_t layer_weights_base_addr;
  std::uint64_t layer_scales_base_addr;
  std::uint64_t k_cache_base_addr;
  std::uint64_t v_cache_base_addr;
  std::uint64_t scratch_base_addr;
};

struct DispatchStatus {
  bool ok;
  int error_code;
  int completed_layer_count;
};

constexpr bool valid_layer_id(int layer_id) {
  return layer_id >= 0 && layer_id < kNumHiddenLayers;
}

inline bool valid_prefill_tile_config(const PrefillTileConfig& tile_config) {
  const bool valid_attention_head_dim =
      tile_config.attention.head_dim == kAttentionMacCols || tile_config.attention.head_dim == kHeadDim;
  return tile_config.attention.seq > 0 &&
         tile_config.attention.query == kAttentionMacRows &&
         tile_config.attention.key == kAttentionMacCols &&
         tile_config.attention.hidden_proj == kAttentionMacCols &&
         tile_config.attention.kv_proj == kAttentionMacCols &&
         valid_attention_head_dim &&
         tile_config.attention.query_heads_parallel > 0 && tile_config.attention.kv_heads_parallel > 0 &&
         tile_config.mlp.seq > 0 && tile_config.mlp.hidden > 0 && tile_config.mlp.ff > 0;
}

}  // namespace llm_accel