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
  int tile_m;
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

}  // namespace llm_accel