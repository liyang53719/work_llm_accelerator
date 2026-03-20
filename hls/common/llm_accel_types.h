#pragma once

#include <cstdint>

namespace llm_accel {

using packed_w4_t = std::uint8_t;
using act_t = std::int8_t;
using acc_t = std::int32_t;
using scalar_t = float;

struct PrefillAttentionTileConfig {
  int seq;
  int query;
  int key;
  int hidden_proj;
  int kv_proj;
  int head_dim;
  int query_heads_parallel;
  int kv_heads_parallel;
};

struct PrefillMLPTileConfig {
  int seq;
  int hidden;
  int ff;
};

struct PrefillTileConfig {
  PrefillAttentionTileConfig attention;
  PrefillMLPTileConfig mlp;
};

struct KernelStatus {
  bool ok;
  int error_code;
};

}  // namespace llm_accel
