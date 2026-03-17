#pragma once

#include "../catapult_shims/cstddef.h"
#include "../catapult_shims/cstdint.h"

#include "qwen2_model_config.h"

namespace llm_accel {

constexpr std::size_t kSramBudgetBytes = 1U << 20;
constexpr std::size_t kWeightBufferBytes = 256U << 10;
constexpr std::size_t kKvWorkingSetBytes = 256U << 10;
constexpr std::size_t kPartialSumBytes = 128U << 10;
constexpr std::size_t kSoftmaxScratchBytes = 128U << 10;
constexpr std::size_t kControlScratchBytes =
    kSramBudgetBytes - kWeightBufferBytes - kKvWorkingSetBytes - kPartialSumBytes - kSoftmaxScratchBytes;

constexpr std::size_t kPackedWeightBytesPerLayer =
    static_cast<std::size_t>(kHiddenSize) * (kHiddenSize + 2 * kNumKeyValueHeads * kHeadDim + kIntermediateSize + kHiddenSize) / 2;

constexpr std::size_t kNormWeightBytesPerLayer =
    2U * static_cast<std::size_t>(kHiddenSize) * sizeof(float);

constexpr std::size_t kQWeightBytes =
  static_cast<std::size_t>(kHiddenSize) * static_cast<std::size_t>(kHiddenSize) / 2U;
constexpr std::size_t kKWeightBytes =
  static_cast<std::size_t>(kHiddenSize) * static_cast<std::size_t>(kNumKeyValueHeads * kHeadDim) / 2U;
constexpr std::size_t kVWeightBytes = kKWeightBytes;
constexpr std::size_t kOWeightBytes = kQWeightBytes;
constexpr std::size_t kGateWeightBytes =
  static_cast<std::size_t>(kHiddenSize) * static_cast<std::size_t>(kIntermediateSize) / 2U;
constexpr std::size_t kUpWeightBytes = kGateWeightBytes;
constexpr std::size_t kDownWeightBytes =
  static_cast<std::size_t>(kIntermediateSize) * static_cast<std::size_t>(kHiddenSize) / 2U;

constexpr std::size_t kProjectionScaleBytesPerLayer =
    static_cast<std::size_t>(3 * kHiddenSize + kIntermediateSize + kHiddenSize) * sizeof(float);

constexpr std::size_t kProjectionBiasBytesPerLayer =
  static_cast<std::size_t>(kHiddenSize + 2 * kNumKeyValueHeads * kHeadDim) * sizeof(float);

constexpr std::size_t kLayerParameterBytes =
  kPackedWeightBytesPerLayer + kNormWeightBytesPerLayer + kProjectionBiasBytesPerLayer + kProjectionScaleBytesPerLayer;

constexpr std::size_t kKvElementsPerTokenPerLayer =
    static_cast<std::size_t>(2 * kNumKeyValueHeads * kHeadDim);

constexpr std::size_t kKvBytesPerTokenPerLayer = kKvElementsPerTokenPerLayer * sizeof(float);

struct LayerParameterLayout {
  std::uint64_t input_layernorm_weight_offset_bytes;
  std::uint64_t q_weight_offset_bytes;
  std::uint64_t k_weight_offset_bytes;
  std::uint64_t v_weight_offset_bytes;
  std::uint64_t o_weight_offset_bytes;
  std::uint64_t post_attention_layernorm_weight_offset_bytes;
  std::uint64_t gate_weight_offset_bytes;
  std::uint64_t up_weight_offset_bytes;
  std::uint64_t down_weight_offset_bytes;
  std::uint64_t q_bias_offset_bytes;
  std::uint64_t k_bias_offset_bytes;
  std::uint64_t v_bias_offset_bytes;
  std::uint64_t q_scale_offset_bytes;
  std::uint64_t k_scale_offset_bytes;
  std::uint64_t v_scale_offset_bytes;
  std::uint64_t o_scale_offset_bytes;
  std::uint64_t gate_scale_offset_bytes;
  std::uint64_t up_scale_offset_bytes;
  std::uint64_t down_scale_offset_bytes;
};

struct KvCacheLayout {
  std::uint64_t k_base_offset_bytes;
  std::uint64_t v_base_offset_bytes;
  std::uint64_t layer_stride_bytes;
  std::uint64_t token_stride_bytes;
};

inline LayerParameterLayout default_layer_parameter_layout() {
  LayerParameterLayout layout{};
  std::uint64_t offset = 0;

  layout.input_layernorm_weight_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kHiddenSize * sizeof(float));
  layout.q_weight_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kQWeightBytes);
  layout.k_weight_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kKWeightBytes);
  layout.v_weight_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kVWeightBytes);
  layout.o_weight_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kOWeightBytes);
  layout.post_attention_layernorm_weight_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kHiddenSize * sizeof(float));
  layout.gate_weight_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kGateWeightBytes);
  layout.up_weight_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kUpWeightBytes);
  layout.down_weight_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kDownWeightBytes);
  layout.q_bias_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kHiddenSize * sizeof(float));
  layout.k_bias_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kNumKeyValueHeads * kHeadDim * sizeof(float));
  layout.v_bias_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kNumKeyValueHeads * kHeadDim * sizeof(float));

  layout.q_scale_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kHiddenSize * sizeof(float));
  layout.k_scale_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kNumKeyValueHeads * kHeadDim * sizeof(float));
  layout.v_scale_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kNumKeyValueHeads * kHeadDim * sizeof(float));
  layout.o_scale_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kHiddenSize * sizeof(float));
  layout.gate_scale_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kIntermediateSize * sizeof(float));
  layout.up_scale_offset_bytes = offset;
  offset += static_cast<std::uint64_t>(kIntermediateSize * sizeof(float));
  layout.down_scale_offset_bytes = offset;

  return layout;
}

constexpr KvCacheLayout default_kv_cache_layout() {
  return KvCacheLayout{
      0,
      static_cast<std::uint64_t>(kNumHiddenLayers) * static_cast<std::uint64_t>(kMaxSequenceLength) *
          static_cast<std::uint64_t>(kNumKeyValueHeads * kHeadDim) * sizeof(float),
      static_cast<std::uint64_t>(kMaxSequenceLength) *
          static_cast<std::uint64_t>(kNumKeyValueHeads * kHeadDim) * sizeof(float),
      static_cast<std::uint64_t>(kNumKeyValueHeads * kHeadDim) * sizeof(float),
  };
}

constexpr std::uint64_t layer_parameter_base_offset(int layer_id) {
  return static_cast<std::uint64_t>(layer_id) * static_cast<std::uint64_t>(kLayerParameterBytes);
}

constexpr std::uint64_t kv_cache_layer_offset_bytes(int layer_id, int token_index) {
  return (static_cast<std::uint64_t>(layer_id) * static_cast<std::uint64_t>(kMaxSequenceLength) +
          static_cast<std::uint64_t>(token_index)) *
         static_cast<std::uint64_t>(kKvBytesPerTokenPerLayer);
}

}  // namespace llm_accel