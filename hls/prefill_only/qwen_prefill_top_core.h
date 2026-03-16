#pragma once

#include "qwen_prefill_top_wrapper.h"

namespace llm_accel {

constexpr int kPrefillInvalidDescriptorError = 20;
constexpr int kPrefillInvalidPortError = 21;
constexpr int kPrefillInvalidMemoryWindowError = 22;
constexpr int kPrefillSequenceCapacity = kDefaultPrefillSeqTile;

constexpr std::size_t kPrefillWeightWindowBytes = kPackedWeightBytesPerLayer;
constexpr std::size_t kPrefillScaleWindowElements =
    (kNormWeightBytesPerLayer + kProjectionBiasBytesPerLayer + kProjectionScaleBytesPerLayer) / sizeof(scalar_t);
constexpr std::size_t kPrefillKvCacheWindowElements =
    2U * static_cast<std::size_t>(kPrefillSequenceCapacity) * static_cast<std::size_t>(kNumKeyValueHeads * kHeadDim);
constexpr std::size_t kPrefillActivationWindowElements =
    3U * static_cast<std::size_t>(kPrefillSequenceCapacity) * static_cast<std::size_t>(kHiddenSize);

KernelStatus qwen_prefill_top_core(
    const PrefillLayerDescriptor& descriptor,
    const PrefillTopLevelPorts& ports,
    scalar_t* attention_scratch);

}  // namespace llm_accel