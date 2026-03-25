#pragma once

#include "qwen_prefill_top_wrapper.h"

namespace llm_accel {

constexpr int kPrefillInvalidDescriptorError = 20;
constexpr int kPrefillInvalidPortError = 21;
constexpr int kPrefillInvalidMemoryWindowError = 22;
constexpr int kPrefillSequenceCapacity = kDefaultPrefillSeqTile;

constexpr std::uint64_t kPrefillWeightWindowBytes = kLayerParameterBytes;
constexpr std::uint64_t kPrefillScaleWindowElements =
    (kLayerParameterBytes + sizeof(scalar_t) - 1ULL) / sizeof(scalar_t);
constexpr std::uint64_t kPrefillKvCacheWindowElements =
    2ULL * static_cast<std::uint64_t>(kPrefillSequenceCapacity) * static_cast<std::uint64_t>(kNumKeyValueHeads * kHeadDim);
constexpr std::uint64_t kPrefillActivationWindowElements =
    3ULL * static_cast<std::uint64_t>(kPrefillSequenceCapacity) * static_cast<std::uint64_t>(kHiddenSize);

KernelStatus qwen_prefill_top_core(
    const PrefillLayerDescriptor& descriptor,
    const PrefillTopLevelPorts& ports);

}  // namespace llm_accel