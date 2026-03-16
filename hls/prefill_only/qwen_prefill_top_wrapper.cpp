#include "qwen_prefill_top_wrapper.h"

#include <vector>

#include "qwen_prefill_top_core.h"
#include "qwen_prefill_attention_kernel.h"
#include "qwen_prefill_mlp_kernel.h"

namespace llm_accel {

KernelStatus qwen_prefill_top_wrapper(
    const PrefillLayerDescriptor& descriptor,
    const PrefillTopLevelPorts& ports) {
  if (descriptor.seq_len <= 0 || descriptor.seq_len > kMaxSequenceLength) {
    return {false, kPrefillInvalidDescriptorError};
  }
  if (descriptor.seq_len > kPrefillSequenceCapacity) {
    return {false, kPrefillInvalidDescriptorError};
  }

  std::vector<scalar_t> attention_output(static_cast<std::size_t>(descriptor.seq_len) * kHiddenSize, 0.0f);
  return qwen_prefill_top_core(descriptor, ports, attention_output.data());
}

DispatchStatus qwen_prefill_dispatch_layers(
    const PrefillLayerDescriptor* descriptors,
    int layer_count,
    const PrefillTopLevelPorts& ports) {
  if (descriptors == nullptr || layer_count <= 0) {
    return {false, kPrefillInvalidDescriptorError, 0};
  }

  for (int index = 0; index < layer_count; ++index) {
    KernelStatus status = qwen_prefill_top_wrapper(descriptors[index], ports);
    if (!status.ok) {
      return {false, status.error_code, index};
    }
  }
  return {true, 0, layer_count};
}

}  // namespace llm_accel