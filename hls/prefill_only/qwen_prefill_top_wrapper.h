#pragma once

#include "../common/llm_layer_dispatch.h"
#include "../common/llm_memory_layout.h"

namespace llm_accel {

struct PrefillTopLevelPorts {
  const packed_w4_t* weight_ddr;
  const scalar_t* scale_ddr;
  scalar_t* kv_cache_ddr;
  scalar_t* activation_ddr;
  packed_w4_t* weight_sram;
  scalar_t* kv_sram;
  acc_t* partial_sum_sram;
  scalar_t* softmax_sram;
  scalar_t* control_sram;
};

KernelStatus qwen_prefill_top_wrapper(
    const PrefillLayerDescriptor& descriptor,
    const PrefillTopLevelPorts& ports);

DispatchStatus qwen_prefill_dispatch_layers(
    const PrefillLayerDescriptor* descriptors,
    int layer_count,
    const PrefillTopLevelPorts& ports);

}  // namespace llm_accel