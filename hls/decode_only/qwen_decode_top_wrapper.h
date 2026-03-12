#pragma once

#include "../common/llm_layer_dispatch.h"
#include "../common/llm_memory_layout.h"

namespace llm_accel {

struct DecodeTopLevelPorts {
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

KernelStatus qwen_decode_top_wrapper(
    const DecodeLayerDescriptor& descriptor,
    const DecodeTopLevelPorts& ports);

DispatchStatus qwen_decode_dispatch_layers(
    const DecodeLayerDescriptor* descriptors,
    int layer_count,
    const DecodeTopLevelPorts& ports);

}  // namespace llm_accel