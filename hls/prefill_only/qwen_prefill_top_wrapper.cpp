#include "qwen_prefill_top_wrapper.h"

#include "qwen_prefill_attention_kernel.h"

namespace llm_accel {
namespace {

constexpr int kInvalidDescriptorError = 20;
constexpr int kInvalidPortError = 21;

const packed_w4_t* weight_ptr(const packed_w4_t* base, std::uint64_t byte_offset) {
  return base + byte_offset;
}

const scalar_t* scalar_ptr(const scalar_t* base, std::uint64_t byte_offset) {
  return base + byte_offset / sizeof(scalar_t);
}

scalar_t* scalar_ptr(scalar_t* base, std::uint64_t byte_offset) {
  return base + byte_offset / sizeof(scalar_t);
}

}  // namespace

KernelStatus qwen_prefill_top_wrapper(
    const PrefillLayerDescriptor& descriptor,
    const PrefillTopLevelPorts& ports) {
  if (!valid_layer_id(descriptor.layer_id) || descriptor.seq_len <= 0 || descriptor.tile_m <= 0) {
    return {false, kInvalidDescriptorError};
  }
  if (ports.weight_ddr == nullptr || ports.scale_ddr == nullptr || ports.kv_cache_ddr == nullptr ||
      ports.activation_ddr == nullptr || ports.weight_sram == nullptr || ports.kv_sram == nullptr ||
      ports.partial_sum_sram == nullptr || ports.softmax_sram == nullptr || ports.control_sram == nullptr) {
    return {false, kInvalidPortError};
  }

  const LayerParameterLayout layout = default_layer_parameter_layout();
  const scalar_t* input_sequence = scalar_ptr(ports.activation_ddr, descriptor.input_sequence_addr);
  scalar_t* output_sequence = scalar_ptr(ports.activation_ddr, descriptor.output_sequence_addr);
  scalar_t* k_cache = scalar_ptr(ports.kv_cache_ddr, descriptor.k_cache_base_addr);
  scalar_t* v_cache = scalar_ptr(ports.kv_cache_ddr, descriptor.v_cache_base_addr);
  const scalar_t* input_layernorm_weight =
      scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.input_layernorm_weight_offset_bytes);

  const packed_w4_t* q_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.q_weight_offset_bytes);
  const packed_w4_t* k_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.k_weight_offset_bytes);
  const packed_w4_t* v_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.v_weight_offset_bytes);
  const packed_w4_t* o_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.o_weight_offset_bytes);

  const scalar_t* q_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.q_scale_offset_bytes);
  const scalar_t* k_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.k_scale_offset_bytes);
  const scalar_t* v_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.v_scale_offset_bytes);
  const scalar_t* o_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.o_scale_offset_bytes);

  return qwen_prefill_attention_kernel(
      input_sequence,
      descriptor.seq_len,
      descriptor.tile_m,
      input_layernorm_weight,
      kRmsNormEps,
      q_weights,
      k_weights,
      v_weights,
      o_weights,
      q_scales,
      k_scales,
      v_scales,
      o_scales,
      k_cache,
      v_cache,
      output_sequence);
}

DispatchStatus qwen_prefill_dispatch_layers(
    const PrefillLayerDescriptor* descriptors,
    int layer_count,
    const PrefillTopLevelPorts& ports) {
  if (descriptors == nullptr || layer_count <= 0) {
    return {false, kInvalidDescriptorError, 0};
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