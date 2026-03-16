#include "qwen_prefill_top_core.h"

#include "qwen_prefill_attention_kernel.h"
#include "qwen_prefill_mlp_kernel.h"

namespace llm_accel {
namespace {

constexpr int kKvWidth = kNumKeyValueHeads * kHeadDim;
constexpr std::uint64_t kPrefillWeightWindowBytes64 = static_cast<std::uint64_t>(kPrefillWeightWindowBytes);
constexpr std::uint64_t kPrefillScaleWindowBytes64 =
    static_cast<std::uint64_t>(kPrefillScaleWindowElements) * sizeof(scalar_t);
constexpr std::uint64_t kPrefillKvCacheWindowBytes64 =
    static_cast<std::uint64_t>(kPrefillKvCacheWindowElements) * sizeof(scalar_t);
constexpr std::uint64_t kPrefillActivationWindowBytes64 =
    static_cast<std::uint64_t>(kPrefillActivationWindowElements) * sizeof(scalar_t);

const packed_w4_t* weight_ptr(const packed_w4_t* base, std::uint64_t byte_offset) {
  return base + byte_offset;
}

const scalar_t* scalar_ptr(const scalar_t* base, std::uint64_t byte_offset) {
  return base + byte_offset / sizeof(scalar_t);
}

scalar_t* scalar_ptr(scalar_t* base, std::uint64_t byte_offset) {
  return base + byte_offset / sizeof(scalar_t);
}

bool scalar_aligned(std::uint64_t byte_offset) {
  return (byte_offset % sizeof(scalar_t)) == 0U;
}

bool range_fits(std::uint64_t byte_offset, std::uint64_t byte_count, std::uint64_t capacity_bytes) {
  return byte_offset <= capacity_bytes && byte_count <= capacity_bytes - byte_offset;
}

bool ranges_overlap(std::uint64_t lhs_offset, std::uint64_t lhs_size, std::uint64_t rhs_offset, std::uint64_t rhs_size) {
  return lhs_offset < rhs_offset + rhs_size && rhs_offset < lhs_offset + lhs_size;
}

bool valid_memory_windows(const PrefillLayerDescriptor& descriptor) {
  if (descriptor.seq_len <= 0 || descriptor.seq_len > kPrefillSequenceCapacity) {
    return false;
  }

  const LayerParameterLayout layout = default_layer_parameter_layout();
  const std::uint64_t sequence_bytes =
      static_cast<std::uint64_t>(descriptor.seq_len) * static_cast<std::uint64_t>(kHiddenSize) * sizeof(scalar_t);
  const std::uint64_t kv_bytes =
      static_cast<std::uint64_t>(descriptor.seq_len) * static_cast<std::uint64_t>(kKvWidth) * sizeof(scalar_t);
  const std::uint64_t packed_weight_bytes =
      layout.down_weight_offset_bytes + static_cast<std::uint64_t>(kDownWeightBytes) - layout.q_weight_offset_bytes;
  const std::uint64_t scale_bytes = layout.down_scale_offset_bytes + static_cast<std::uint64_t>(kHiddenSize) * sizeof(scalar_t);

  if (!scalar_aligned(descriptor.input_sequence_addr) || !scalar_aligned(descriptor.output_sequence_addr) ||
      !scalar_aligned(descriptor.k_cache_base_addr) || !scalar_aligned(descriptor.v_cache_base_addr) ||
      !scalar_aligned(descriptor.scratch_base_addr) || !scalar_aligned(descriptor.layer_scales_base_addr)) {
    return false;
  }

  if (!range_fits(descriptor.input_sequence_addr, sequence_bytes, kPrefillActivationWindowBytes64) ||
      !range_fits(descriptor.output_sequence_addr, sequence_bytes, kPrefillActivationWindowBytes64) ||
      !range_fits(descriptor.scratch_base_addr, sequence_bytes, kPrefillActivationWindowBytes64) ||
      !range_fits(descriptor.k_cache_base_addr, kv_bytes, kPrefillKvCacheWindowBytes64) ||
      !range_fits(descriptor.v_cache_base_addr, kv_bytes, kPrefillKvCacheWindowBytes64) ||
      !range_fits(descriptor.layer_weights_base_addr + layout.q_weight_offset_bytes, packed_weight_bytes, kPrefillWeightWindowBytes64) ||
      !range_fits(descriptor.layer_scales_base_addr, scale_bytes, kPrefillScaleWindowBytes64)) {
    return false;
  }

  if (ranges_overlap(descriptor.input_sequence_addr, sequence_bytes, descriptor.output_sequence_addr, sequence_bytes) ||
      ranges_overlap(descriptor.input_sequence_addr, sequence_bytes, descriptor.scratch_base_addr, sequence_bytes) ||
      ranges_overlap(descriptor.output_sequence_addr, sequence_bytes, descriptor.scratch_base_addr, sequence_bytes) ||
      ranges_overlap(descriptor.k_cache_base_addr, kv_bytes, descriptor.v_cache_base_addr, kv_bytes)) {
    return false;
  }

  return true;
}

}  // namespace

KernelStatus qwen_prefill_top_core(
    const PrefillLayerDescriptor& descriptor,
    const PrefillTopLevelPorts& ports,
    scalar_t* attention_scratch) {
  if (!valid_layer_id(descriptor.layer_id) || descriptor.seq_len <= 0 ||
      !valid_prefill_tile_config(descriptor.tile_config)) {
    return {false, kPrefillInvalidDescriptorError};
  }
  if (ports.weight_ddr == nullptr || ports.scale_ddr == nullptr || ports.kv_cache_ddr == nullptr ||
      ports.activation_ddr == nullptr || ports.weight_sram == nullptr || ports.kv_sram == nullptr ||
      ports.partial_sum_sram == nullptr || ports.softmax_sram == nullptr || ports.control_sram == nullptr ||
      attention_scratch == nullptr) {
    return {false, kPrefillInvalidPortError};
  }
  if (!valid_memory_windows(descriptor)) {
    return {false, kPrefillInvalidMemoryWindowError};
  }

  const LayerParameterLayout layout = default_layer_parameter_layout();
  const scalar_t* input_sequence = scalar_ptr(ports.activation_ddr, descriptor.input_sequence_addr);
  scalar_t* output_sequence = scalar_ptr(ports.activation_ddr, descriptor.output_sequence_addr);
  scalar_t* k_cache = scalar_ptr(ports.kv_cache_ddr, descriptor.k_cache_base_addr);
  scalar_t* v_cache = scalar_ptr(ports.kv_cache_ddr, descriptor.v_cache_base_addr);
  const scalar_t* input_layernorm_weight =
      scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.input_layernorm_weight_offset_bytes);
  const scalar_t* post_attention_layernorm_weight =
      scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.post_attention_layernorm_weight_offset_bytes);

  const packed_w4_t* q_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.q_weight_offset_bytes);
  const packed_w4_t* k_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.k_weight_offset_bytes);
  const packed_w4_t* v_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.v_weight_offset_bytes);
  const packed_w4_t* o_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.o_weight_offset_bytes);
  const scalar_t* q_bias = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.q_bias_offset_bytes);
  const scalar_t* k_bias = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.k_bias_offset_bytes);
  const scalar_t* v_bias = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.v_bias_offset_bytes);

  const scalar_t* q_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.q_scale_offset_bytes);
  const scalar_t* k_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.k_scale_offset_bytes);
  const scalar_t* v_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.v_scale_offset_bytes);
  const scalar_t* o_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.o_scale_offset_bytes);
  const scalar_t* gate_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.gate_scale_offset_bytes);
  const scalar_t* up_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.up_scale_offset_bytes);
  const scalar_t* down_scales = scalar_ptr(ports.scale_ddr, descriptor.layer_scales_base_addr + layout.down_scale_offset_bytes);

  const packed_w4_t* gate_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.gate_weight_offset_bytes);
  const packed_w4_t* up_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.up_weight_offset_bytes);
  const packed_w4_t* down_weights =
      weight_ptr(ports.weight_ddr, descriptor.layer_weights_base_addr + layout.down_weight_offset_bytes);

  KernelStatus attention_status = qwen_prefill_attention_kernel(
      input_sequence,
      descriptor.seq_len,
      descriptor.tile_config.attention,
      input_layernorm_weight,
      kRmsNormEps,
      q_weights,
      k_weights,
      v_weights,
      o_weights,
      q_bias,
      k_bias,
      v_bias,
      q_scales,
      k_scales,
      v_scales,
      o_scales,
      k_cache,
      v_cache,
      attention_scratch);
  if (!attention_status.ok) {
    return attention_status;
  }

  return qwen_prefill_mlp_kernel(
      attention_scratch,
      descriptor.seq_len,
      descriptor.tile_config.mlp,
      post_attention_layernorm_weight,
      kRmsNormEps,
      gate_weights,
      up_weights,
      down_weights,
      gate_scales,
      up_scales,
      down_scales,
      output_sequence);
}

}  // namespace llm_accel