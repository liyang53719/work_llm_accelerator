#include "qwen_prefill_mlp_kernel.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace {

float decode_int4_weight(llm_accel::packed_w4_t packed_value, bool high_nibble) {
  const int nibble = high_nibble ? static_cast<int>((packed_value >> 4) & 0xF) : static_cast<int>(packed_value & 0xF);
  const int signed_nibble = nibble >= 8 ? nibble - 16 : nibble;
  return static_cast<float>(signed_nibble);
}

float dequantized_weight(
    const llm_accel::packed_w4_t* packed_weights,
    const llm_accel::scalar_t* scales,
    int out_index,
    int in_index,
    int in_dim) {
  const std::size_t flat_index = static_cast<std::size_t>(out_index) * in_dim + in_index;
  const llm_accel::packed_w4_t packed_value = packed_weights[flat_index / 2];
  const bool high_nibble = (flat_index & 1U) != 0U;
  return decode_int4_weight(packed_value, high_nibble) * scales[out_index];
}

void rmsnorm_token(
    const llm_accel::scalar_t* input,
    const llm_accel::scalar_t* weight,
    llm_accel::scalar_t rms_eps,
    llm_accel::scalar_t* output) {
  double mean_square = 0.0;
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    mean_square += static_cast<double>(input[dim]) * static_cast<double>(input[dim]);
  }
  mean_square /= static_cast<double>(llm_accel::kHiddenSize);
  const double inv_rms = 1.0 / std::sqrt(mean_square + static_cast<double>(rms_eps));
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    output[dim] = static_cast<float>(static_cast<double>(input[dim]) * inv_rms * static_cast<double>(weight[dim]));
  }
}

void project_tiled(
    const llm_accel::scalar_t* input_token,
    const llm_accel::packed_w4_t* packed_weights,
    const llm_accel::scalar_t* scales,
    int in_dim,
    int out_dim,
    llm_accel::scalar_t* output) {
  std::array<llm_accel::scalar_t, llm_accel::kTileN> input_tile{};
  std::array<llm_accel::scalar_t, llm_accel::kTileN> partial_sum{};

  for (int out_base = 0; out_base < out_dim; out_base += llm_accel::kTileN) {
    const int out_extent = std::min(llm_accel::kTileN, out_dim - out_base);
    partial_sum.fill(0.0f);

    for (int in_base = 0; in_base < in_dim; in_base += llm_accel::kTileN) {
      const int in_extent = std::min(llm_accel::kTileN, in_dim - in_base);
      for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
        input_tile[in_offset] = input_token[in_base + in_offset];
      }
      for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
        const int out_index = out_base + out_offset;
        float accum = partial_sum[out_offset];
        for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
          accum += input_tile[in_offset] * dequantized_weight(packed_weights, scales, out_index, in_base + in_offset, in_dim);
        }
        partial_sum[out_offset] = accum;
      }
    }

    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      output[out_base + out_offset] = partial_sum[out_offset];
    }
  }
}

inline float silu(float value) {
  return value / (1.0f + std::exp(-value));
}

}  // namespace

namespace llm_accel {

KernelStatus qwen_prefill_mlp_kernel(
    const scalar_t* attention_residual,
    int seq_len,
    int tile_m,
    const scalar_t* post_attention_layernorm_weight,
    scalar_t rms_eps,
    const packed_w4_t* gate_packed_weights,
    const packed_w4_t* up_packed_weights,
    const packed_w4_t* down_packed_weights,
    const scalar_t* gate_scales,
    const scalar_t* up_scales,
    const scalar_t* down_scales,
    scalar_t* output_sequence) {
  if (attention_residual == nullptr || output_sequence == nullptr || seq_len <= 0 || tile_m <= 0 ||
      post_attention_layernorm_weight == nullptr || gate_packed_weights == nullptr || up_packed_weights == nullptr ||
      down_packed_weights == nullptr || gate_scales == nullptr || up_scales == nullptr || down_scales == nullptr) {
    return {false, 1};
  }

  for (int token_index = 0; token_index < seq_len; ++token_index) {
    const scalar_t* attention_token = attention_residual + static_cast<std::size_t>(token_index) * kHiddenSize;
    scalar_t* output_token = output_sequence + static_cast<std::size_t>(token_index) * kHiddenSize;

    std::array<scalar_t, kHiddenSize> post_norm{};
    std::array<scalar_t, kIntermediateSize> gate_proj{};
    std::array<scalar_t, kIntermediateSize> up_proj{};
    std::array<scalar_t, kIntermediateSize> silu_mul{};
    std::array<scalar_t, kHiddenSize> down_proj{};

    rmsnorm_token(attention_token, post_attention_layernorm_weight, rms_eps, post_norm.data());
    project_tiled(post_norm.data(), gate_packed_weights, gate_scales, kHiddenSize, kIntermediateSize, gate_proj.data());
    project_tiled(post_norm.data(), up_packed_weights, up_scales, kHiddenSize, kIntermediateSize, up_proj.data());
    for (int index = 0; index < kIntermediateSize; ++index) {
      silu_mul[index] = silu(gate_proj[index]) * up_proj[index];
    }
    project_tiled(silu_mul.data(), down_packed_weights, down_scales, kIntermediateSize, kHiddenSize, down_proj.data());
    for (int index = 0; index < kHiddenSize; ++index) {
      output_token[index] = attention_token[index] + down_proj[index];
    }
  }

  return {true, 0};
}

}  // namespace llm_accel