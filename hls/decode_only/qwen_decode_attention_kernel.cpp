#include "qwen_decode_attention_kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace {

constexpr int kKvWidth = llm_accel::kNumKeyValueHeads * llm_accel::kHeadDim;
constexpr int kNumGroups = llm_accel::kNumAttentionHeads / llm_accel::kNumKeyValueHeads;

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

void apply_rope_inplace(llm_accel::scalar_t* head, int token_index) {
  for (int pair = 0; pair < llm_accel::kHeadDim / 2; ++pair) {
    const double angle = static_cast<double>(token_index) *
        std::pow(static_cast<double>(llm_accel::kRopeTheta), -2.0 * static_cast<double>(pair) / static_cast<double>(llm_accel::kHeadDim));
    const float cosv = static_cast<float>(std::cos(angle));
    const float sinv = static_cast<float>(std::sin(angle));
    const int even_index = pair;
    const int odd_index = pair + llm_accel::kHeadDim / 2;
    const float even = head[even_index];
    const float odd = head[odd_index];
    head[even_index] = even * cosv - odd * sinv;
    head[odd_index] = odd * cosv + even * sinv;
  }
}

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

void project_tiled(
    const llm_accel::scalar_t* input_token,
    const llm_accel::packed_w4_t* packed_weights,
    const llm_accel::scalar_t* scales,
    int out_dim,
    llm_accel::scalar_t* output) {
  std::array<llm_accel::scalar_t, llm_accel::kTileN> input_tile{};
  std::array<llm_accel::scalar_t, llm_accel::kTileN> partial_sum{};

  for (int out_base = 0; out_base < out_dim; out_base += llm_accel::kTileN) {
    const int out_extent = std::min(llm_accel::kTileN, out_dim - out_base);
    partial_sum.fill(0.0f);

    for (int in_base = 0; in_base < llm_accel::kHiddenSize; in_base += llm_accel::kTileN) {
      const int in_extent = std::min(llm_accel::kTileN, llm_accel::kHiddenSize - in_base);
      for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
        input_tile[in_offset] = input_token[in_base + in_offset];
      }

      for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
        const int out_index = out_base + out_offset;
        float accum = partial_sum[out_offset];
        for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
          accum += input_tile[in_offset] * dequantized_weight(packed_weights, scales, out_index, in_base + in_offset, llm_accel::kHiddenSize);
        }
        partial_sum[out_offset] = accum;
      }
    }

    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      output[out_base + out_offset] = partial_sum[out_offset];
    }
  }
}

void append_kv_cache(
    const llm_accel::scalar_t* k_proj,
    const llm_accel::scalar_t* v_proj,
    int past_seq_len,
    llm_accel::scalar_t* k_cache,
    llm_accel::scalar_t* v_cache) {
  const std::size_t write_offset = static_cast<std::size_t>(past_seq_len) * kKvWidth;
  for (int index = 0; index < kKvWidth; ++index) {
    k_cache[write_offset + index] = k_proj[index];
    v_cache[write_offset + index] = v_proj[index];
  }
}

void decode_attention_context(
    const llm_accel::scalar_t* q_proj,
    const llm_accel::scalar_t* k_cache,
    const llm_accel::scalar_t* v_cache,
    int total_seq_len,
    llm_accel::scalar_t* context) {
  const float scaling = static_cast<float>(1.0 / std::sqrt(static_cast<double>(llm_accel::kHeadDim)));

  for (int head = 0; head < llm_accel::kNumAttentionHeads; ++head) {
    const int kv_head = head / kNumGroups;
    const llm_accel::scalar_t* q_head = q_proj + head * llm_accel::kHeadDim;

    float max_score = -1.0e30f;
    for (int token = 0; token < total_seq_len; ++token) {
      const llm_accel::scalar_t* k_head =
          k_cache + static_cast<std::size_t>(token) * kKvWidth + kv_head * llm_accel::kHeadDim;
      float score = 0.0f;
      for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
        score += q_head[dim] * k_head[dim];
      }
      max_score = std::max(max_score, score * scaling);
    }

    float denom = 0.0f;
    std::array<float, llm_accel::kHeadDim> accum{};
    accum.fill(0.0f);
    for (int token = 0; token < total_seq_len; ++token) {
      const llm_accel::scalar_t* k_head =
          k_cache + static_cast<std::size_t>(token) * kKvWidth + kv_head * llm_accel::kHeadDim;
      const llm_accel::scalar_t* v_head =
          v_cache + static_cast<std::size_t>(token) * kKvWidth + kv_head * llm_accel::kHeadDim;

      float score = 0.0f;
      for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
        score += q_head[dim] * k_head[dim];
      }
      const float exp_score = std::exp(score * scaling - max_score);
      denom += exp_score;
      for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
        accum[dim] += exp_score * v_head[dim];
      }
    }

    llm_accel::scalar_t* context_head = context + head * llm_accel::kHeadDim;
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
      context_head[dim] = accum[dim] * inv_denom;
    }
  }
}

}  // namespace

namespace llm_accel {

KernelStatus qwen_decode_attention_kernel(
    const scalar_t* input_token,
    int past_seq_len,
  const scalar_t* input_layernorm_weight,
  scalar_t rms_eps,
    const packed_w4_t* q_packed_weights,
    const packed_w4_t* k_packed_weights,
    const packed_w4_t* v_packed_weights,
    const packed_w4_t* o_packed_weights,
    const scalar_t* q_scales,
    const scalar_t* k_scales,
    const scalar_t* v_scales,
    const scalar_t* o_scales,
    scalar_t* k_cache,
    scalar_t* v_cache,
    scalar_t* output_token) {
  if (input_token == nullptr || output_token == nullptr || past_seq_len < 0 || input_layernorm_weight == nullptr ||
      q_packed_weights == nullptr ||
      k_packed_weights == nullptr || v_packed_weights == nullptr || o_packed_weights == nullptr || q_scales == nullptr ||
      k_scales == nullptr || v_scales == nullptr || o_scales == nullptr || k_cache == nullptr || v_cache == nullptr) {
    return {false, 1};
  }

  std::array<scalar_t, kHiddenSize> input_norm{};
  std::array<scalar_t, kHiddenSize> q_proj{};
  std::array<scalar_t, kKvWidth> k_proj{};
  std::array<scalar_t, kKvWidth> v_proj{};
  std::array<scalar_t, kHiddenSize> context{};

  rmsnorm_token(input_token, input_layernorm_weight, rms_eps, input_norm.data());
  project_tiled(input_norm.data(), q_packed_weights, q_scales, kHiddenSize, q_proj.data());
  project_tiled(input_norm.data(), k_packed_weights, k_scales, kKvWidth, k_proj.data());
  project_tiled(input_norm.data(), v_packed_weights, v_scales, kKvWidth, v_proj.data());

  for (int head = 0; head < kNumAttentionHeads; ++head) {
    apply_rope_inplace(q_proj.data() + head * kHeadDim, past_seq_len);
  }
  for (int head = 0; head < kNumKeyValueHeads; ++head) {
    apply_rope_inplace(k_proj.data() + head * kHeadDim, past_seq_len);
  }

  append_kv_cache(k_proj.data(), v_proj.data(), past_seq_len, k_cache, v_cache);
  decode_attention_context(q_proj.data(), k_cache, v_cache, past_seq_len + 1, context.data());
  project_tiled(context.data(), o_packed_weights, o_scales, kHiddenSize, output_token);

  return {true, 0};
}

}  // namespace llm_accel