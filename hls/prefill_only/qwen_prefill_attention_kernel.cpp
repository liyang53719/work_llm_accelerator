#include "qwen_prefill_attention_kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

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

void project_tiled_token(
    const llm_accel::scalar_t* input_token,
    const llm_accel::packed_w4_t* packed_weights,
    const llm_accel::scalar_t* bias,
    const llm_accel::scalar_t* scales,
    int out_dim,
    int in_dim,
    int out_tile,
    int in_tile,
    llm_accel::scalar_t* output) {
  std::vector<llm_accel::scalar_t> input_tile(static_cast<std::size_t>(in_tile), 0.0f);
  std::vector<llm_accel::scalar_t> partial_sum(static_cast<std::size_t>(out_tile), 0.0f);

  for (int out_base = 0; out_base < out_dim; out_base += out_tile) {
    const int out_extent = std::min(out_tile, out_dim - out_base);
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      partial_sum[out_offset] = bias == nullptr ? 0.0f : bias[out_base + out_offset];
    }

    for (int in_base = 0; in_base < in_dim; in_base += in_tile) {
      const int in_extent = std::min(in_tile, in_dim - in_base);
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

void prefill_attention_context_block(
    const llm_accel::scalar_t* q_proj,
    const llm_accel::scalar_t* k_proj,
    const llm_accel::scalar_t* v_proj,
    int seq_len,
    int query_begin,
    int query_end,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    llm_accel::scalar_t* context) {
  const float scaling = static_cast<float>(1.0 / std::sqrt(static_cast<double>(llm_accel::kHeadDim)));
  const int key_tile = std::max(1, tile_config.key);
  const int query_heads_parallel = std::max(1, tile_config.query_heads_parallel);

  for (int query_index = query_begin; query_index < query_end; ++query_index) {
    const llm_accel::scalar_t* q_token = q_proj + static_cast<std::size_t>(query_index) * llm_accel::kHiddenSize;
    llm_accel::scalar_t* context_token = context + static_cast<std::size_t>(query_index) * llm_accel::kHiddenSize;

    for (int head_base = 0; head_base < llm_accel::kNumAttentionHeads; head_base += query_heads_parallel) {
      const int head_end = std::min(llm_accel::kNumAttentionHeads, head_base + query_heads_parallel);
      std::vector<float> max_score(static_cast<std::size_t>(head_end - head_base), -1.0e30f);
      std::vector<float> denom(static_cast<std::size_t>(head_end - head_base), 0.0f);
      std::vector<std::vector<float>> accum(
          static_cast<std::size_t>(head_end - head_base),
          std::vector<float>(llm_accel::kHeadDim, 0.0f));

      for (int key_begin = 0; key_begin <= query_index && key_begin < seq_len; key_begin += key_tile) {
        const int key_end = std::min(seq_len, std::min(query_index + 1, key_begin + key_tile));
        for (int head = head_base; head < head_end; ++head) {
          const int head_offset = head - head_base;
          const int kv_head = head / kNumGroups;
          const llm_accel::scalar_t* q_head = q_token + head * llm_accel::kHeadDim;
          for (int key_index = key_begin; key_index < key_end; ++key_index) {
            const llm_accel::scalar_t* k_head =
                k_proj + static_cast<std::size_t>(key_index) * kKvWidth + kv_head * llm_accel::kHeadDim;
            float score = 0.0f;
            for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
              score += q_head[dim] * k_head[dim];
            }
            max_score[head_offset] = std::max(max_score[head_offset], score * scaling);
          }
        }
      }

      for (int key_begin = 0; key_begin <= query_index && key_begin < seq_len; key_begin += key_tile) {
        const int key_end = std::min(seq_len, std::min(query_index + 1, key_begin + key_tile));
        for (int head = head_base; head < head_end; ++head) {
          const int head_offset = head - head_base;
          const int kv_head = head / kNumGroups;
          const llm_accel::scalar_t* q_head = q_token + head * llm_accel::kHeadDim;
          for (int key_index = key_begin; key_index < key_end; ++key_index) {
            const llm_accel::scalar_t* k_head =
                k_proj + static_cast<std::size_t>(key_index) * kKvWidth + kv_head * llm_accel::kHeadDim;
            const llm_accel::scalar_t* v_head =
                v_proj + static_cast<std::size_t>(key_index) * kKvWidth + kv_head * llm_accel::kHeadDim;
            float score = 0.0f;
            for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
              score += q_head[dim] * k_head[dim];
            }
            const float exp_score = std::exp(score * scaling - max_score[head_offset]);
            denom[head_offset] += exp_score;
            for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
              accum[head_offset][dim] += exp_score * v_head[dim];
            }
          }
        }
      }

      for (int head = head_base; head < head_end; ++head) {
        const int head_offset = head - head_base;
        llm_accel::scalar_t* context_head = context_token + head * llm_accel::kHeadDim;
        const float inv_denom = denom[head_offset] > 0.0f ? 1.0f / denom[head_offset] : 0.0f;
        for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
          context_head[dim] = accum[head_offset][dim] * inv_denom;
        }
      }
    }
  }
}

}  // namespace

namespace llm_accel {

KernelStatus qwen_prefill_attention_kernel(
    const scalar_t* input_sequence,
    int seq_len,
  const PrefillAttentionTileConfig& tile_config,
    const scalar_t* input_layernorm_weight,
    scalar_t rms_eps,
    const packed_w4_t* q_packed_weights,
    const packed_w4_t* k_packed_weights,
    const packed_w4_t* v_packed_weights,
    const packed_w4_t* o_packed_weights,
    const scalar_t* q_bias,
    const scalar_t* k_bias,
    const scalar_t* v_bias,
    const scalar_t* q_scales,
    const scalar_t* k_scales,
    const scalar_t* v_scales,
    const scalar_t* o_scales,
    scalar_t* k_cache,
    scalar_t* v_cache,
    scalar_t* output_sequence) {
  if (input_sequence == nullptr || output_sequence == nullptr || seq_len <= 0 ||
      input_layernorm_weight == nullptr || q_packed_weights == nullptr || k_packed_weights == nullptr ||
      v_packed_weights == nullptr || o_packed_weights == nullptr || q_bias == nullptr || k_bias == nullptr ||
      v_bias == nullptr || q_scales == nullptr || k_scales == nullptr ||
      v_scales == nullptr || o_scales == nullptr || k_cache == nullptr || v_cache == nullptr ||
      tile_config.seq <= 0 || tile_config.query <= 0 || tile_config.key <= 0 || tile_config.hidden_proj <= 0 ||
      tile_config.kv_proj <= 0 || tile_config.head_dim != kHeadDim || tile_config.query_heads_parallel <= 0 ||
      tile_config.kv_heads_parallel <= 0) {
    return {false, 1};
  }

  const int seq_tile = std::max(1, tile_config.seq);
  const int query_tile = std::max(1, tile_config.query);

  std::vector<scalar_t> input_norm(static_cast<std::size_t>(seq_len) * kHiddenSize, 0.0f);
  std::vector<scalar_t> q_proj(static_cast<std::size_t>(seq_len) * kHiddenSize, 0.0f);
  std::vector<scalar_t> k_proj(static_cast<std::size_t>(seq_len) * kKvWidth, 0.0f);
  std::vector<scalar_t> v_proj(static_cast<std::size_t>(seq_len) * kKvWidth, 0.0f);
  std::vector<scalar_t> context(static_cast<std::size_t>(seq_len) * kHiddenSize, 0.0f);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = std::min(seq_len, token_begin + seq_tile);
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      const scalar_t* input_token = input_sequence + static_cast<std::size_t>(token_index) * kHiddenSize;
      scalar_t* input_norm_token = input_norm.data() + static_cast<std::size_t>(token_index) * kHiddenSize;
      scalar_t* q_proj_token = q_proj.data() + static_cast<std::size_t>(token_index) * kHiddenSize;
      scalar_t* k_proj_token = k_proj.data() + static_cast<std::size_t>(token_index) * kKvWidth;
      scalar_t* v_proj_token = v_proj.data() + static_cast<std::size_t>(token_index) * kKvWidth;

      rmsnorm_token(input_token, input_layernorm_weight, rms_eps, input_norm_token);
      project_tiled_token(
          input_norm_token,
          q_packed_weights,
          q_bias,
          q_scales,
          kHiddenSize,
          kHiddenSize,
          tile_config.hidden_proj,
          tile_config.hidden_proj,
          q_proj_token);
      project_tiled_token(
          input_norm_token,
          k_packed_weights,
          k_bias,
          k_scales,
          kKvWidth,
          kHiddenSize,
          tile_config.kv_proj,
          tile_config.hidden_proj,
          k_proj_token);
      project_tiled_token(
          input_norm_token,
          v_packed_weights,
          v_bias,
          v_scales,
          kKvWidth,
          kHiddenSize,
          tile_config.kv_proj,
          tile_config.hidden_proj,
          v_proj_token);

      for (int head_base = 0; head_base < kNumAttentionHeads; head_base += tile_config.query_heads_parallel) {
        const int head_end = std::min(kNumAttentionHeads, head_base + tile_config.query_heads_parallel);
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace(q_proj_token + head * kHeadDim, token_index);
        }
      }
      for (int head_base = 0; head_base < kNumKeyValueHeads; head_base += tile_config.kv_heads_parallel) {
        const int head_end = std::min(kNumKeyValueHeads, head_base + tile_config.kv_heads_parallel);
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace(k_proj_token + head * kHeadDim, token_index);
        }
      }

      for (int index = 0; index < kKvWidth; ++index) {
        k_cache[static_cast<std::size_t>(token_index) * kKvWidth + index] = k_proj_token[index];
        v_cache[static_cast<std::size_t>(token_index) * kKvWidth + index] = v_proj_token[index];
      }
    }
  }

  for (int query_begin = 0; query_begin < seq_len; query_begin += query_tile) {
    const int query_end = std::min(seq_len, query_begin + query_tile);
    prefill_attention_context_block(
        q_proj.data(),
        k_proj.data(),
        v_proj.data(),
        seq_len,
        query_begin,
        query_end,
        tile_config,
        context.data());
  }

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = std::min(seq_len, token_begin + seq_tile);
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      const scalar_t* context_token = context.data() + static_cast<std::size_t>(token_index) * kHiddenSize;
      scalar_t* output_token = output_sequence + static_cast<std::size_t>(token_index) * kHiddenSize;
      project_tiled_token(
          context_token,
          o_packed_weights,
          nullptr,
          o_scales,
          kHiddenSize,
          kHiddenSize,
          tile_config.hidden_proj,
          tile_config.hidden_proj,
          output_token);
    }
  }

  return {true, 0};
}

}  // namespace llm_accel