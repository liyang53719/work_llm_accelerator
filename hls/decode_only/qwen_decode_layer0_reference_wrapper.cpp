#include "qwen_decode_host_wrapper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

#include "../common/qwen2_model_config.h"

namespace {

float round_to_bfloat16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7FFFU + lsb;
  bits &= 0xFFFF0000U;
  float rounded = 0.0f;
  std::memcpy(&rounded, &bits, sizeof(rounded));
  return rounded;
}

void rmsnorm_token(const float* input, const float* weight, float rms_eps, float* output) {
  double mean_square = 0.0;
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    mean_square += static_cast<double>(input[dim]) * static_cast<double>(input[dim]);
  }
  mean_square /= static_cast<double>(llm_accel::kHiddenSize);
  const double inv_rms = 1.0 / std::sqrt(mean_square + static_cast<double>(rms_eps));
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    output[dim] = round_to_bfloat16(
        static_cast<float>(static_cast<double>(input[dim]) * inv_rms * static_cast<double>(weight[dim])));
  }
}

void linear_row_major(
    const float* input,
    const float* weight,
    const float* bias,
    int in_dim,
    int out_dim,
    float* output) {
  for (int out_index = 0; out_index < out_dim; ++out_index) {
    double sum = bias != nullptr ? static_cast<double>(bias[out_index]) : 0.0;
    const float* weight_row = weight + static_cast<std::size_t>(out_index) * in_dim;
    for (int in_index = 0; in_index < in_dim; ++in_index) {
      sum += static_cast<double>(input[in_index]) * static_cast<double>(weight_row[in_index]);
    }
    output[out_index] = round_to_bfloat16(static_cast<float>(sum));
  }
}

float silu(float value) {
  return value / (1.0f + std::exp(-value));
}

void apply_rope_inplace(float* head, int token_index, const std::vector<double>& inv_freq) {
  for (int pair = 0; pair < llm_accel::kHeadDim / 2; ++pair) {
    const double angle = static_cast<double>(token_index) * inv_freq[pair];
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

}  // namespace

extern "C" int qwen_decode_layer0_reference_forward(
    const float* input_token,
    int past_seq_len,
    const float* past_k_cache,
    const float* past_v_cache,
    const float* input_layernorm_weight,
    const float* q_weight,
    const float* q_bias,
    const float* k_weight,
    const float* k_bias,
    const float* v_weight,
    const float* v_bias,
    const float* o_weight,
    const float* o_bias,
    const float* post_attention_layernorm_weight,
    const float* gate_weight,
    const float* gate_bias,
    const float* up_weight,
    const float* up_bias,
    const float* down_weight,
    const float* down_bias,
    float rms_eps,
    float* output_token,
    float* next_k_cache,
    float* next_v_cache) {
  if (input_token == nullptr || output_token == nullptr || next_k_cache == nullptr || next_v_cache == nullptr || past_seq_len < 0) {
    return 1;
  }

  const int hidden_size = llm_accel::kHiddenSize;
  const int kv_width = llm_accel::kNumKeyValueHeads * llm_accel::kHeadDim;
  const int num_groups = llm_accel::kNumAttentionHeads / llm_accel::kNumKeyValueHeads;
  const int total_seq_len = past_seq_len + 1;
  const float scaling = static_cast<float>(1.0 / std::sqrt(static_cast<double>(llm_accel::kHeadDim)));

  std::vector<double> inv_freq(llm_accel::kHeadDim / 2, 0.0);
  for (int index = 0; index < llm_accel::kHeadDim / 2; ++index) {
    inv_freq[index] = std::pow(1000000.0, -2.0 * static_cast<double>(index) / static_cast<double>(llm_accel::kHeadDim));
  }

  std::vector<float> input_norm(hidden_size, 0.0f);
  std::vector<float> q_proj_out(hidden_size, 0.0f);
  std::vector<float> k_proj_out(kv_width, 0.0f);
  std::vector<float> v_proj_out(kv_width, 0.0f);
  std::vector<float> attn_context(hidden_size, 0.0f);
  std::vector<float> o_proj_out(hidden_size, 0.0f);
  std::vector<float> attention_residual(hidden_size, 0.0f);
  std::vector<float> post_norm(hidden_size, 0.0f);
  std::vector<float> gate_proj_out(llm_accel::kIntermediateSize, 0.0f);
  std::vector<float> up_proj_out(llm_accel::kIntermediateSize, 0.0f);
  std::vector<float> silu_mul(llm_accel::kIntermediateSize, 0.0f);
  std::vector<float> down_proj_out(hidden_size, 0.0f);

  rmsnorm_token(input_token, input_layernorm_weight, rms_eps, input_norm.data());
  linear_row_major(input_norm.data(), q_weight, q_bias, hidden_size, hidden_size, q_proj_out.data());
  linear_row_major(input_norm.data(), k_weight, k_bias, hidden_size, kv_width, k_proj_out.data());
  linear_row_major(input_norm.data(), v_weight, v_bias, hidden_size, kv_width, v_proj_out.data());

  for (int head = 0; head < llm_accel::kNumAttentionHeads; ++head) {
    apply_rope_inplace(q_proj_out.data() + head * llm_accel::kHeadDim, past_seq_len, inv_freq);
  }
  for (int kv_head = 0; kv_head < llm_accel::kNumKeyValueHeads; ++kv_head) {
    apply_rope_inplace(k_proj_out.data() + kv_head * llm_accel::kHeadDim, past_seq_len, inv_freq);
  }

  const std::size_t cache_stride = static_cast<std::size_t>(total_seq_len) * llm_accel::kHeadDim;
  for (int kv_head = 0; kv_head < llm_accel::kNumKeyValueHeads; ++kv_head) {
    for (int token = 0; token < past_seq_len; ++token) {
      for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
        next_k_cache[static_cast<std::size_t>(kv_head) * cache_stride + static_cast<std::size_t>(token) * llm_accel::kHeadDim + dim] =
            past_k_cache[static_cast<std::size_t>(kv_head) * past_seq_len * llm_accel::kHeadDim + static_cast<std::size_t>(token) * llm_accel::kHeadDim + dim];
        next_v_cache[static_cast<std::size_t>(kv_head) * cache_stride + static_cast<std::size_t>(token) * llm_accel::kHeadDim + dim] =
            past_v_cache[static_cast<std::size_t>(kv_head) * past_seq_len * llm_accel::kHeadDim + static_cast<std::size_t>(token) * llm_accel::kHeadDim + dim];
      }
    }
    for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
      next_k_cache[static_cast<std::size_t>(kv_head) * cache_stride + static_cast<std::size_t>(past_seq_len) * llm_accel::kHeadDim + dim] =
          k_proj_out[kv_head * llm_accel::kHeadDim + dim];
      next_v_cache[static_cast<std::size_t>(kv_head) * cache_stride + static_cast<std::size_t>(past_seq_len) * llm_accel::kHeadDim + dim] =
          v_proj_out[kv_head * llm_accel::kHeadDim + dim];
    }
  }

  for (int head = 0; head < llm_accel::kNumAttentionHeads; ++head) {
    const int kv_head = head / num_groups;
    const float* q_head = q_proj_out.data() + head * llm_accel::kHeadDim;
    std::vector<double> scores(total_seq_len, 0.0);
    double max_score = -1.0e300;

    for (int token = 0; token < total_seq_len; ++token) {
      const float* k_head = next_k_cache + static_cast<std::size_t>(kv_head) * cache_stride + static_cast<std::size_t>(token) * llm_accel::kHeadDim;
      double dot = 0.0;
      for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
        dot += static_cast<double>(q_head[dim]) * static_cast<double>(k_head[dim]);
      }
      scores[token] = dot * scaling;
      max_score = std::max(max_score, scores[token]);
    }

    std::vector<double> probs(total_seq_len, 0.0);
    double sum_exp = 0.0;
    for (int token = 0; token < total_seq_len; ++token) {
      probs[token] = std::exp(scores[token] - max_score);
      sum_exp += probs[token];
    }

    float* context_head = attn_context.data() + head * llm_accel::kHeadDim;
    for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
      double accum = 0.0;
      for (int token = 0; token < total_seq_len; ++token) {
        const float* v_head = next_v_cache + static_cast<std::size_t>(kv_head) * cache_stride + static_cast<std::size_t>(token) * llm_accel::kHeadDim;
        accum += (probs[token] / sum_exp) * static_cast<double>(v_head[dim]);
      }
      context_head[dim] = static_cast<float>(accum);
    }
  }

  linear_row_major(attn_context.data(), o_weight, o_bias, hidden_size, hidden_size, o_proj_out.data());
  for (int dim = 0; dim < hidden_size; ++dim) {
    attention_residual[dim] = input_token[dim] + o_proj_out[dim];
  }

  rmsnorm_token(attention_residual.data(), post_attention_layernorm_weight, rms_eps, post_norm.data());
  linear_row_major(post_norm.data(), gate_weight, gate_bias, hidden_size, llm_accel::kIntermediateSize, gate_proj_out.data());
  linear_row_major(post_norm.data(), up_weight, up_bias, hidden_size, llm_accel::kIntermediateSize, up_proj_out.data());
  for (int dim = 0; dim < llm_accel::kIntermediateSize; ++dim) {
    silu_mul[dim] = silu(gate_proj_out[dim]) * up_proj_out[dim];
  }
  linear_row_major(silu_mul.data(), down_weight, down_bias, llm_accel::kIntermediateSize, hidden_size, down_proj_out.data());
  for (int dim = 0; dim < hidden_size; ++dim) {
    output_token[dim] = attention_residual[dim] + down_proj_out[dim];
  }

  return 0;
}