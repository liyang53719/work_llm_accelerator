#include "qwen_prefill_host_wrapper.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
  float mean_square = 0.0f;
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    mean_square += input[dim] * input[dim];
  }
  mean_square /= static_cast<float>(llm_accel::kHiddenSize);
  const float inv_rms = 1.0f / std::sqrt(mean_square + rms_eps);
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    const float normalized = input[dim] * inv_rms;
    const float normalized_bf16 = round_to_bfloat16(normalized);
    output[dim] = round_to_bfloat16(normalized_bf16 * weight[dim]);
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

int qwen_prefill_layer0_reference_forward_impl(
    const float* input_sequence,
    int seq_len,
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
    float* output_sequence,
    float* k_cache,
    float* v_cache) {
  if (input_sequence == nullptr || output_sequence == nullptr || seq_len <= 0 || seq_len > llm_accel::kMaxSequenceLength) {
    return 1;
  }

  if ((k_cache == nullptr) != (v_cache == nullptr)) {
    return 1;
  }

  const int hidden_size = llm_accel::kHiddenSize;
  const int kv_width = llm_accel::kNumKeyValueHeads * llm_accel::kHeadDim;
  const int num_groups = llm_accel::kNumAttentionHeads / llm_accel::kNumKeyValueHeads;
  const float scaling = static_cast<float>(1.0 / std::sqrt(static_cast<double>(llm_accel::kHeadDim)));
  const std::size_t seq_hidden = static_cast<std::size_t>(seq_len) * hidden_size;
  const std::size_t seq_kv = static_cast<std::size_t>(seq_len) * kv_width;
  const std::size_t seq_mlp = static_cast<std::size_t>(seq_len) * llm_accel::kIntermediateSize;

  std::vector<float> input_norm(seq_hidden, 0.0f);
  std::vector<float> q_proj_out(seq_hidden, 0.0f);
  std::vector<float> k_proj_out(seq_kv, 0.0f);
  std::vector<float> v_proj_out(seq_kv, 0.0f);
  std::vector<float> attn_context(seq_hidden, 0.0f);
  std::vector<float> o_proj_out(seq_hidden, 0.0f);
  std::vector<float> attention_residual(seq_hidden, 0.0f);
  std::vector<float> post_norm(seq_hidden, 0.0f);
  std::vector<float> gate_proj_out(seq_mlp, 0.0f);
  std::vector<float> up_proj_out(seq_mlp, 0.0f);
  std::vector<float> silu_mul(seq_mlp, 0.0f);
  std::vector<float> down_proj_out(seq_hidden, 0.0f);
  std::vector<double> inv_freq(llm_accel::kHeadDim / 2, 0.0);

  for (int index = 0; index < llm_accel::kHeadDim / 2; ++index) {
    inv_freq[index] = std::pow(1000000.0, -2.0 * static_cast<double>(index) / static_cast<double>(llm_accel::kHeadDim));
  }

  for (int token = 0; token < seq_len; ++token) {
    const float* input_token = input_sequence + static_cast<std::size_t>(token) * hidden_size;
    float* norm_token = input_norm.data() + static_cast<std::size_t>(token) * hidden_size;
    rmsnorm_token(input_token, input_layernorm_weight, rms_eps, norm_token);

    linear_row_major(norm_token, q_weight, q_bias, hidden_size, hidden_size, q_proj_out.data() + static_cast<std::size_t>(token) * hidden_size);
    linear_row_major(norm_token, k_weight, k_bias, hidden_size, kv_width, k_proj_out.data() + static_cast<std::size_t>(token) * kv_width);
    linear_row_major(norm_token, v_weight, v_bias, hidden_size, kv_width, v_proj_out.data() + static_cast<std::size_t>(token) * kv_width);
  }

  for (int token = 0; token < seq_len; ++token) {
    for (int head = 0; head < llm_accel::kNumAttentionHeads; ++head) {
      float* q_head = q_proj_out.data() + static_cast<std::size_t>(token) * hidden_size + head * llm_accel::kHeadDim;
      for (int pair = 0; pair < llm_accel::kHeadDim / 2; ++pair) {
        const double angle = static_cast<double>(token) * inv_freq[pair];
        const float cosv = round_to_bfloat16(static_cast<float>(std::cos(angle)));
        const float sinv = round_to_bfloat16(static_cast<float>(std::sin(angle)));
        const int even_index = pair;
        const int odd_index = pair + llm_accel::kHeadDim / 2;
        const float even = q_head[even_index];
        const float odd = q_head[odd_index];
        const float even_cos = round_to_bfloat16(even * cosv);
        const float odd_sin = round_to_bfloat16(odd * sinv);
        const float odd_cos = round_to_bfloat16(odd * cosv);
        const float even_sin = round_to_bfloat16(even * sinv);
        q_head[even_index] = round_to_bfloat16(even_cos - odd_sin);
        q_head[odd_index] = round_to_bfloat16(odd_cos + even_sin);
      }
    }

    for (int kv_head = 0; kv_head < llm_accel::kNumKeyValueHeads; ++kv_head) {
      float* k_head = k_proj_out.data() + static_cast<std::size_t>(token) * kv_width + kv_head * llm_accel::kHeadDim;
      for (int pair = 0; pair < llm_accel::kHeadDim / 2; ++pair) {
        const double angle = static_cast<double>(token) * inv_freq[pair];
        const float cosv = round_to_bfloat16(static_cast<float>(std::cos(angle)));
        const float sinv = round_to_bfloat16(static_cast<float>(std::sin(angle)));
        const int even_index = pair;
        const int odd_index = pair + llm_accel::kHeadDim / 2;
        const float even = k_head[even_index];
        const float odd = k_head[odd_index];
        const float even_cos = round_to_bfloat16(even * cosv);
        const float odd_sin = round_to_bfloat16(odd * sinv);
        const float odd_cos = round_to_bfloat16(odd * cosv);
        const float even_sin = round_to_bfloat16(even * sinv);
        k_head[even_index] = round_to_bfloat16(even_cos - odd_sin);
        k_head[odd_index] = round_to_bfloat16(odd_cos + even_sin);
      }
    }
  }

  if (k_cache != nullptr && v_cache != nullptr) {
    const std::size_t cache_stride = static_cast<std::size_t>(seq_len) * llm_accel::kHeadDim;
    for (int token = 0; token < seq_len; ++token) {
      const float* token_k = k_proj_out.data() + static_cast<std::size_t>(token) * kv_width;
      const float* token_v = v_proj_out.data() + static_cast<std::size_t>(token) * kv_width;
      for (int kv_head = 0; kv_head < llm_accel::kNumKeyValueHeads; ++kv_head) {
        const float* src_k = token_k + kv_head * llm_accel::kHeadDim;
        const float* src_v = token_v + kv_head * llm_accel::kHeadDim;
        float* dst_k = k_cache + static_cast<std::size_t>(kv_head) * cache_stride + static_cast<std::size_t>(token) * llm_accel::kHeadDim;
        float* dst_v = v_cache + static_cast<std::size_t>(kv_head) * cache_stride + static_cast<std::size_t>(token) * llm_accel::kHeadDim;
        for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
          dst_k[dim] = round_to_bfloat16(src_k[dim]);
          dst_v[dim] = round_to_bfloat16(src_v[dim]);
        }
      }
    }
  }

  for (int token = 0; token < seq_len; ++token) {
    for (int head = 0; head < llm_accel::kNumAttentionHeads; ++head) {
      const int kv_head = head / num_groups;
      const float* q_head = q_proj_out.data() + static_cast<std::size_t>(token) * hidden_size + head * llm_accel::kHeadDim;
      std::vector<double> scores(token + 1, 0.0);
      double max_score = -1.0e300;

      for (int src = 0; src <= token; ++src) {
        const float* k_head = k_proj_out.data() + static_cast<std::size_t>(src) * kv_width + kv_head * llm_accel::kHeadDim;
        double dot = 0.0;
        for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
          dot += static_cast<double>(q_head[dim]) * static_cast<double>(k_head[dim]);
        }
        scores[src] = dot * scaling;
        max_score = std::max(max_score, scores[src]);
      }

      std::vector<double> probs(token + 1, 0.0);
      double sum_exp = 0.0;
      for (int src = 0; src <= token; ++src) {
        probs[src] = std::exp(scores[src] - max_score);
        sum_exp += probs[src];
      }

      float* context_head = attn_context.data() + static_cast<std::size_t>(token) * hidden_size + head * llm_accel::kHeadDim;
      for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
        double accum = 0.0;
        for (int src = 0; src <= token; ++src) {
          const float* v_head = v_proj_out.data() + static_cast<std::size_t>(src) * kv_width + kv_head * llm_accel::kHeadDim;
          accum += (probs[src] / sum_exp) * static_cast<double>(v_head[dim]);
        }
        context_head[dim] = static_cast<float>(accum);
      }
    }
  }

  for (int token = 0; token < seq_len; ++token) {
    const float* context_token = attn_context.data() + static_cast<std::size_t>(token) * hidden_size;
    float* o_proj_token = o_proj_out.data() + static_cast<std::size_t>(token) * hidden_size;
    linear_row_major(context_token, o_weight, o_bias, hidden_size, hidden_size, o_proj_token);

    const float* input_token = input_sequence + static_cast<std::size_t>(token) * hidden_size;
    float* residual_token = attention_residual.data() + static_cast<std::size_t>(token) * hidden_size;
    for (int dim = 0; dim < hidden_size; ++dim) {
      residual_token[dim] = input_token[dim] + o_proj_token[dim];
    }

    float* post_norm_token = post_norm.data() + static_cast<std::size_t>(token) * hidden_size;
    rmsnorm_token(residual_token, post_attention_layernorm_weight, rms_eps, post_norm_token);

    float* gate_token = gate_proj_out.data() + static_cast<std::size_t>(token) * llm_accel::kIntermediateSize;
    float* up_token = up_proj_out.data() + static_cast<std::size_t>(token) * llm_accel::kIntermediateSize;
    float* silu_token = silu_mul.data() + static_cast<std::size_t>(token) * llm_accel::kIntermediateSize;
    linear_row_major(post_norm_token, gate_weight, gate_bias, hidden_size, llm_accel::kIntermediateSize, gate_token);
    linear_row_major(post_norm_token, up_weight, up_bias, hidden_size, llm_accel::kIntermediateSize, up_token);
    for (int dim = 0; dim < llm_accel::kIntermediateSize; ++dim) {
      silu_token[dim] = silu(gate_token[dim]) * up_token[dim];
    }

    float* down_token = down_proj_out.data() + static_cast<std::size_t>(token) * hidden_size;
    linear_row_major(silu_token, down_weight, down_bias, llm_accel::kIntermediateSize, hidden_size, down_token);
    for (int dim = 0; dim < hidden_size; ++dim) {
      output_sequence[static_cast<std::size_t>(token) * hidden_size + dim] = residual_token[dim] + down_token[dim];
    }
  }

  return 0;
}

}  // namespace

extern "C" int qwen_prefill_layer0_reference_forward(
    const float* input_sequence,
    int seq_len,
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
    float* output_sequence) {
  return qwen_prefill_layer0_reference_forward_impl(
      input_sequence,
      seq_len,
      input_layernorm_weight,
      q_weight,
      q_bias,
      k_weight,
      k_bias,
      v_weight,
      v_bias,
      o_weight,
      o_bias,
      post_attention_layernorm_weight,
      gate_weight,
      gate_bias,
      up_weight,
      up_bias,
      down_weight,
      down_bias,
      rms_eps,
      output_sequence,
      nullptr,
      nullptr);
}

extern "C" int qwen_prefill_layer0_reference_forward_with_cache(
    const float* input_sequence,
    int seq_len,
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
    float* output_sequence,
    float* k_cache,
    float* v_cache) {
  return qwen_prefill_layer0_reference_forward_impl(
      input_sequence,
      seq_len,
      input_layernorm_weight,
      q_weight,
      q_bias,
      k_weight,
      k_bias,
      v_weight,
      v_bias,
      o_weight,
      o_bias,
      post_attention_layernorm_weight,
      gate_weight,
      gate_bias,
      up_weight,
      up_bias,
      down_weight,
      down_bias,
      rms_eps,
      output_sequence,
      k_cache,
      v_cache);
}