#include "qwen_prefill_attention_kernel.h"

#ifdef __SYNTHESIS__
#include <ac_int.h>
#include <ac_std_float.h>
#include <ccs_dw_fp_lib.h>
#endif

namespace {

constexpr int kKvWidth = llm_accel::kNumKeyValueHeads * llm_accel::kHeadDim;
constexpr int kNumGroups = llm_accel::kNumAttentionHeads / llm_accel::kNumKeyValueHeads;
constexpr int kPrefillSeqCapacity = llm_accel::kDefaultPrefillSeqTile;
constexpr int kPrefillQueryCapacity = llm_accel::kDefaultPrefillAttentionQueryTile;
constexpr int kPrefillKeyCapacity = llm_accel::kDefaultPrefillAttentionKeyTile;
constexpr int kProjectionTileCapacity =
    llm_accel::kDefaultPrefillAttentionHiddenProjTile > llm_accel::kDefaultPrefillAttentionKvProjTile
        ? llm_accel::kDefaultPrefillAttentionHiddenProjTile
        : llm_accel::kDefaultPrefillAttentionKvProjTile;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kHalfPi = 1.57079632679489661923f;
constexpr float kAttentionScaling = 0.08838834764831845f;
constexpr float kRopeFreqStep = 0.8058421877614801f;

llm_accel::scalar_t g_q_proj_buffer[kPrefillSeqCapacity][llm_accel::kHiddenSize];
llm_accel::scalar_t g_context_buffer[kPrefillQueryCapacity][llm_accel::kHiddenSize];

inline int min_int(int lhs, int rhs) {
  return lhs < rhs ? lhs : rhs;
}

inline int max_int(int lhs, int rhs) {
  return lhs > rhs ? lhs : rhs;
}

float approx_sqrt(float value) {
  if (value <= 0.0f) {
    return 0.0f;
  }

  float guess = value > 1.0f ? value : 1.0f;
  for (int iter = 0; iter < 6; ++iter) {
    guess = 0.5f * (guess + value / guess);
  }
  return guess;
}

float wrap_angle(float angle) {
  while (angle > kPi) {
    angle -= kTwoPi;
  }
  while (angle < -kPi) {
    angle += kTwoPi;
  }
  return angle;
}

void approx_sincos(float angle, float* sin_value, float* cos_value) {
  float reduced = wrap_angle(angle);
  float cos_sign = 1.0f;
  if (reduced > kHalfPi) {
    reduced = kPi - reduced;
    cos_sign = -1.0f;
  } else if (reduced < -kHalfPi) {
    reduced = -kPi - reduced;
    cos_sign = -1.0f;
  }

  const float x2 = reduced * reduced;
  const float sin_poly = reduced * (1.0f + x2 * (-1.0f / 6.0f + x2 * (1.0f / 120.0f + x2 * (-1.0f / 5040.0f))));
  const float cos_poly = 1.0f + x2 * (-0.5f + x2 * (1.0f / 24.0f + x2 * (-1.0f / 720.0f)));
  *sin_value = sin_poly;
  *cos_value = cos_sign * cos_poly;
}

float approx_exp(float value) {
  if (value <= -16.0f) {
    return 0.0f;
  }

  int half_steps = 0;
  float scaled = value;
  while (scaled < -1.0f) {
    scaled *= 0.5f;
    ++half_steps;
  }

  float term = 1.0f;
  float series = 1.0f;
  for (int degree = 1; degree <= 6; ++degree) {
    term *= scaled / static_cast<float>(degree);
    series += term;
  }

  for (int step = 0; step < half_steps; ++step) {
    series *= series;
  }
  return series > 0.0f ? series : 0.0f;
}

void rmsnorm_token(
    const llm_accel::scalar_t* input,
    const llm_accel::scalar_t* weight,
    llm_accel::scalar_t rms_eps,
    llm_accel::scalar_t* output) {
  float mean_square = 0.0f;
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    mean_square += input[dim] * input[dim];
  }
  mean_square /= static_cast<float>(llm_accel::kHiddenSize);
  const float inv_rms = 1.0f / approx_sqrt(mean_square + rms_eps);
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    output[dim] = input[dim] * inv_rms * weight[dim];
  }
}

void apply_rope_inplace(llm_accel::scalar_t* head, int token_index) {
  float inv_freq = 1.0f;
  for (int pair = 0; pair < llm_accel::kHeadDim / 2; ++pair) {
    const float angle = static_cast<float>(token_index) * inv_freq;
    float sinv = 0.0f;
    float cosv = 1.0f;
    approx_sincos(angle, &sinv, &cosv);
    const int even_index = pair;
    const int odd_index = pair + llm_accel::kHeadDim / 2;
    const float even = head[even_index];
    const float odd = head[odd_index];
    head[even_index] = even * cosv - odd * sinv;
    head[odd_index] = odd * cosv + even * sinv;
    inv_freq *= kRopeFreqStep;
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
  const int flat_index = out_index * in_dim + in_index;
  const llm_accel::packed_w4_t packed_value = packed_weights[flat_index / 2];
  const bool high_nibble = (flat_index & 1) != 0;
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
  llm_accel::scalar_t input_tile[kProjectionTileCapacity];
  llm_accel::scalar_t partial_sum[kProjectionTileCapacity];

  for (int out_base = 0; out_base < out_dim; out_base += out_tile) {
    const int out_extent = min_int(out_tile, out_dim - out_base);
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      partial_sum[out_offset] = bias == nullptr ? 0.0f : bias[out_base + out_offset];
    }

    for (int in_base = 0; in_base < in_dim; in_base += in_tile) {
      const int in_extent = min_int(in_tile, in_dim - in_base);
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
    const llm_accel::scalar_t q_proj[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    const llm_accel::scalar_t* k_proj,
    const llm_accel::scalar_t* v_proj,
    int seq_len,
    int query_begin,
    int query_end,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    llm_accel::scalar_t context[kPrefillQueryCapacity][llm_accel::kHiddenSize]) {
  const int key_tile = max_int(1, tile_config.key);
  const int query_heads_parallel = max_int(1, tile_config.query_heads_parallel);

  for (int query_index = query_begin; query_index < query_end; ++query_index) {
    const llm_accel::scalar_t* q_token = q_proj[query_index];
    llm_accel::scalar_t* context_token = context[query_index - query_begin];

    for (int head_base = 0; head_base < llm_accel::kNumAttentionHeads; head_base += query_heads_parallel) {
      const int head_end = min_int(llm_accel::kNumAttentionHeads, head_base + query_heads_parallel);
      float max_score[llm_accel::kNumAttentionHeads];
      float denom[llm_accel::kNumAttentionHeads];
      float accum[llm_accel::kNumAttentionHeads][llm_accel::kHeadDim];

      for (int head_offset = 0; head_offset < head_end - head_base; ++head_offset) {
        max_score[head_offset] = -1.0e30f;
        denom[head_offset] = 0.0f;
        for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
          accum[head_offset][dim] = 0.0f;
        }
      }

      for (int key_begin = 0; key_begin <= query_index && key_begin < seq_len; key_begin += key_tile) {
        const int query_limit = query_index + 1;
        const int key_end = min_int(seq_len, min_int(query_limit, key_begin + key_tile));
        for (int head = head_base; head < head_end; ++head) {
          const int head_offset = head - head_base;
          const int kv_head = head / kNumGroups;
          const llm_accel::scalar_t* q_head = q_token + head * llm_accel::kHeadDim;
          for (int key_index = key_begin; key_index < key_end; ++key_index) {
            const llm_accel::scalar_t* k_head =
                k_proj + key_index * kKvWidth + kv_head * llm_accel::kHeadDim;
            float score = 0.0f;
            for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
              score += q_head[dim] * k_head[dim];
            }
            const float scaled_score = score * kAttentionScaling;
            max_score[head_offset] = scaled_score > max_score[head_offset] ? scaled_score : max_score[head_offset];
          }
        }
      }

      for (int key_begin = 0; key_begin <= query_index && key_begin < seq_len; key_begin += key_tile) {
        const int query_limit = query_index + 1;
        const int key_end = min_int(seq_len, min_int(query_limit, key_begin + key_tile));
        for (int head = head_base; head < head_end; ++head) {
          const int head_offset = head - head_base;
          const int kv_head = head / kNumGroups;
          const llm_accel::scalar_t* q_head = q_token + head * llm_accel::kHeadDim;
          for (int key_index = key_begin; key_index < key_end; ++key_index) {
            const llm_accel::scalar_t* k_head =
                k_proj + key_index * kKvWidth + kv_head * llm_accel::kHeadDim;
            const llm_accel::scalar_t* v_head =
                v_proj + key_index * kKvWidth + kv_head * llm_accel::kHeadDim;
            float score = 0.0f;
            for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
              score += q_head[dim] * k_head[dim];
            }
            const float exp_score = approx_exp(score * kAttentionScaling - max_score[head_offset]);
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

#ifdef __SYNTHESIS__

using catapult_fp_t = ac_std_float<32, 8>;

constexpr int kFpIeeeCompliance = 0;
catapult_fp_t g_q_proj_buffer_fp[kPrefillSeqCapacity][llm_accel::kHiddenSize];
catapult_fp_t g_context_buffer_fp[kPrefillQueryCapacity][llm_accel::kHiddenSize];

inline catapult_fp_t fp_const(float value) {
  return catapult_fp_t(value);
}

inline catapult_fp_t fp_const_int(int value) {
  return catapult_fp_t(value);
}

inline catapult_fp_t fp_zero() {
  return fp_const(0.0f);
}

inline catapult_fp_t fp_one() {
  return fp_const(1.0f);
}

inline catapult_fp_t fp_add_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return fp_add<AC_RND_CONV, kFpIeeeCompliance>(lhs, rhs);
}

inline catapult_fp_t fp_sub_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return fp_sub<AC_RND_CONV, kFpIeeeCompliance>(lhs, rhs);
}

inline catapult_fp_t fp_mul_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return fp_mult<AC_RND_CONV, kFpIeeeCompliance>(lhs, rhs);
}

inline catapult_fp_t fp_div_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return fp_div<AC_RND_CONV, kFpIeeeCompliance>(lhs, rhs);
}

inline catapult_fp_t fp_mac_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs, const catapult_fp_t& acc) {
  return fp_mac<AC_RND_CONV, kFpIeeeCompliance>(lhs, rhs, acc);
}

inline catapult_fp_t fp_sqrt_op(const catapult_fp_t& value) {
  return fp_sqrt<AC_RND_CONV, kFpIeeeCompliance>(value);
}

inline bool fp_eq_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  ac_int<4, false> rel;
  catapult_fp_t z0;
  catapult_fp_t z1;
  fp_cmp<kFpIeeeCompliance>(lhs, rhs, 0, rel, z0, z1);
  return rel[0] != 0;
}

inline bool fp_lt_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  ac_int<4, false> rel;
  catapult_fp_t z0;
  catapult_fp_t z1;
  fp_cmp<kFpIeeeCompliance>(lhs, rhs, 0, rel, z0, z1);
  return rel[1] != 0;
}

inline bool fp_gt_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  ac_int<4, false> rel;
  catapult_fp_t z0;
  catapult_fp_t z1;
  fp_cmp<kFpIeeeCompliance>(lhs, rhs, 0, rel, z0, z1);
  return rel[2] != 0;
}

inline bool fp_le_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return fp_lt_op(lhs, rhs) || fp_eq_op(lhs, rhs);
}

catapult_fp_t approx_exp_fp(const catapult_fp_t& value) {
  const catapult_fp_t neg_sixteen = fp_const(-16.0f);
  const catapult_fp_t neg_one = fp_const(-1.0f);
  const catapult_fp_t half = fp_const(0.5f);

  if (fp_le_op(value, neg_sixteen)) {
    return fp_zero();
  }

  int half_steps = 0;
  catapult_fp_t scaled = value;
  for (int step = 0; step < 6; ++step) {
    if (!fp_lt_op(scaled, neg_one)) {
      break;
    }
    scaled = fp_mul_op(scaled, half);
    ++half_steps;
  }

  catapult_fp_t term = fp_one();
  catapult_fp_t series = fp_one();
  for (int degree = 1; degree <= 6; ++degree) {
    term = fp_mul_op(term, fp_div_op(scaled, fp_const_int(degree)));
    series = fp_add_op(series, term);
  }

  for (int step = 0; step < half_steps; ++step) {
    series = fp_mul_op(series, series);
  }
  return series;
}

catapult_fp_t wrap_angle_fp(const catapult_fp_t& angle) {
  catapult_fp_t reduced = angle;
  const catapult_fp_t pi = fp_const(3.14159265358979323846f);
  const catapult_fp_t two_pi = fp_const(6.28318530717958647692f);

  for (int iter = 0; iter < 32; ++iter) {
    if (!fp_gt_op(reduced, pi)) {
      break;
    }
    reduced = fp_sub_op(reduced, two_pi);
  }
  for (int iter = 0; iter < 32; ++iter) {
    if (!fp_lt_op(reduced, fp_sub_op(fp_zero(), pi))) {
      break;
    }
    reduced = fp_add_op(reduced, two_pi);
  }
  return reduced;
}

void approx_sincos_fp(const catapult_fp_t& angle, catapult_fp_t* sin_value, catapult_fp_t* cos_value) {
  const catapult_fp_t pi = fp_const(3.14159265358979323846f);
  const catapult_fp_t half_pi = fp_const(1.57079632679489661923f);
  const catapult_fp_t neg_half_pi = fp_sub_op(fp_zero(), half_pi);

  catapult_fp_t reduced = wrap_angle_fp(angle);
  catapult_fp_t cos_sign = fp_one();

  if (fp_gt_op(reduced, half_pi)) {
    reduced = fp_sub_op(pi, reduced);
    cos_sign = fp_const(-1.0f);
  } else if (fp_lt_op(reduced, neg_half_pi)) {
    reduced = fp_sub_op(fp_sub_op(fp_zero(), pi), reduced);
    cos_sign = fp_const(-1.0f);
  }

  const catapult_fp_t x2 = fp_mul_op(reduced, reduced);
  const catapult_fp_t sin_poly = fp_mul_op(
      reduced,
      fp_add_op(
          fp_one(),
          fp_mul_op(
              x2,
              fp_add_op(
                  fp_const(-1.0f / 6.0f),
                  fp_mul_op(
                      x2,
                      fp_add_op(
                          fp_const(1.0f / 120.0f),
                          fp_mul_op(x2, fp_const(-1.0f / 5040.0f)))))))));
  const catapult_fp_t cos_poly = fp_add_op(
      fp_one(),
      fp_mul_op(
          x2,
          fp_add_op(
              fp_const(-0.5f),
              fp_mul_op(
                  x2,
                  fp_add_op(
                      fp_const(1.0f / 24.0f),
                      fp_mul_op(x2, fp_const(-1.0f / 720.0f)))))));
  *sin_value = sin_poly;
  *cos_value = fp_mul_op(cos_sign, cos_poly);
}

void rmsnorm_token_fp(
    const catapult_fp_t* input,
    const catapult_fp_t* weight,
    const catapult_fp_t& rms_eps,
    catapult_fp_t* output) {
  catapult_fp_t mean_square = fp_zero();
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    mean_square = fp_mac_op(input[dim], input[dim], mean_square);
  }
  mean_square = fp_div_op(mean_square, fp_const_int(llm_accel::kHiddenSize));
  const catapult_fp_t inv_rms = fp_div_op(fp_one(), fp_sqrt_op(fp_add_op(mean_square, rms_eps)));
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    output[dim] = fp_mul_op(fp_mul_op(input[dim], inv_rms), weight[dim]);
  }
}

void apply_rope_inplace_fp(catapult_fp_t* head, int token_index) {
  catapult_fp_t inv_freq = fp_one();
  const catapult_fp_t token_index_fp = fp_const_int(token_index);
  const catapult_fp_t rope_step = fp_const(0.8058421877614801f);
  for (int pair = 0; pair < llm_accel::kHeadDim / 2; ++pair) {
    const catapult_fp_t angle = fp_mul_op(token_index_fp, inv_freq);
    catapult_fp_t sinv = fp_zero();
    catapult_fp_t cosv = fp_one();
    approx_sincos_fp(angle, &sinv, &cosv);
    const int even_index = pair;
    const int odd_index = pair + llm_accel::kHeadDim / 2;
    const catapult_fp_t even = head[even_index];
    const catapult_fp_t odd = head[odd_index];
    head[even_index] = fp_sub_op(fp_mul_op(even, cosv), fp_mul_op(odd, sinv));
    head[odd_index] = fp_add_op(fp_mul_op(odd, cosv), fp_mul_op(even, sinv));
    inv_freq = fp_mul_op(inv_freq, rope_step);
  }
}

catapult_fp_t decode_int4_weight_fp(llm_accel::packed_w4_t packed_value, bool high_nibble) {
  const int nibble = high_nibble ? static_cast<int>((packed_value >> 4) & 0xF) : static_cast<int>(packed_value & 0xF);
  const int signed_nibble = nibble >= 8 ? nibble - 16 : nibble;
  return fp_const_int(signed_nibble);
}

catapult_fp_t dequantized_weight_fp(
    const llm_accel::packed_w4_t* packed_weights,
    const catapult_fp_t* scales,
    int out_index,
    int in_index,
    int in_dim) {
  const int flat_index = out_index * in_dim + in_index;
  const llm_accel::packed_w4_t packed_value = packed_weights[flat_index / 2];
  const bool high_nibble = (flat_index & 1) != 0;
  return fp_mul_op(decode_int4_weight_fp(packed_value, high_nibble), scales[out_index]);
}

void project_tiled_token_fp(
    const catapult_fp_t* input_token,
    const llm_accel::packed_w4_t* packed_weights,
    const catapult_fp_t* bias,
    const catapult_fp_t* scales,
    int out_dim,
    int in_dim,
    int out_tile,
    int in_tile,
    catapult_fp_t* output) {
  catapult_fp_t input_tile[kProjectionTileCapacity];
  catapult_fp_t partial_sum[kProjectionTileCapacity];

  for (int out_base = 0; out_base < out_dim; out_base += out_tile) {
    const int out_extent = min_int(out_tile, out_dim - out_base);
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      partial_sum[out_offset] = bias == nullptr ? fp_zero() : bias[out_base + out_offset];
    }

    for (int in_base = 0; in_base < in_dim; in_base += in_tile) {
      const int in_extent = min_int(in_tile, in_dim - in_base);
      for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
        input_tile[in_offset] = input_token[in_base + in_offset];
      }

      for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
        const int out_index = out_base + out_offset;
        catapult_fp_t accum = partial_sum[out_offset];
        for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
          accum = fp_mac_op(
              input_tile[in_offset],
              dequantized_weight_fp(packed_weights, scales, out_index, in_base + in_offset, in_dim),
              accum);
        }
        partial_sum[out_offset] = accum;
      }
    }

    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      output[out_base + out_offset] = partial_sum[out_offset];
    }
  }
}

void prefill_attention_context_block_fp(
    const catapult_fp_t q_proj[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int seq_len,
    int query_begin,
    int query_end,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    catapult_fp_t context[kPrefillQueryCapacity][llm_accel::kHiddenSize]) {
  const int key_tile = max_int(1, tile_config.key);
  const int query_heads_parallel = max_int(1, tile_config.query_heads_parallel);
  const catapult_fp_t attention_scaling = fp_const(0.08838834764831845f);

  for (int query_index = query_begin; query_index < query_end; ++query_index) {
    const catapult_fp_t* q_token = q_proj[query_index];
    catapult_fp_t* context_token = context[query_index - query_begin];

    for (int head_base = 0; head_base < llm_accel::kNumAttentionHeads; head_base += query_heads_parallel) {
      const int head_end = min_int(llm_accel::kNumAttentionHeads, head_base + query_heads_parallel);
      catapult_fp_t max_score[llm_accel::kNumAttentionHeads];
      catapult_fp_t denom[llm_accel::kNumAttentionHeads];
      catapult_fp_t accum[llm_accel::kNumAttentionHeads][llm_accel::kHeadDim];

      for (int head_offset = 0; head_offset < head_end - head_base; ++head_offset) {
        max_score[head_offset] = fp_const(-1.0e30f);
        denom[head_offset] = fp_zero();
        for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
          accum[head_offset][dim] = fp_zero();
        }
      }

      for (int key_begin = 0; key_begin <= query_index && key_begin < seq_len; key_begin += key_tile) {
        const int query_limit = query_index + 1;
        const int key_end = min_int(seq_len, min_int(query_limit, key_begin + key_tile));
        for (int head = head_base; head < head_end; ++head) {
          const int head_offset = head - head_base;
          const int kv_head = head / kNumGroups;
          const catapult_fp_t* q_head = q_token + head * llm_accel::kHeadDim;
          for (int key_index = key_begin; key_index < key_end; ++key_index) {
            const catapult_fp_t* k_head = k_proj + key_index * kKvWidth + kv_head * llm_accel::kHeadDim;
            catapult_fp_t score = fp_zero();
            for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
              score = fp_mac_op(q_head[dim], k_head[dim], score);
            }
            const catapult_fp_t scaled_score = fp_mul_op(score, attention_scaling);
            if (fp_gt_op(scaled_score, max_score[head_offset])) {
              max_score[head_offset] = scaled_score;
            }
          }
        }
      }

      for (int key_begin = 0; key_begin <= query_index && key_begin < seq_len; key_begin += key_tile) {
        const int query_limit = query_index + 1;
        const int key_end = min_int(seq_len, min_int(query_limit, key_begin + key_tile));
        for (int head = head_base; head < head_end; ++head) {
          const int head_offset = head - head_base;
          const int kv_head = head / kNumGroups;
          const catapult_fp_t* q_head = q_token + head * llm_accel::kHeadDim;
          for (int key_index = key_begin; key_index < key_end; ++key_index) {
            const catapult_fp_t* k_head = k_proj + key_index * kKvWidth + kv_head * llm_accel::kHeadDim;
            const catapult_fp_t* v_head = v_proj + key_index * kKvWidth + kv_head * llm_accel::kHeadDim;
            catapult_fp_t score = fp_zero();
            for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
              score = fp_mac_op(q_head[dim], k_head[dim], score);
            }
            const catapult_fp_t exp_score = approx_exp_fp(
                fp_sub_op(fp_mul_op(score, attention_scaling), max_score[head_offset]));
            denom[head_offset] = fp_add_op(denom[head_offset], exp_score);
            for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
              accum[head_offset][dim] = fp_mac_op(exp_score, v_head[dim], accum[head_offset][dim]);
            }
          }
        }
      }

      for (int head = head_base; head < head_end; ++head) {
        const int head_offset = head - head_base;
        catapult_fp_t* context_head = context_token + head * llm_accel::kHeadDim;
        const catapult_fp_t inv_denom = fp_gt_op(denom[head_offset], fp_zero())
            ? fp_div_op(fp_one(), denom[head_offset])
            : fp_zero();
        for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
          context_head[dim] = fp_mul_op(accum[head_offset][dim], inv_denom);
        }
      }
    }
  }
}

#endif

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

  if (seq_len > kPrefillSeqCapacity || tile_config.seq > kPrefillSeqCapacity ||
      tile_config.query > kPrefillQueryCapacity || tile_config.key > kPrefillKeyCapacity ||
      tile_config.hidden_proj > kProjectionTileCapacity || tile_config.kv_proj > kProjectionTileCapacity ||
      tile_config.query_heads_parallel > kNumAttentionHeads || tile_config.kv_heads_parallel > kNumKeyValueHeads) {
    return {false, 2};
  }

  const int seq_tile = max_int(1, tile_config.seq);
  const int query_tile = max_int(1, tile_config.query);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      scalar_t input_norm_token[kHiddenSize];
      const scalar_t* input_token = input_sequence + token_index * kHiddenSize;
      scalar_t* q_proj_token = g_q_proj_buffer[token_index];
      scalar_t* k_proj_token = k_cache + token_index * kKvWidth;
      scalar_t* v_proj_token = v_cache + token_index * kKvWidth;

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
        const int head_end = min_int(kNumAttentionHeads, head_base + tile_config.query_heads_parallel);
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace(q_proj_token + head * kHeadDim, token_index);
        }
      }
      for (int head_base = 0; head_base < kNumKeyValueHeads; head_base += tile_config.kv_heads_parallel) {
        const int head_end = min_int(kNumKeyValueHeads, head_base + tile_config.kv_heads_parallel);
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace(k_proj_token + head * kHeadDim, token_index);
        }
      }
    }
  }

  for (int query_begin = 0; query_begin < seq_len; query_begin += query_tile) {
    const int query_end = min_int(seq_len, query_begin + query_tile);
    prefill_attention_context_block(
        g_q_proj_buffer,
        k_cache,
        v_cache,
        seq_len,
        query_begin,
        query_end,
        tile_config,
        g_context_buffer);

    for (int query_index = query_begin; query_index < query_end; ++query_index) {
      const scalar_t* context_token = g_context_buffer[query_index - query_begin];
      scalar_t* output_token = output_sequence + query_index * kHiddenSize;
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

#ifdef __SYNTHESIS__

#pragma hls_design top
KernelStatus qwen_prefill_attention_kernel_catapult(
    const catapult_fp_t* input_sequence,
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t* input_layernorm_weight,
    catapult_fp_t rms_eps,
    const packed_w4_t* q_packed_weights,
    const packed_w4_t* k_packed_weights,
    const packed_w4_t* v_packed_weights,
    const packed_w4_t* o_packed_weights,
    const catapult_fp_t* q_bias,
    const catapult_fp_t* k_bias,
    const catapult_fp_t* v_bias,
    const catapult_fp_t* q_scales,
    const catapult_fp_t* k_scales,
    const catapult_fp_t* v_scales,
    const catapult_fp_t* o_scales,
    catapult_fp_t* k_cache,
    catapult_fp_t* v_cache,
    catapult_fp_t* output_sequence) {
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

  if (seq_len > kPrefillSeqCapacity || tile_config.seq > kPrefillSeqCapacity ||
      tile_config.query > kPrefillQueryCapacity || tile_config.key > kPrefillKeyCapacity ||
      tile_config.hidden_proj > kProjectionTileCapacity || tile_config.kv_proj > kProjectionTileCapacity ||
      tile_config.query_heads_parallel > kNumAttentionHeads || tile_config.kv_heads_parallel > kNumKeyValueHeads) {
    return {false, 2};
  }

  const int seq_tile = max_int(1, tile_config.seq);
  const int query_tile = max_int(1, tile_config.query);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      catapult_fp_t input_norm_token[kHiddenSize];
      const catapult_fp_t* input_token = input_sequence + token_index * kHiddenSize;
      catapult_fp_t* q_proj_token = g_q_proj_buffer_fp[token_index];
      catapult_fp_t* k_proj_token = k_cache + token_index * kKvWidth;
      catapult_fp_t* v_proj_token = v_cache + token_index * kKvWidth;

      rmsnorm_token_fp(input_token, input_layernorm_weight, rms_eps, input_norm_token);
      project_tiled_token_fp(
          input_norm_token,
          q_packed_weights,
          q_bias,
          q_scales,
          kHiddenSize,
          kHiddenSize,
          tile_config.hidden_proj,
          tile_config.hidden_proj,
          q_proj_token);
      project_tiled_token_fp(
          input_norm_token,
          k_packed_weights,
          k_bias,
          k_scales,
          kKvWidth,
          kHiddenSize,
          tile_config.kv_proj,
          tile_config.hidden_proj,
          k_proj_token);
      project_tiled_token_fp(
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
        const int head_end = min_int(kNumAttentionHeads, head_base + tile_config.query_heads_parallel);
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace_fp(q_proj_token + head * kHeadDim, token_index);
        }
      }
      for (int head_base = 0; head_base < kNumKeyValueHeads; head_base += tile_config.kv_heads_parallel) {
        const int head_end = min_int(kNumKeyValueHeads, head_base + tile_config.kv_heads_parallel);
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace_fp(k_proj_token + head * kHeadDim, token_index);
        }
      }
    }
  }

  for (int query_begin = 0; query_begin < seq_len; query_begin += query_tile) {
    const int query_end = min_int(seq_len, query_begin + query_tile);
    prefill_attention_context_block_fp(
        g_q_proj_buffer_fp,
        k_cache,
        v_cache,
        seq_len,
        query_begin,
        query_end,
        tile_config,
        g_context_buffer_fp);

    for (int query_index = query_begin; query_index < query_end; ++query_index) {
      const catapult_fp_t* context_token = g_context_buffer_fp[query_index - query_begin];
      catapult_fp_t* output_token = output_sequence + query_index * kHiddenSize;
      project_tiled_token_fp(
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

#endif

}  // namespace llm_accel