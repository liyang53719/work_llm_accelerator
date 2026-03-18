#include "qwen_prefill_attention_kernel.h"
#include "../include/ac_channel.h"

#ifdef __SYNTHESIS__
#include "../include/ac_int.h"
#include "../include/ac_std_float.h"
#include "../include/ccs_dw_fp_lib.h"
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
constexpr int kMaxDdrPortBitWidth = 256;
constexpr int kCatapultFpBitWidth = 32;
constexpr int kPackedW4BitWidth = 8;
constexpr int kMaxFpWordsPerBeat = kMaxDdrPortBitWidth / kCatapultFpBitWidth;
constexpr int kMaxPackedWordsPerBeat = kMaxDdrPortBitWidth / kPackedW4BitWidth;
constexpr int kContextTokenWordCount = llm_accel::kHiddenSize / kMaxFpWordsPerBeat;
constexpr int kContextScoreWordCount =
  (llm_accel::kNumAttentionHeads + kMaxFpWordsPerBeat - 1) / kMaxFpWordsPerBeat;
constexpr int kHiddenProjFpWordCount = kProjectionTileCapacity / kMaxFpWordsPerBeat;
constexpr int kHiddenProjPackedTileSize = kProjectionTileCapacity * kProjectionTileCapacity / 2;
constexpr int kHiddenProjPackedWordCount = kHiddenProjPackedTileSize / kMaxPackedWordsPerBeat;

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

using catapult_fp_t = llm_accel::prefill_catapult_fp_t;

constexpr int kFpIeeeCompliance = 0;
constexpr int kParallelMacLaneCount = llm_accel::kTileN;
constexpr int kRmsNormChunkCount = llm_accel::kHiddenSize / kParallelMacLaneCount;

#define HLS_DO_1(M, base) M(base)
#define HLS_DO_2(M, base) HLS_DO_1(M, base) M((base) + 1)
#define HLS_DO_4(M, base) HLS_DO_2(M, base) HLS_DO_2(M, (base) + 2)
#define HLS_DO_8(M, base) HLS_DO_4(M, base) HLS_DO_4(M, (base) + 4)
#define HLS_DO_16(M, base) HLS_DO_8(M, base) HLS_DO_8(M, (base) + 8)
#define HLS_DO_32(M, base) HLS_DO_16(M, base) HLS_DO_16(M, (base) + 16)
#define HLS_DO_64(M, base) HLS_DO_32(M, base) HLS_DO_32(M, (base) + 32)
#define HLS_DO_128(M, base) HLS_DO_64(M, base) HLS_DO_64(M, (base) + 64)

inline catapult_fp_t fp_add_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs);
inline catapult_fp_t fp_mul_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs);
inline catapult_fp_t fp_zero();
catapult_fp_t reduce_sum_128_fp(catapult_fp_t values[kParallelMacLaneCount]);
void rmsnorm_square_chunk_fp(
    const catapult_fp_t input[llm_accel::kHiddenSize],
    int base_index,
    catapult_fp_t lane_square[kParallelMacLaneCount]);
void rmsnorm_scale_weight_chunk_fp(
    const catapult_fp_t weight[llm_accel::kHiddenSize],
    const catapult_fp_t& inv_rms,
    int base_index,
    catapult_fp_t scaled_weight[llm_accel::kHiddenSize]);
void rmsnorm_apply_scale_chunk_fp(
    const catapult_fp_t input[llm_accel::kHiddenSize],
    const catapult_fp_t scaled_weight[llm_accel::kHiddenSize],
    int base_index,
    catapult_fp_t output[llm_accel::kHiddenSize]);
catapult_fp_t dequantized_weight_fp(
  const llm_accel::packed_w4_t* packed_weights,
  const catapult_fp_t* scales,
  int out_index,
  int in_index,
  int in_dim);

void weighted_chunk_128_fp(
    const catapult_fp_t* input_tile,
    const llm_accel::packed_w4_t* packed_weights,
    const catapult_fp_t* scales,
    int out_index,
    int in_index_base,
    int in_dim,
    int lane_extent,
    catapult_fp_t lane_products[kParallelMacLaneCount]) {
#pragma hls_unroll yes
  for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
    if (lane < lane_extent) {
      lane_products[lane] = fp_mul_op(
          input_tile[lane],
          dequantized_weight_fp(packed_weights, scales, out_index, in_index_base + lane, in_dim));
    } else {
      lane_products[lane] = fp_zero();
    }
  }
}

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
#pragma hls_unroll yes
  for (int step = 0; step < 6; ++step) {
    if (!fp_lt_op(scaled, neg_one)) {
      break;
    }
    scaled = fp_mul_op(scaled, half);
    ++half_steps;
  }

  catapult_fp_t term = fp_one();
  catapult_fp_t series = fp_one();
  term = fp_mul_op(term, scaled);
  series = fp_add_op(series, term);
  term = fp_mul_op(term, fp_mul_op(scaled, fp_const(0.5f)));
  series = fp_add_op(series, term);
  term = fp_mul_op(term, fp_mul_op(scaled, fp_const(1.0f / 3.0f)));
  series = fp_add_op(series, term);
  term = fp_mul_op(term, fp_mul_op(scaled, fp_const(0.25f)));
  series = fp_add_op(series, term);
  term = fp_mul_op(term, fp_mul_op(scaled, fp_const(0.2f)));
  series = fp_add_op(series, term);
  term = fp_mul_op(term, fp_mul_op(scaled, fp_const(1.0f / 6.0f)));
  series = fp_add_op(series, term);

  if (half_steps == 0) {
    return series;
  }
  catapult_fp_t squared = fp_mul_op(series, series);
  if (half_steps == 1) {
    return squared;
  }
  squared = fp_mul_op(squared, squared);
  if (half_steps == 2) {
    return squared;
  }
  squared = fp_mul_op(squared, squared);
  if (half_steps == 3) {
    return squared;
  }
  squared = fp_mul_op(squared, squared);
  if (half_steps == 4) {
    return squared;
  }
  squared = fp_mul_op(squared, squared);
  if (half_steps == 5) {
    return squared;
  }
  return fp_mul_op(squared, squared);
}

catapult_fp_t approx_rsqrt_fp(const catapult_fp_t& value) {
  if (fp_le_op(value, fp_zero())) {
    return fp_zero();
  }

  const catapult_fp_t half = fp_const(0.5f);
  const catapult_fp_t three_halves = fp_const(1.5f);
  const ac_int<32, false> value_bits = value.data_ac_int();
  const ac_int<32, false> guess_bits = ac_int<32, false>(0x5f3759dfU) - (value_bits >> 1);

  catapult_fp_t guess;
  guess.set_data(guess_bits);

  catapult_fp_t guess_sq = fp_mul_op(guess, guess);
  catapult_fp_t correction = fp_sub_op(three_halves, fp_mul_op(fp_mul_op(half, value), guess_sq));
  guess = fp_mul_op(guess, correction);

  guess_sq = fp_mul_op(guess, guess);
  correction = fp_sub_op(three_halves, fp_mul_op(fp_mul_op(half, value), guess_sq));
  guess = fp_mul_op(guess, correction);

  guess_sq = fp_mul_op(guess, guess);
  correction = fp_sub_op(three_halves, fp_mul_op(fp_mul_op(half, value), guess_sq));
  return fp_mul_op(guess, correction);
}

inline catapult_fp_t approx_reciprocal_fp(const catapult_fp_t& value) {
  const catapult_fp_t inv_sqrt = approx_rsqrt_fp(value);
  return fp_mul_op(inv_sqrt, inv_sqrt);
}

catapult_fp_t wrap_angle_fp(const catapult_fp_t& angle) {
  catapult_fp_t reduced = angle;
  const catapult_fp_t pi = fp_const(3.14159265358979323846f);
  const catapult_fp_t two_pi = fp_const(6.28318530717958647692f);

 #pragma hls_unroll yes
  for (int iter = 0; iter < 32; ++iter) {
    if (!fp_gt_op(reduced, pi)) {
      break;
    }
    reduced = fp_sub_op(reduced, two_pi);
  }
 #pragma hls_unroll yes
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
    const catapult_fp_t sin_tail = fp_add_op(fp_const(1.0f / 120.0f), fp_mul_op(x2, fp_const(-1.0f / 5040.0f)));
    const catapult_fp_t sin_mid = fp_add_op(fp_const(-1.0f / 6.0f), fp_mul_op(x2, sin_tail));
    const catapult_fp_t sin_scale = fp_add_op(fp_one(), fp_mul_op(x2, sin_mid));
    const catapult_fp_t sin_poly = fp_mul_op(reduced, sin_scale);

    const catapult_fp_t cos_tail = fp_add_op(fp_const(1.0f / 24.0f), fp_mul_op(x2, fp_const(-1.0f / 720.0f)));
    const catapult_fp_t cos_mid = fp_add_op(fp_const(-0.5f), fp_mul_op(x2, cos_tail));
    const catapult_fp_t cos_poly = fp_add_op(fp_one(), fp_mul_op(x2, cos_mid));
  *sin_value = sin_poly;
  *cos_value = fp_mul_op(cos_sign, cos_poly);
}

catapult_fp_t reduce_sum_128_fp(catapult_fp_t values[kParallelMacLaneCount]);

catapult_fp_t rmsnorm_square_sum_fp(
    const catapult_fp_t input[llm_accel::kHiddenSize]) {
  catapult_fp_t square_sum = fp_zero();
  for (int chunk_index = 0; chunk_index < kRmsNormChunkCount; ++chunk_index) {
    catapult_fp_t lane_square[kParallelMacLaneCount];
    rmsnorm_square_chunk_fp(input, chunk_index * kParallelMacLaneCount, lane_square);
    square_sum = fp_add_op(square_sum, reduce_sum_128_fp(lane_square));
  }
  return square_sum;
}

void rmsnorm_scale_weight_fp(
    const catapult_fp_t weight[llm_accel::kHiddenSize],
    const catapult_fp_t& inv_rms,
    catapult_fp_t scaled_weight[llm_accel::kHiddenSize]) {
  for (int chunk_index = 0; chunk_index < kRmsNormChunkCount; ++chunk_index) {
    rmsnorm_scale_weight_chunk_fp(weight, inv_rms, chunk_index * kParallelMacLaneCount, scaled_weight);
  }
}

void rmsnorm_apply_scale_fp(
    const catapult_fp_t input[llm_accel::kHiddenSize],
    const catapult_fp_t scaled_weight[llm_accel::kHiddenSize],
    catapult_fp_t output[llm_accel::kHiddenSize]) {
  for (int chunk_index = 0; chunk_index < kRmsNormChunkCount; ++chunk_index) {
    rmsnorm_apply_scale_chunk_fp(input, scaled_weight, chunk_index * kParallelMacLaneCount, output);
  }
}

void rmsnorm_token_fp(
    const catapult_fp_t input[llm_accel::kHiddenSize],
    const catapult_fp_t weight[llm_accel::kHiddenSize],
    const catapult_fp_t& rms_eps,
    catapult_fp_t output[llm_accel::kHiddenSize]) {
  catapult_fp_t scaled_weight[llm_accel::kHiddenSize];
  const catapult_fp_t mean_square = fp_mul_op(rmsnorm_square_sum_fp(input), fp_const(1.0f / 1536.0f));
  const catapult_fp_t inv_rms = approx_rsqrt_fp(fp_add_op(mean_square, rms_eps));
  rmsnorm_scale_weight_fp(weight, inv_rms, scaled_weight);
  rmsnorm_apply_scale_fp(input, scaled_weight, output);
}

void apply_rope_inplace_fp(catapult_fp_t* head, int token_index) {
  catapult_fp_t inv_freq = fp_one();
  const catapult_fp_t token_index_fp = fp_const_int(token_index);
  const catapult_fp_t rope_step = fp_const(0.8058421877614801f);
#pragma hls_unroll yes
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

catapult_fp_t reduce_sum_128_fp(catapult_fp_t values[kParallelMacLaneCount]) {
  catapult_fp_t stage64[kParallelMacLaneCount / 2];
  catapult_fp_t stage32[kParallelMacLaneCount / 4];
  catapult_fp_t stage16[kParallelMacLaneCount / 8];
  catapult_fp_t stage8[kParallelMacLaneCount / 16];
  catapult_fp_t stage4[kParallelMacLaneCount / 32];
  catapult_fp_t stage2[kParallelMacLaneCount / 64];

#define REDUCE_STAGE64(i) stage64[i] = fp_add_op(values[(i) * 2], values[(i) * 2 + 1]);
  HLS_DO_64(REDUCE_STAGE64, 0)
#undef REDUCE_STAGE64

#define REDUCE_STAGE32(i) stage32[i] = fp_add_op(stage64[(i) * 2], stage64[(i) * 2 + 1]);
  HLS_DO_32(REDUCE_STAGE32, 0)
#undef REDUCE_STAGE32

#define REDUCE_STAGE16(i) stage16[i] = fp_add_op(stage32[(i) * 2], stage32[(i) * 2 + 1]);
  HLS_DO_16(REDUCE_STAGE16, 0)
#undef REDUCE_STAGE16

#define REDUCE_STAGE8(i) stage8[i] = fp_add_op(stage16[(i) * 2], stage16[(i) * 2 + 1]);
  HLS_DO_8(REDUCE_STAGE8, 0)
#undef REDUCE_STAGE8

#define REDUCE_STAGE4(i) stage4[i] = fp_add_op(stage8[(i) * 2], stage8[(i) * 2 + 1]);
  HLS_DO_4(REDUCE_STAGE4, 0)
#undef REDUCE_STAGE4

#define REDUCE_STAGE2(i) stage2[i] = fp_add_op(stage4[(i) * 2], stage4[(i) * 2 + 1]);
  HLS_DO_2(REDUCE_STAGE2, 0)
#undef REDUCE_STAGE2

  return fp_add_op(stage2[0], stage2[1]);
}

void rmsnorm_square_chunk_fp(
    const catapult_fp_t input[llm_accel::kHiddenSize],
    int base_index,
    catapult_fp_t lane_square[kParallelMacLaneCount]) {
#pragma hls_unroll yes
  for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
    const catapult_fp_t value = input[base_index + lane];
    lane_square[lane] = fp_mul_op(value, value);
  }
}

void rmsnorm_scale_weight_chunk_fp(
    const catapult_fp_t weight[llm_accel::kHiddenSize],
    const catapult_fp_t& inv_rms,
    int base_index,
    catapult_fp_t scaled_weight[llm_accel::kHiddenSize]) {
#pragma hls_unroll yes
  for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
    scaled_weight[base_index + lane] = fp_mul_op(weight[base_index + lane], inv_rms);
  }
}

void rmsnorm_apply_scale_chunk_fp(
    const catapult_fp_t input[llm_accel::kHiddenSize],
    const catapult_fp_t scaled_weight[llm_accel::kHiddenSize],
    int base_index,
    catapult_fp_t output[llm_accel::kHiddenSize]) {
#pragma hls_unroll yes
  for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
    const int index = base_index + lane;
    output[index] = fp_mul_op(input[index], scaled_weight[index]);
  }
}

catapult_fp_t dot_product_128_fp(const catapult_fp_t* lhs, const catapult_fp_t* rhs) {
  catapult_fp_t lane_products[kParallelMacLaneCount];
#define DOT_LANE(i) lane_products[i] = fp_mul_op(lhs[i], rhs[i]);
  HLS_DO_128(DOT_LANE, 0)
#undef DOT_LANE
  return reduce_sum_128_fp(lane_products);
}

inline void zero_head_accum_128_fp(catapult_fp_t* accum_head) {
#define ZERO_ACCUM_LANE(i) accum_head[i] = fp_zero();
  HLS_DO_128(ZERO_ACCUM_LANE, 0)
#undef ZERO_ACCUM_LANE
}

catapult_fp_t weighted_chunk_dot_fp(
    const catapult_fp_t* input_tile,
    const llm_accel::packed_w4_t* packed_weights,
    const catapult_fp_t* scales,
    int out_index,
    int in_index_base,
    int in_dim,
    int lane_extent) {
  catapult_fp_t lane_products[kParallelMacLaneCount];
  weighted_chunk_128_fp(
      input_tile,
      packed_weights,
      scales,
      out_index,
      in_index_base,
      in_dim,
      lane_extent,
      lane_products);
  return reduce_sum_128_fp(lane_products);
}

void scaled_accum_128_fp(const catapult_fp_t& scale, const catapult_fp_t* vector, catapult_fp_t* accum) {
#pragma hls_unroll yes
  for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
    accum[dim] = fp_mac_op(scale, vector[dim], accum[dim]);
  }
}

void scale_store_128_fp(const catapult_fp_t* accum, const catapult_fp_t& scale, catapult_fp_t* output) {
#pragma hls_unroll yes
  for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
    output[dim] = fp_mul_op(accum[dim], scale);
  }
}

struct ContextKvTokenPacket {
  catapult_fp_t k_data[kKvWidth];
  catapult_fp_t v_data[kKvWidth];
};

struct ContextScorePacket {
  catapult_fp_t data[llm_accel::kNumAttentionHeads];
};

void load_context_kv_token_packet(
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int key_index,
    ContextKvTokenPacket* packet) {
#pragma hls_unroll yes
  for (int dim = 0; dim < kKvWidth; ++dim) {
    packet->k_data[dim] = k_proj[key_index * kKvWidth + dim];
    packet->v_data[dim] = v_proj[key_index * kKvWidth + dim];
  }
}

void compute_context_score_packet(
    const catapult_fp_t* q_token,
    const ContextKvTokenPacket& kv_packet,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    ContextScorePacket* packet) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < llm_accel::kNumAttentionHeads; ++head_offset) {
    if (head_base + head_offset < head_end) {
      const int head = head_base + head_offset;
      const int kv_head = head / kNumGroups;
      const catapult_fp_t* q_head = q_token + head * llm_accel::kHeadDim;
      const catapult_fp_t* k_head = kv_packet.k_data + kv_head * llm_accel::kHeadDim;
      const catapult_fp_t score = dot_product_128_fp(q_head, k_head);
      packet->data[head_offset] = fp_mul_op(score, attention_scaling);
    }
  }
}

void update_context_max_score_packet(
    const ContextScorePacket& packet,
    int head_base,
    int head_end,
    catapult_fp_t max_score[llm_accel::kNumAttentionHeads]) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < llm_accel::kNumAttentionHeads; ++head_offset) {
    if (head_base + head_offset < head_end && fp_gt_op(packet.data[head_offset], max_score[head_offset])) {
      max_score[head_offset] = packet.data[head_offset];
    }
  }
}

void accumulate_context_value_packet(
    const ContextScorePacket& score_packet,
    const ContextKvTokenPacket& kv_packet,
    int head_base,
    int head_end,
    const catapult_fp_t max_score[llm_accel::kNumAttentionHeads],
    catapult_fp_t denom[llm_accel::kNumAttentionHeads],
    catapult_fp_t accum[llm_accel::kNumAttentionHeads][llm_accel::kHeadDim]) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < llm_accel::kNumAttentionHeads; ++head_offset) {
    if (head_base + head_offset < head_end) {
      const int head = head_base + head_offset;
      const int kv_head = head / kNumGroups;
      const catapult_fp_t exp_score = approx_exp_fp(fp_sub_op(score_packet.data[head_offset], max_score[head_offset]));
      const catapult_fp_t* v_head = kv_packet.v_data + kv_head * llm_accel::kHeadDim;
      denom[head_offset] = fp_add_op(denom[head_offset], exp_score);
      scaled_accum_128_fp(exp_score, v_head, accum[head_offset]);
    }
  }
}

void process_context_score_key(
    const catapult_fp_t* q_token,
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int key_index,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    ContextKvTokenPacket* kv_packet,
    ContextScorePacket* score_packet) {
  load_context_kv_token_packet(k_proj, v_proj, key_index, kv_packet);
  compute_context_score_packet(q_token, *kv_packet, head_base, head_end, attention_scaling, score_packet);
}

void process_context_max_score_key(
    const catapult_fp_t* q_token,
    const catapult_fp_t* k_proj,
    int key_index,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    catapult_fp_t max_score[llm_accel::kNumAttentionHeads]) {
  ContextKvTokenPacket kv_packet;
  ContextScorePacket score_packet;
  process_context_score_key(
      q_token,
      k_proj,
      k_proj,
      key_index,
      head_base,
      head_end,
      attention_scaling,
      &kv_packet,
      &score_packet);
  update_context_max_score_packet(score_packet, head_base, head_end, max_score);
}

void process_context_value_key(
    const catapult_fp_t* q_token,
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int key_index,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    const catapult_fp_t max_score[llm_accel::kNumAttentionHeads],
    catapult_fp_t denom[llm_accel::kNumAttentionHeads],
    catapult_fp_t accum[llm_accel::kNumAttentionHeads][llm_accel::kHeadDim]) {
  ContextKvTokenPacket kv_packet;
  ContextScorePacket score_packet;
  process_context_score_key(
      q_token,
      k_proj,
      v_proj,
      key_index,
      head_base,
      head_end,
      attention_scaling,
      &kv_packet,
      &score_packet);
  accumulate_context_value_packet(
      score_packet,
      kv_packet,
      head_base,
      head_end,
      max_score,
      denom,
      accum);
}

void process_context_max_score_tile(
    const catapult_fp_t* q_token,
    const catapult_fp_t* k_proj,
    int key_begin,
    int key_end,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    catapult_fp_t max_score[llm_accel::kNumAttentionHeads]) {
ATTN_CONTEXT_MAX_KEY_LOOP:
#pragma hls_pipeline_init_interval 2
  for (int key_index = key_begin; key_index < key_end; ++key_index) {
    process_context_max_score_key(
        q_token,
        k_proj,
        key_index,
        head_base,
        head_end,
        attention_scaling,
        max_score);
  }
}

void process_context_value_tile(
    const catapult_fp_t* q_token,
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int key_begin,
    int key_end,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    const catapult_fp_t max_score[llm_accel::kNumAttentionHeads],
    catapult_fp_t denom[llm_accel::kNumAttentionHeads],
    catapult_fp_t accum[llm_accel::kNumAttentionHeads][llm_accel::kHeadDim]) {
ATTN_CONTEXT_VALUE_KEY_LOOP:
#pragma hls_pipeline_init_interval 2
  for (int key_index = key_begin; key_index < key_end; ++key_index) {
    process_context_value_key(
        q_token,
        k_proj,
        v_proj,
        key_index,
        head_base,
        head_end,
        attention_scaling,
        max_score,
        denom,
        accum);
  }
}

void attention_max_score_pass_fp(
    const catapult_fp_t* q_token,
    const catapult_fp_t* k_proj,
    int seq_len,
    int query_index,
    int key_tile,
    int head_base,
    int head_end,
    catapult_fp_t max_score[llm_accel::kNumAttentionHeads]) {
  const catapult_fp_t attention_scaling = fp_const(0.08838834764831845f);

  for (int key_begin = 0; key_begin <= query_index && key_begin < seq_len; key_begin += key_tile) {
    const int query_limit = query_index + 1;
    const int key_end = min_int(seq_len, min_int(query_limit, key_begin + key_tile));
    process_context_max_score_tile(
        q_token,
        k_proj,
        key_begin,
        key_end,
        head_base,
        head_end,
        attention_scaling,
        max_score);
  }
}

void attention_value_accum_pass_fp(
    const catapult_fp_t* q_token,
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int seq_len,
    int query_index,
    int key_tile,
    int head_base,
    int head_end,
    const catapult_fp_t max_score[llm_accel::kNumAttentionHeads],
    catapult_fp_t denom[llm_accel::kNumAttentionHeads],
    catapult_fp_t accum[llm_accel::kNumAttentionHeads][llm_accel::kHeadDim]) {
  const catapult_fp_t attention_scaling = fp_const(0.08838834764831845f);

  for (int key_begin = 0; key_begin <= query_index && key_begin < seq_len; key_begin += key_tile) {
    const int query_limit = query_index + 1;
    const int key_end = min_int(seq_len, min_int(query_limit, key_begin + key_tile));
    process_context_value_tile(
        q_token,
        k_proj,
        v_proj,
        key_begin,
        key_end,
        head_base,
        head_end,
        attention_scaling,
        max_score,
        denom,
        accum);
  }
}

struct ContextHeadStatePacket {
  catapult_fp_t max_score[llm_accel::kNumAttentionHeads];
  catapult_fp_t denom[llm_accel::kNumAttentionHeads];
  catapult_fp_t accum[llm_accel::kNumAttentionHeads][llm_accel::kHeadDim];
};

struct ContextQueryPacket {
  catapult_fp_t data[llm_accel::kHiddenSize];
};

struct ContextTokenPacket {
  catapult_fp_t data[llm_accel::kHiddenSize];
};

struct ContextQueryMetaPacket {
  int query_index;
  int query_offset;
};

struct ContextResultMetaPacket {
  int query_offset;
};

struct ContextFpWordPacket {
  catapult_fp_t data[kMaxFpWordsPerBeat];
};

void load_context_query_packet_from_sequence(
    const catapult_fp_t q_proj[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    int query_index,
    ContextQueryPacket* packet) {
#pragma hls_unroll yes
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    packet->data[dim] = q_proj[query_index][dim];
  }
}

void load_context_query_packet_from_tile(
    const catapult_fp_t q_proj_tile[kPrefillQueryCapacity][llm_accel::kHiddenSize],
    int query_offset,
    ContextQueryPacket* packet) {
#pragma hls_unroll yes
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    packet->data[dim] = q_proj_tile[query_offset][dim];
  }
}

void store_context_token_packet(
    const ContextTokenPacket& packet,
    catapult_fp_t context[kPrefillQueryCapacity][llm_accel::kHiddenSize],
    int query_offset) {
#pragma hls_unroll yes
  for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
    context[query_offset][dim] = packet.data[dim];
  }
}

void init_context_query_meta_packet(
    int query_index,
    int query_offset,
    ContextQueryMetaPacket* packet) {
  packet->query_index = query_index;
  packet->query_offset = query_offset;
}

void init_context_result_meta_packet(
    int query_offset,
    ContextResultMetaPacket* packet) {
  packet->query_offset = query_offset;
}

void load_context_fp_word_packet(
    const catapult_fp_t* source,
    int base,
    ContextFpWordPacket* packet) {
#pragma hls_unroll yes
  for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
    packet->data[index] = source[base + index];
  }
}

void store_context_fp_word_packet(
    const ContextFpWordPacket& packet,
    catapult_fp_t* destination,
    int base) {
#pragma hls_unroll yes
  for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
    destination[base + index] = packet.data[index];
  }
}

void stream_context_query_packet_words(
    const ContextQueryPacket& query_packet,
    ac_channel<ContextFpWordPacket>& query_word_chan) {
  for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
    ContextFpWordPacket word_packet;
    load_context_fp_word_packet(query_packet.data, word_index * kMaxFpWordsPerBeat, &word_packet);
    query_word_chan.write(word_packet);
  }
}

void read_context_query_packet_words(
    ac_channel<ContextFpWordPacket>& query_word_chan,
    ContextQueryPacket* query_packet) {
  for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
    const ContextFpWordPacket word_packet = query_word_chan.read();
    store_context_fp_word_packet(word_packet, query_packet->data, word_index * kMaxFpWordsPerBeat);
  }
}

void stream_context_result_packet_words(
    const ContextTokenPacket& context_packet,
    ac_channel<ContextFpWordPacket>& context_word_chan) {
  for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
    ContextFpWordPacket word_packet;
    load_context_fp_word_packet(context_packet.data, word_index * kMaxFpWordsPerBeat, &word_packet);
    context_word_chan.write(word_packet);
  }
}

void read_context_result_packet_words(
    ac_channel<ContextFpWordPacket>& context_word_chan,
    ContextTokenPacket* context_packet) {
  for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
    const ContextFpWordPacket word_packet = context_word_chan.read();
    store_context_fp_word_packet(word_packet, context_packet->data, word_index * kMaxFpWordsPerBeat);
  }
}

void stream_context_score_packet_words(
    const catapult_fp_t max_score[llm_accel::kNumAttentionHeads],
    ac_channel<ContextFpWordPacket>& max_score_word_chan) {
  for (int word_index = 0; word_index < kContextScoreWordCount; ++word_index) {
    ContextFpWordPacket word_packet;

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      const int head_index = word_index * kMaxFpWordsPerBeat + index;
      word_packet.data[index] = head_index < llm_accel::kNumAttentionHeads ? max_score[head_index] : fp_zero();
    }

    max_score_word_chan.write(word_packet);
  }
}

void read_context_score_packet_words(
    ac_channel<ContextFpWordPacket>& max_score_word_chan,
    catapult_fp_t max_score[llm_accel::kNumAttentionHeads]) {
  for (int word_index = 0; word_index < kContextScoreWordCount; ++word_index) {
    const ContextFpWordPacket word_packet = max_score_word_chan.read();

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      const int head_index = word_index * kMaxFpWordsPerBeat + index;
      if (head_index < llm_accel::kNumAttentionHeads) {
        max_score[head_index] = word_packet.data[index];
      }
    }
  }
}

void init_context_max_score_packet(
    int head_base,
    int head_end,
    catapult_fp_t max_score[llm_accel::kNumAttentionHeads]) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < llm_accel::kNumAttentionHeads; ++head_offset) {
    if (head_offset < head_end - head_base) {
      max_score[head_offset] = fp_const(-1.0e30f);
    }
  }
}

void init_context_value_head_state_packet(
    const catapult_fp_t max_score[llm_accel::kNumAttentionHeads],
    int head_base,
    int head_end,
    ContextHeadStatePacket* packet) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < llm_accel::kNumAttentionHeads; ++head_offset) {
    if (head_offset < head_end - head_base) {
      packet->max_score[head_offset] = max_score[head_base + head_offset];
      packet->denom[head_offset] = fp_zero();
      zero_head_accum_128_fp(packet->accum[head_offset]);
    }
  }
}

void compute_context_max_score_head_state_packet(
    const catapult_fp_t* q_token,
    const catapult_fp_t* k_proj,
    int seq_len,
    int query_index,
    int key_tile,
    int head_base,
    int head_end,
    catapult_fp_t max_score[llm_accel::kNumAttentionHeads]) {
  attention_max_score_pass_fp(
      q_token,
      k_proj,
      seq_len,
      query_index,
      key_tile,
      head_base,
      head_end,
      max_score);
}

void compute_context_value_head_state_packet(
    const catapult_fp_t* q_token,
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int seq_len,
    int query_index,
    int key_tile,
    int head_base,
    int head_end,
    ContextHeadStatePacket* packet) {
  attention_value_accum_pass_fp(
      q_token,
      k_proj,
      v_proj,
      seq_len,
      query_index,
      key_tile,
      head_base,
      head_end,
      packet->max_score,
      packet->denom,
      packet->accum);
}

void store_context_head_state_packet(
    const ContextHeadStatePacket& packet,
    int head_base,
    int head_end,
    catapult_fp_t* context_token) {
#pragma hls_unroll yes
  for (int head = head_base; head < head_end; ++head) {
    const int head_offset = head - head_base;
    catapult_fp_t* context_head = context_token + head * llm_accel::kHeadDim;
    const catapult_fp_t inv_denom = fp_gt_op(packet.denom[head_offset], fp_zero())
        ? approx_reciprocal_fp(packet.denom[head_offset])
        : fp_zero();
    scale_store_128_fp(packet.accum[head_offset], inv_denom, context_head);
  }
}

void process_context_head_group(
    const ContextQueryPacket& q_packet,
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int seq_len,
    int query_index,
    int key_tile,
    int head_base,
    int head_end,
    ContextTokenPacket* context_packet) {
  ContextHeadStatePacket head_state_packet;
  catapult_fp_t max_score[llm_accel::kNumAttentionHeads];

  init_context_max_score_packet(head_base, head_end, max_score);
  compute_context_max_score_head_state_packet(
      q_packet.data,
      k_proj,
      seq_len,
      query_index,
      key_tile,
      head_base,
      head_end,
      max_score);
  init_context_value_head_state_packet(max_score, head_base, head_end, &head_state_packet);
  compute_context_value_head_state_packet(
      q_packet.data,
      k_proj,
      v_proj,
      seq_len,
      query_index,
      key_tile,
      head_base,
      head_end,
      &head_state_packet);
  store_context_head_state_packet(head_state_packet, head_base, head_end, context_packet->data);
}

void prefill_attention_context_query_max_score_fp(
    const ContextQueryPacket& q_packet,
    const catapult_fp_t* k_proj,
    int seq_len,
    int query_index,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    catapult_fp_t max_score[llm_accel::kNumAttentionHeads]) {
  const int key_tile = max_int(1, tile_config.key);
  const int query_heads_parallel = max_int(1, tile_config.query_heads_parallel);

  for (int head_base = 0; head_base < llm_accel::kNumAttentionHeads; head_base += query_heads_parallel) {
    const int head_end = min_int(llm_accel::kNumAttentionHeads, head_base + query_heads_parallel);
    catapult_fp_t head_group_max_score[llm_accel::kNumAttentionHeads];

    init_context_max_score_packet(head_base, head_end, head_group_max_score);
    compute_context_max_score_head_state_packet(
        q_packet.data,
        k_proj,
        seq_len,
        query_index,
        key_tile,
        head_base,
        head_end,
        head_group_max_score);

#pragma hls_unroll yes
    for (int head = head_base; head < head_end; ++head) {
      max_score[head] = head_group_max_score[head - head_base];
    }
  }
}

void process_context_value_head_group(
    const ContextQueryPacket& q_packet,
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int seq_len,
    int query_index,
    int key_tile,
    int head_base,
    int head_end,
    const catapult_fp_t max_score[llm_accel::kNumAttentionHeads],
    ContextTokenPacket* context_packet) {
  ContextHeadStatePacket head_state_packet;

  init_context_value_head_state_packet(max_score, head_base, head_end, &head_state_packet);
  compute_context_value_head_state_packet(
      q_packet.data,
      k_proj,
      v_proj,
      seq_len,
      query_index,
      key_tile,
      head_base,
      head_end,
      &head_state_packet);
  store_context_head_state_packet(head_state_packet, head_base, head_end, context_packet->data);
}

void prefill_attention_context_query_value_fp(
    const ContextQueryPacket& q_packet,
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int seq_len,
    int query_index,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t max_score[llm_accel::kNumAttentionHeads],
    ContextTokenPacket* context_packet) {
  const int key_tile = max_int(1, tile_config.key);
  const int query_heads_parallel = max_int(1, tile_config.query_heads_parallel);

  for (int head_base = 0; head_base < llm_accel::kNumAttentionHeads; head_base += query_heads_parallel) {
    const int head_end = min_int(llm_accel::kNumAttentionHeads, head_base + query_heads_parallel);
    process_context_value_head_group(
        q_packet,
        k_proj,
        v_proj,
        seq_len,
        query_index,
        key_tile,
        head_base,
        head_end,
        max_score,
        context_packet);
  }
}

void stream_context_query_tasks_from_sequence(
    const catapult_fp_t q_proj[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    int query_begin,
    int query_end,
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan) {
  for (int query_index = query_begin; query_index < query_end; ++query_index) {
    ContextQueryPacket q_packet;
    ContextQueryMetaPacket meta_packet;
    const int query_offset = query_index - query_begin;

    load_context_query_packet_from_sequence(q_proj, query_index, &q_packet);
    init_context_query_meta_packet(query_index, query_offset, &meta_packet);
    query_meta_chan.write(meta_packet);
    stream_context_query_packet_words(q_packet, query_word_chan);
  }
}

void stream_context_query_tasks_from_tile(
    const catapult_fp_t q_proj_tile[kPrefillQueryCapacity][llm_accel::kHiddenSize],
    int query_begin,
    int query_end,
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan) {
  for (int query_index = query_begin; query_index < query_end; ++query_index) {
    ContextQueryPacket q_packet;
    ContextQueryMetaPacket meta_packet;
    const int query_offset = query_index - query_begin;

    load_context_query_packet_from_tile(q_proj_tile, query_offset, &q_packet);
    init_context_query_meta_packet(query_index, query_offset, &meta_packet);
    query_meta_chan.write(meta_packet);
    stream_context_query_packet_words(q_packet, query_word_chan);
  }
}

void compute_context_score_tasks(
    const catapult_fp_t* k_proj,
    int seq_len,
    int query_count,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan,
    ac_channel<ContextQueryMetaPacket>& score_meta_chan,
    ac_channel<ContextFpWordPacket>& score_query_word_chan,
    ac_channel<ContextFpWordPacket>& max_score_word_chan) {
  for (int query_slot = 0; query_slot < query_count; ++query_slot) {
    const ContextQueryMetaPacket meta_packet = query_meta_chan.read();
    ContextQueryPacket query_packet;
    catapult_fp_t max_score[llm_accel::kNumAttentionHeads];

    read_context_query_packet_words(query_word_chan, &query_packet);
    prefill_attention_context_query_max_score_fp(
        query_packet,
        k_proj,
        seq_len,
        meta_packet.query_index,
        tile_config,
        max_score);
    score_meta_chan.write(meta_packet);
    stream_context_query_packet_words(query_packet, score_query_word_chan);
    stream_context_score_packet_words(max_score, max_score_word_chan);
  }
}

void compute_context_value_tasks(
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int seq_len,
    int query_count,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    ac_channel<ContextQueryMetaPacket>& score_meta_chan,
    ac_channel<ContextFpWordPacket>& score_query_word_chan,
    ac_channel<ContextFpWordPacket>& max_score_word_chan,
    ac_channel<ContextResultMetaPacket>& context_meta_chan,
    ac_channel<ContextFpWordPacket>& context_word_chan) {
  for (int query_slot = 0; query_slot < query_count; ++query_slot) {
    const ContextQueryMetaPacket meta_packet = score_meta_chan.read();
    ContextQueryPacket query_packet;
    ContextResultMetaPacket result_meta_packet;
    ContextTokenPacket context_packet;
    catapult_fp_t max_score[llm_accel::kNumAttentionHeads];

    read_context_query_packet_words(score_query_word_chan, &query_packet);
    read_context_score_packet_words(max_score_word_chan, max_score);
    prefill_attention_context_query_value_fp(
        query_packet,
        k_proj,
        v_proj,
        seq_len,
        meta_packet.query_index,
        tile_config,
        max_score,
        &context_packet);
    init_context_result_meta_packet(meta_packet.query_offset, &result_meta_packet);
    context_meta_chan.write(result_meta_packet);
    stream_context_result_packet_words(context_packet, context_word_chan);
  }
}

void store_context_result_packets(
    int query_count,
    ac_channel<ContextResultMetaPacket>& context_meta_chan,
    ac_channel<ContextFpWordPacket>& context_word_chan,
    catapult_fp_t context[kPrefillQueryCapacity][llm_accel::kHiddenSize]) {
  for (int query_slot = 0; query_slot < query_count; ++query_slot) {
    const ContextResultMetaPacket meta_packet = context_meta_chan.read();
    ContextTokenPacket context_packet;
    read_context_result_packet_words(context_word_chan, &context_packet);
    store_context_token_packet(context_packet, context, meta_packet.query_offset);
  }
}

void prefill_attention_context_block_stream_fp(
    const catapult_fp_t q_proj[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int seq_len,
    int query_begin,
    int query_end,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    catapult_fp_t context[kPrefillQueryCapacity][llm_accel::kHiddenSize]) {
  ac_channel<ContextQueryMetaPacket> query_meta_chan;
  ac_channel<ContextFpWordPacket> query_word_chan;
    ac_channel<ContextQueryMetaPacket> score_meta_chan;
    ac_channel<ContextFpWordPacket> score_query_word_chan;
    ac_channel<ContextFpWordPacket> max_score_word_chan;
  ac_channel<ContextResultMetaPacket> context_meta_chan;
  ac_channel<ContextFpWordPacket> context_word_chan;
  const int query_count = query_end - query_begin;

  stream_context_query_tasks_from_sequence(q_proj, query_begin, query_end, query_meta_chan, query_word_chan);
    compute_context_score_tasks(
      k_proj,
      seq_len,
      query_count,
      tile_config,
      query_meta_chan,
      query_word_chan,
      score_meta_chan,
      score_query_word_chan,
      max_score_word_chan);
    compute_context_value_tasks(
      k_proj,
      v_proj,
      seq_len,
      query_count,
      tile_config,
      score_meta_chan,
      score_query_word_chan,
      max_score_word_chan,
      context_meta_chan,
      context_word_chan);
  store_context_result_packets(query_count, context_meta_chan, context_word_chan, context);
}

void prefill_attention_context_query_tile_stream_fp(
    const catapult_fp_t q_proj_tile[kPrefillQueryCapacity][llm_accel::kHiddenSize],
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int seq_len,
    int query_begin,
    int query_end,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    catapult_fp_t context[kPrefillQueryCapacity][llm_accel::kHiddenSize]) {
  ac_channel<ContextQueryMetaPacket> query_meta_chan;
  ac_channel<ContextFpWordPacket> query_word_chan;
    ac_channel<ContextQueryMetaPacket> score_meta_chan;
    ac_channel<ContextFpWordPacket> score_query_word_chan;
    ac_channel<ContextFpWordPacket> max_score_word_chan;
  ac_channel<ContextResultMetaPacket> context_meta_chan;
  ac_channel<ContextFpWordPacket> context_word_chan;
  const int query_count = query_end - query_begin;

  stream_context_query_tasks_from_tile(q_proj_tile, query_begin, query_end, query_meta_chan, query_word_chan);
    compute_context_score_tasks(
      k_proj,
      seq_len,
      query_count,
      tile_config,
      query_meta_chan,
      query_word_chan,
      score_meta_chan,
      score_query_word_chan,
      max_score_word_chan);
    compute_context_value_tasks(
      k_proj,
      v_proj,
      seq_len,
      query_count,
      tile_config,
      score_meta_chan,
      score_query_word_chan,
      max_score_word_chan,
      context_meta_chan,
      context_word_chan);
  store_context_result_packets(query_count, context_meta_chan, context_word_chan, context);
}

template <bool HasBias, int OutDim, int InDim>
void project_tiled_token_fp_impl(
    const catapult_fp_t* input_token,
    const llm_accel::packed_w4_t* packed_weights,
    const catapult_fp_t* bias,
    const catapult_fp_t* scales,
    int out_tile,
    int in_tile,
    catapult_fp_t* output) {
  catapult_fp_t input_tile[kProjectionTileCapacity];
  catapult_fp_t partial_sum[kProjectionTileCapacity];

  for (int out_base = 0; out_base < OutDim; out_base += out_tile) {
    const int out_extent = min_int(out_tile, OutDim - out_base);
ATTN_PROJ_INIT_LOOP:
#pragma hls_pipeline_init_interval 1
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      partial_sum[out_offset] = HasBias ? bias[out_base + out_offset] : fp_zero();
    }

    for (int in_base = 0; in_base < InDim; in_base += in_tile) {
      const int in_extent = min_int(in_tile, InDim - in_base);
ATTN_PROJ_LOAD_LOOP:
#pragma hls_pipeline_init_interval 1
      for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
        input_tile[in_offset] = input_token[in_base + in_offset];
      }

ATTN_PROJ_ACCUM_LOOP:
#pragma hls_pipeline_init_interval 2
      for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
        const int out_index = out_base + out_offset;
        catapult_fp_t accum = partial_sum[out_offset];
        for (int in_offset = 0; in_offset < in_extent; in_offset += kParallelMacLaneCount) {
          const int lane_extent = min_int(kParallelMacLaneCount, in_extent - in_offset);
          accum = fp_add_op(
              accum,
              weighted_chunk_dot_fp(
                  input_tile + in_offset,
                  packed_weights,
                  scales,
                  out_index,
                  in_base + in_offset,
                  InDim,
                  lane_extent));
        }
        partial_sum[out_offset] = accum;
      }
    }

ATTN_PROJ_STORE_LOOP:
#pragma hls_pipeline_init_interval 1
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      output[out_base + out_offset] = partial_sum[out_offset];
    }
  }
}

void project_hidden_token_bias_fp(
    const catapult_fp_t input_token[llm_accel::kHiddenSize],
    const llm_accel::packed_w4_t packed_weights[llm_accel::kHiddenSize * llm_accel::kHiddenSize / 2],
    const catapult_fp_t bias[llm_accel::kHiddenSize],
    const catapult_fp_t scales[llm_accel::kHiddenSize],
    int out_tile,
    int in_tile,
    catapult_fp_t output[llm_accel::kHiddenSize]) {
  project_tiled_token_fp_impl<true, llm_accel::kHiddenSize, llm_accel::kHiddenSize>(
      input_token,
      packed_weights,
      bias,
      scales,
      out_tile,
      in_tile,
      output);
}

void project_kv_token_bias_fp(
    const catapult_fp_t input_token[llm_accel::kHiddenSize],
    const llm_accel::packed_w4_t packed_weights[kKvWidth * llm_accel::kHiddenSize / 2],
    const catapult_fp_t bias[kKvWidth],
    const catapult_fp_t scales[kKvWidth],
    int out_tile,
    int in_tile,
    catapult_fp_t output[kKvWidth]) {
  project_tiled_token_fp_impl<true, kKvWidth, llm_accel::kHiddenSize>(
      input_token,
      packed_weights,
      bias,
      scales,
      out_tile,
      in_tile,
      output);
}

void project_hidden_token_fp(
    const catapult_fp_t input_token[llm_accel::kHiddenSize],
    const llm_accel::packed_w4_t packed_weights[llm_accel::kHiddenSize * llm_accel::kHiddenSize / 2],
    const catapult_fp_t scales[llm_accel::kHiddenSize],
    int out_tile,
    int in_tile,
    catapult_fp_t output[llm_accel::kHiddenSize]) {
  project_tiled_token_fp_impl<false, llm_accel::kHiddenSize, llm_accel::kHiddenSize>(
      input_token,
      packed_weights,
      nullptr,
      scales,
      out_tile,
      in_tile,
      output);
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
  prefill_attention_context_block_stream_fp(
      q_proj,
      k_proj,
      v_proj,
      seq_len,
      query_begin,
      query_end,
      tile_config,
      context);
}

void prefill_attention_context_query_tile_fp(
    const catapult_fp_t q_proj_tile[kPrefillQueryCapacity][llm_accel::kHiddenSize],
    const catapult_fp_t* k_proj,
    const catapult_fp_t* v_proj,
    int seq_len,
    int query_begin,
    int query_end,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    catapult_fp_t context[kPrefillQueryCapacity][llm_accel::kHiddenSize]) {
  prefill_attention_context_query_tile_stream_fp(
      q_proj_tile,
      k_proj,
      v_proj,
      seq_len,
      query_begin,
      query_end,
      tile_config,
      context);
}

#endif

}  // namespace

namespace llm_accel {

struct HiddenProjFpTilePacket {
  prefill_catapult_fp_t data[kProjectionTileCapacity];
};

struct HiddenProjPackedWeightTilePacket {
  packed_w4_t data[kProjectionTileCapacity * kProjectionTileCapacity / 2];
};

struct HiddenProjScaleTilePacket {
  prefill_catapult_fp_t data[kProjectionTileCapacity];
};

struct HiddenProjPartialTilePacket {
  prefill_catapult_fp_t data[kProjectionTileCapacity];
};

struct HiddenProjFpWordPacket {
  prefill_catapult_fp_t data[kMaxFpWordsPerBeat];
};

struct HiddenProjPackedWeightWordPacket {
  packed_w4_t data[kMaxPackedWordsPerBeat];
};

void qwen_prefill_attention_hidden_proj_tile_array_core(
    const prefill_catapult_fp_t input_tile[kProjectionTileCapacity],
    const prefill_catapult_fp_t input_layernorm_weight_tile[kProjectionTileCapacity],
    const packed_w4_t packed_weights_tile[kProjectionTileCapacity * kProjectionTileCapacity / 2],
    const prefill_catapult_fp_t scales_tile[kProjectionTileCapacity],
    prefill_catapult_fp_t partial_sum_tile[kProjectionTileCapacity],
    prefill_catapult_fp_t inv_rms,
    int lane_extent,
    int out_extent,
    bool apply_rmsnorm);

void load_hidden_proj_fp_tile_packet(
    const prefill_catapult_fp_t* source,
    int base,
    int extent,
    HiddenProjFpTilePacket* packet) {
  for (int index = 0; index < kProjectionTileCapacity; ++index) {
    packet->data[index] = index < extent ? source[base + index] : prefill_catapult_fp_t(0.0f);
  }
}

void init_hidden_proj_zero_tile_packet(HiddenProjFpTilePacket* packet) {
  for (int index = 0; index < kProjectionTileCapacity; ++index) {
    packet->data[index] = prefill_catapult_fp_t(0.0f);
  }
}

void load_hidden_proj_packed_weight_tile_packet(
    const packed_w4_t* packed_weights,
    int out_base,
    int out_extent,
    int in_base,
    int lane_extent,
    HiddenProjPackedWeightTilePacket* packet) {
#pragma hls_pipeline_init_interval 1
  for (int out_offset = 0; out_offset < kProjectionTileCapacity; ++out_offset) {
    for (int lane_pair = 0; lane_pair < kProjectionTileCapacity / 2; ++lane_pair) {
      int low_nibble = 0;
      int high_nibble = 0;

      if (out_offset < out_extent) {
        const int lane0 = lane_pair * 2;
        const int lane1 = lane0 + 1;

        if (lane0 < lane_extent) {
          const int flat_index0 = (out_base + out_offset) * kHiddenSize + in_base + lane0;
          const packed_w4_t packed_value0 = packed_weights[flat_index0 / 2];
          low_nibble = static_cast<int>((packed_value0 >> ((flat_index0 & 1) != 0 ? 4 : 0)) & 0xF);
        }

        if (lane1 < lane_extent) {
          const int flat_index1 = (out_base + out_offset) * kHiddenSize + in_base + lane1;
          const packed_w4_t packed_value1 = packed_weights[flat_index1 / 2];
          high_nibble = static_cast<int>((packed_value1 >> ((flat_index1 & 1) != 0 ? 4 : 0)) & 0xF);
        }
      }

      packet->data[out_offset * (kProjectionTileCapacity / 2) + lane_pair] =
          static_cast<packed_w4_t>((high_nibble << 4) | low_nibble);
    }
  }
}

void load_hidden_proj_scale_tile_packet(
    const prefill_catapult_fp_t* scales,
    int out_base,
    int out_extent,
    HiddenProjScaleTilePacket* packet) {
  for (int index = 0; index < kProjectionTileCapacity; ++index) {
    packet->data[index] = index < out_extent ? scales[out_base + index] : prefill_catapult_fp_t(0.0f);
  }
}

void init_hidden_proj_partial_tile_packet(
    const prefill_catapult_fp_t* bias,
    int out_base,
    int out_extent,
    HiddenProjPartialTilePacket* packet) {
  for (int index = 0; index < kProjectionTileCapacity; ++index) {
    if (index < out_extent) {
      packet->data[index] = bias == nullptr ? fp_zero() : bias[out_base + index];
    } else {
      packet->data[index] = fp_zero();
    }
  }
}

void store_hidden_proj_partial_tile_packet(
    const HiddenProjPartialTilePacket& packet,
    int out_extent,
    prefill_catapult_fp_t* output,
    int out_base) {
  for (int index = 0; index < out_extent; ++index) {
    output[out_base + index] = packet.data[index];
  }
}

void load_hidden_proj_fp_word_packet(
    const prefill_catapult_fp_t* source,
    int base,
    HiddenProjFpWordPacket* packet) {
  for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
    packet->data[index] = source[base + index];
  }
}

void store_hidden_proj_fp_word_packet(
    const HiddenProjFpWordPacket& packet,
    prefill_catapult_fp_t* destination,
    int base) {
  for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
    destination[base + index] = packet.data[index];
  }
}

void load_hidden_proj_packed_weight_word_packet(
    const packed_w4_t* source,
    int base,
    HiddenProjPackedWeightWordPacket* packet) {
  for (int index = 0; index < kMaxPackedWordsPerBeat; ++index) {
    packet->data[index] = source[base + index];
  }
}

void store_hidden_proj_packed_weight_word_packet(
    const HiddenProjPackedWeightWordPacket& packet,
    packed_w4_t* destination,
    int base) {
  for (int index = 0; index < kMaxPackedWordsPerBeat; ++index) {
    destination[base + index] = packet.data[index];
  }
}

void read_hidden_proj_fp_tile_packet(
    ac_channel<HiddenProjFpWordPacket>& channel,
    HiddenProjFpTilePacket* packet) {
  for (int word_index = 0; word_index < kHiddenProjFpWordCount; ++word_index) {
    const HiddenProjFpWordPacket word_packet = channel.read();
    store_hidden_proj_fp_word_packet(word_packet, packet->data, word_index * kMaxFpWordsPerBeat);
  }
}

void read_hidden_proj_fp_tile_words(
    ac_channel<HiddenProjFpWordPacket>& channel,
    prefill_catapult_fp_t* destination) {
  for (int word_index = 0; word_index < kHiddenProjFpWordCount; ++word_index) {
    const HiddenProjFpWordPacket word_packet = channel.read();
    store_hidden_proj_fp_word_packet(word_packet, destination, word_index * kMaxFpWordsPerBeat);
  }
}

void write_hidden_proj_fp_tile_packet(
    const HiddenProjFpTilePacket& packet,
    ac_channel<HiddenProjFpWordPacket>& channel) {
  for (int word_index = 0; word_index < kHiddenProjFpWordCount; ++word_index) {
    HiddenProjFpWordPacket word_packet;
    load_hidden_proj_fp_word_packet(packet.data, word_index * kMaxFpWordsPerBeat, &word_packet);
    channel.write(word_packet);
  }
}

void write_hidden_proj_fp_tile_words(
    const prefill_catapult_fp_t* source,
    ac_channel<HiddenProjFpWordPacket>& channel) {
  for (int word_index = 0; word_index < kHiddenProjFpWordCount; ++word_index) {
    HiddenProjFpWordPacket word_packet;
    load_hidden_proj_fp_word_packet(source, word_index * kMaxFpWordsPerBeat, &word_packet);
    channel.write(word_packet);
  }
}

void read_hidden_proj_packed_weight_tile_packet(
    ac_channel<HiddenProjPackedWeightWordPacket>& channel,
    HiddenProjPackedWeightTilePacket* packet) {
  for (int word_index = 0; word_index < kHiddenProjPackedWordCount; ++word_index) {
    const HiddenProjPackedWeightWordPacket word_packet = channel.read();
    store_hidden_proj_packed_weight_word_packet(
        word_packet,
        packet->data,
        word_index * kMaxPackedWordsPerBeat);
  }
}

void project_hidden_token_tilewise_fp(
    const prefill_catapult_fp_t* input_token,
    const prefill_catapult_fp_t* input_layernorm_weight,
    const packed_w4_t* packed_weights,
    const prefill_catapult_fp_t* bias,
    const prefill_catapult_fp_t* scales,
    prefill_catapult_fp_t inv_rms,
    int tile_span,
    bool apply_rmsnorm,
    prefill_catapult_fp_t* output_token) {
#ifdef __SYNTHESIS__
  const int proj_tile = min_int(kProjectionTileCapacity, max_int(1, tile_span));

  for (int out_base = 0; out_base < kHiddenSize; out_base += proj_tile) {
    const int out_extent = min_int(proj_tile, kHiddenSize - out_base);
    HiddenProjScaleTilePacket scale_tile_packet;
    HiddenProjPartialTilePacket partial_sum_tile_packet;

    load_hidden_proj_scale_tile_packet(scales, out_base, out_extent, &scale_tile_packet);
    init_hidden_proj_partial_tile_packet(bias, out_base, out_extent, &partial_sum_tile_packet);

    for (int in_base = 0; in_base < kHiddenSize; in_base += proj_tile) {
      const int in_extent = min_int(proj_tile, kHiddenSize - in_base);
      HiddenProjFpTilePacket input_tile_packet;
      HiddenProjFpTilePacket layernorm_tile_packet;
      HiddenProjPackedWeightTilePacket packed_weight_tile_packet;

      load_hidden_proj_fp_tile_packet(input_token, in_base, in_extent, &input_tile_packet);
      if (apply_rmsnorm && input_layernorm_weight != nullptr) {
        load_hidden_proj_fp_tile_packet(input_layernorm_weight, in_base, in_extent, &layernorm_tile_packet);
      } else {
        init_hidden_proj_zero_tile_packet(&layernorm_tile_packet);
      }
      load_hidden_proj_packed_weight_tile_packet(
          packed_weights,
          out_base,
          out_extent,
          in_base,
          in_extent,
          &packed_weight_tile_packet);

      qwen_prefill_attention_hidden_proj_tile_array_core(
          input_tile_packet.data,
          layernorm_tile_packet.data,
          packed_weight_tile_packet.data,
          scale_tile_packet.data,
          partial_sum_tile_packet.data,
          inv_rms,
          in_extent,
          out_extent,
          apply_rmsnorm);
    }

    store_hidden_proj_partial_tile_packet(partial_sum_tile_packet, out_extent, output_token, out_base);
  }
#else
  (void)input_token;
  (void)input_layernorm_weight;
  (void)packed_weights;
  (void)bias;
  (void)scales;
  (void)inv_rms;
  (void)tile_span;
  (void)apply_rmsnorm;
  (void)output_token;
#endif
}

void qwen_prefill_attention_hidden_proj_tile_array_core(
    const prefill_catapult_fp_t input_tile[kProjectionTileCapacity],
    const prefill_catapult_fp_t input_layernorm_weight_tile[kProjectionTileCapacity],
    const packed_w4_t packed_weights_tile[kProjectionTileCapacity * kProjectionTileCapacity / 2],
    const prefill_catapult_fp_t scales_tile[kProjectionTileCapacity],
    prefill_catapult_fp_t partial_sum_tile[kProjectionTileCapacity],
    prefill_catapult_fp_t inv_rms,
    int lane_extent,
    int out_extent,
    bool apply_rmsnorm) {
#ifdef __SYNTHESIS__
  catapult_fp_t normalized_input_tile[kProjectionTileCapacity];

#pragma hls_unroll yes
  for (int lane = 0; lane < kProjectionTileCapacity; ++lane) {
    if (lane < lane_extent) {
      normalized_input_tile[lane] = apply_rmsnorm
          ? fp_mul_op(input_tile[lane], fp_mul_op(input_layernorm_weight_tile[lane], inv_rms))
          : input_tile[lane];
    } else {
      normalized_input_tile[lane] = fp_zero();
    }
  }

#pragma hls_unroll no
#pragma hls_pipeline_init_interval 2
  for (int out_offset = 0; out_offset < kProjectionTileCapacity; ++out_offset) {
    if (out_offset < out_extent) {
      partial_sum_tile[out_offset] = fp_add_op(
          partial_sum_tile[out_offset],
          weighted_chunk_dot_fp(
              normalized_input_tile,
              packed_weights_tile,
              scales_tile,
              out_offset,
              0,
              kProjectionTileCapacity,
              lane_extent));
    }
  }
#else
  (void)input_tile;
  (void)input_layernorm_weight_tile;
  (void)packed_weights_tile;
  (void)scales_tile;
  (void)partial_sum_tile;
  (void)inv_rms;
  (void)lane_extent;
  (void)out_extent;
  (void)apply_rmsnorm;
#endif
}

#pragma hls_pipeline_init_interval 1
#pragma hls_resource inv_rms:rsc variables="inv_rms" map_to_module="[DirectInput]"
#pragma hls_resource lane_extent:rsc variables="lane_extent" map_to_module="[DirectInput]"
#pragma hls_resource out_extent:rsc variables="out_extent" map_to_module="[DirectInput]"
#pragma hls_resource apply_rmsnorm:rsc variables="apply_rmsnorm" map_to_module="[DirectInput]"
void qwen_prefill_attention_q_context_output_tile_stream_catapult(
    ac_channel<HiddenProjFpWordPacket>& input_tile_chan,
    ac_channel<HiddenProjFpWordPacket>& input_layernorm_weight_tile_chan,
    ac_channel<HiddenProjPackedWeightWordPacket>& packed_weight_tile_chan,
    ac_channel<HiddenProjFpWordPacket>& scale_tile_chan,
    ac_channel<HiddenProjFpWordPacket>& partial_sum_tile_in_chan,
    prefill_catapult_fp_t inv_rms,
    int lane_extent,
    int out_extent,
    bool apply_rmsnorm,
    ac_channel<HiddenProjFpWordPacket>& partial_sum_tile_out_chan) {
  HiddenProjFpTilePacket input_tile_packet;
  HiddenProjFpTilePacket input_layernorm_weight_tile_packet;
  HiddenProjPackedWeightTilePacket packed_weight_tile_packet;
  HiddenProjScaleTilePacket scale_tile_packet;
  HiddenProjPartialTilePacket partial_sum_tile_packet;

  read_hidden_proj_fp_tile_packet(input_tile_chan, &input_tile_packet);
  read_hidden_proj_fp_tile_packet(input_layernorm_weight_tile_chan, &input_layernorm_weight_tile_packet);
  read_hidden_proj_packed_weight_tile_packet(packed_weight_tile_chan, &packed_weight_tile_packet);
  read_hidden_proj_fp_tile_words(scale_tile_chan, scale_tile_packet.data);
  read_hidden_proj_fp_tile_words(partial_sum_tile_in_chan, partial_sum_tile_packet.data);

  qwen_prefill_attention_hidden_proj_tile_array_core(
      input_tile_packet.data,
      input_layernorm_weight_tile_packet.data,
      packed_weight_tile_packet.data,
      scale_tile_packet.data,
      partial_sum_tile_packet.data,
      inv_rms,
      lane_extent,
      out_extent,
      apply_rmsnorm);

  write_hidden_proj_fp_tile_words(partial_sum_tile_packet.data, partial_sum_tile_out_chan);
}

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

void qwen_prefill_attention_input_norm_stage_catapult(
    const catapult_fp_t input_sequence[kPrefillSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t input_layernorm_weight[kHiddenSize],
    catapult_fp_t rms_eps,
    catapult_fp_t normalized_sequence[kPrefillSeqCapacity][llm_accel::kHiddenSize]) {
  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
ATTN_TOKEN_NORM_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      const int token_offset = token_index * llm_accel::kHiddenSize;
      rmsnorm_token_fp(input_sequence + token_offset, input_layernorm_weight, rms_eps, normalized_sequence[token_index]);
    }
  }
}

void qwen_prefill_attention_q_projection_stage_catapult(
    const catapult_fp_t normalized_sequence[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const packed_w4_t q_packed_weights[kHiddenSize * kHiddenSize / 2],
    const catapult_fp_t q_bias[kHiddenSize],
    const catapult_fp_t q_scales[kHiddenSize],
    catapult_fp_t q_proj_buffer[kPrefillSeqCapacity][llm_accel::kHiddenSize]) {
  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
ATTN_Q_PROJ_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      const catapult_fp_t* normalized_token = normalized_sequence[token_index];
      catapult_fp_t* q_proj_token = q_proj_buffer[token_index];
      project_hidden_token_bias_fp(
          normalized_token,
          q_packed_weights,
          q_bias,
          q_scales,
          tile_config.hidden_proj,
          tile_config.hidden_proj,
          q_proj_token);
    }
  }
}

void qwen_prefill_attention_k_projection_stage_catapult(
    const catapult_fp_t normalized_sequence[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const packed_w4_t k_packed_weights[kKvWidth * kHiddenSize / 2],
    const catapult_fp_t k_bias[kKvWidth],
    const catapult_fp_t k_scales[kKvWidth],
    catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth]) {
  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
ATTN_K_PROJ_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      const catapult_fp_t* normalized_token = normalized_sequence[token_index];
      catapult_fp_t* k_proj_token = k_cache + token_index * kKvWidth;
      project_kv_token_bias_fp(
          normalized_token,
          k_packed_weights,
          k_bias,
          k_scales,
          tile_config.kv_proj,
          tile_config.hidden_proj,
          k_proj_token);
    }
  }
}

void qwen_prefill_attention_v_projection_stage_catapult(
    const catapult_fp_t normalized_sequence[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const packed_w4_t v_packed_weights[kKvWidth * kHiddenSize / 2],
    const catapult_fp_t v_bias[kKvWidth],
    const catapult_fp_t v_scales[kKvWidth],
    catapult_fp_t v_cache[kPrefillSeqCapacity * kKvWidth]) {
  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
ATTN_V_PROJ_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      const catapult_fp_t* normalized_token = normalized_sequence[token_index];
      catapult_fp_t* v_proj_token = v_cache + token_index * kKvWidth;
      project_kv_token_bias_fp(
          normalized_token,
          v_packed_weights,
          v_bias,
          v_scales,
          tile_config.kv_proj,
          tile_config.hidden_proj,
          v_proj_token);
    }
  }
}

void qwen_prefill_attention_kv_projection_stage_catapult(
    const catapult_fp_t normalized_sequence[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const packed_w4_t k_packed_weights[kKvWidth * kHiddenSize / 2],
    const packed_w4_t v_packed_weights[kKvWidth * kHiddenSize / 2],
    const catapult_fp_t k_bias[kKvWidth],
    const catapult_fp_t v_bias[kKvWidth],
    const catapult_fp_t k_scales[kKvWidth],
    const catapult_fp_t v_scales[kKvWidth],
    catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth],
    catapult_fp_t v_cache[kPrefillSeqCapacity * kKvWidth]) {
  qwen_prefill_attention_k_projection_stage_catapult(
      normalized_sequence,
      seq_len,
      tile_config,
      k_packed_weights,
      k_bias,
      k_scales,
      k_cache);
  qwen_prefill_attention_v_projection_stage_catapult(
      normalized_sequence,
      seq_len,
      tile_config,
      v_packed_weights,
      v_bias,
      v_scales,
      v_cache);
}

void qwen_prefill_attention_qkv_projection_stage_catapult(
    const catapult_fp_t input_sequence[kPrefillSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t input_layernorm_weight[kHiddenSize],
    catapult_fp_t rms_eps,
    const packed_w4_t q_packed_weights[kHiddenSize * kHiddenSize / 2],
    const packed_w4_t k_packed_weights[kKvWidth * kHiddenSize / 2],
    const packed_w4_t v_packed_weights[kKvWidth * kHiddenSize / 2],
    const catapult_fp_t q_bias[kHiddenSize],
    const catapult_fp_t k_bias[kKvWidth],
    const catapult_fp_t v_bias[kKvWidth],
    const catapult_fp_t q_scales[kHiddenSize],
    const catapult_fp_t k_scales[kKvWidth],
    const catapult_fp_t v_scales[kKvWidth],
    catapult_fp_t q_proj_buffer[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth],
    catapult_fp_t v_cache[kPrefillSeqCapacity * kKvWidth]) {
  catapult_fp_t normalized_sequence[kPrefillSeqCapacity][llm_accel::kHiddenSize];

  qwen_prefill_attention_input_norm_stage_catapult(
      input_sequence,
      seq_len,
      tile_config,
      input_layernorm_weight,
      rms_eps,
      normalized_sequence);
  qwen_prefill_attention_q_projection_stage_catapult(
      normalized_sequence,
      seq_len,
      tile_config,
      q_packed_weights,
      q_bias,
      q_scales,
      q_proj_buffer);
  qwen_prefill_attention_k_projection_stage_catapult(
      normalized_sequence,
      seq_len,
      tile_config,
      k_packed_weights,
      k_bias,
      k_scales,
      k_cache);
  qwen_prefill_attention_v_projection_stage_catapult(
      normalized_sequence,
      seq_len,
      tile_config,
      v_packed_weights,
      v_bias,
      v_scales,
      v_cache);
}

void qwen_prefill_attention_kv_cache_stage_catapult(
    const catapult_fp_t input_sequence[kPrefillSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t input_layernorm_weight[kHiddenSize],
    catapult_fp_t rms_eps,
    const packed_w4_t k_packed_weights[kKvWidth * kHiddenSize / 2],
    const packed_w4_t v_packed_weights[kKvWidth * kHiddenSize / 2],
    const catapult_fp_t k_bias[kKvWidth],
    const catapult_fp_t v_bias[kKvWidth],
    const catapult_fp_t k_scales[kKvWidth],
    const catapult_fp_t v_scales[kKvWidth],
    catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth],
    catapult_fp_t v_cache[kPrefillSeqCapacity * kKvWidth]) {
  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
ATNN_KV_STREAM_LOOP:
#pragma hls_pipeline_init_interval 2
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      catapult_fp_t normalized_token[llm_accel::kHiddenSize];
      catapult_fp_t* k_proj_token = k_cache + token_index * kKvWidth;
      catapult_fp_t* v_proj_token = v_cache + token_index * kKvWidth;

      rmsnorm_token_fp(input_sequence + token_index * llm_accel::kHiddenSize, input_layernorm_weight, rms_eps, normalized_token);
      project_kv_token_bias_fp(
          normalized_token,
          k_packed_weights,
          k_bias,
          k_scales,
          tile_config.kv_proj,
          tile_config.hidden_proj,
          k_proj_token);
      project_kv_token_bias_fp(
          normalized_token,
          v_packed_weights,
          v_bias,
          v_scales,
          tile_config.kv_proj,
          tile_config.hidden_proj,
          v_proj_token);

      for (int head_base = 0; head_base < kNumKeyValueHeads; head_base += tile_config.kv_heads_parallel) {
        const int head_end = min_int(kNumKeyValueHeads, head_base + tile_config.kv_heads_parallel);
#pragma hls_unroll yes
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace_fp(k_proj_token + head * kHeadDim, token_index);
        }
      }
    }
  }
}

void qwen_prefill_attention_q_rope_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    catapult_fp_t q_proj_buffer[kPrefillSeqCapacity][llm_accel::kHiddenSize]) {
  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
ATTN_Q_ROPE_TOKEN_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      catapult_fp_t* q_proj_token = q_proj_buffer[token_index];

      for (int head_base = 0; head_base < kNumAttentionHeads; head_base += tile_config.query_heads_parallel) {
        const int head_end = min_int(kNumAttentionHeads, head_base + tile_config.query_heads_parallel);
#pragma hls_unroll yes
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace_fp(q_proj_token + head * kHeadDim, token_index);
        }
      }
    }
  }
}

void qwen_prefill_attention_k_rope_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth]) {
  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
ATTN_K_ROPE_TOKEN_LOOP:
#pragma hls_pipeline_init_interval 2
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      catapult_fp_t* k_proj_token = k_cache + token_index * kKvWidth;

      for (int head_base = 0; head_base < kNumKeyValueHeads; head_base += tile_config.kv_heads_parallel) {
        const int head_end = min_int(kNumKeyValueHeads, head_base + tile_config.kv_heads_parallel);
#pragma hls_unroll yes
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace_fp(k_proj_token + head * kHeadDim, token_index);
        }
      }
    }
  }
}

void qwen_prefill_attention_rope_apply_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    catapult_fp_t q_proj_buffer[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth]) {
  qwen_prefill_attention_q_rope_stage_catapult(seq_len, tile_config, q_proj_buffer);
  qwen_prefill_attention_k_rope_stage_catapult(seq_len, tile_config, k_cache);
}

void qwen_prefill_attention_qkv_rope_stage_catapult(
    const catapult_fp_t input_sequence[kPrefillSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t input_layernorm_weight[kHiddenSize],
    catapult_fp_t rms_eps,
    const packed_w4_t q_packed_weights[kHiddenSize * kHiddenSize / 2],
    const packed_w4_t k_packed_weights[kKvWidth * kHiddenSize / 2],
    const packed_w4_t v_packed_weights[kKvWidth * kHiddenSize / 2],
    const catapult_fp_t q_bias[kHiddenSize],
    const catapult_fp_t k_bias[kKvWidth],
    const catapult_fp_t v_bias[kKvWidth],
    const catapult_fp_t q_scales[kHiddenSize],
    const catapult_fp_t k_scales[kKvWidth],
    const catapult_fp_t v_scales[kKvWidth],
    catapult_fp_t q_proj_buffer[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth],
    catapult_fp_t v_cache[kPrefillSeqCapacity * kKvWidth]) {
  catapult_fp_t normalized_sequence[kPrefillSeqCapacity][llm_accel::kHiddenSize];

  qwen_prefill_attention_input_norm_stage_catapult(
      input_sequence,
      seq_len,
      tile_config,
      input_layernorm_weight,
      rms_eps,
      normalized_sequence);
  qwen_prefill_attention_q_projection_stage_catapult(
      normalized_sequence,
      seq_len,
      tile_config,
      q_packed_weights,
      q_bias,
      q_scales,
      q_proj_buffer);
    qwen_prefill_attention_k_projection_stage_catapult(
      normalized_sequence,
      seq_len,
      tile_config,
      k_packed_weights,
      k_bias,
      k_scales,
      k_cache);
    qwen_prefill_attention_v_projection_stage_catapult(
      normalized_sequence,
      seq_len,
      tile_config,
      v_packed_weights,
      v_bias,
      v_scales,
      v_cache);
    qwen_prefill_attention_q_rope_stage_catapult(seq_len, tile_config, q_proj_buffer);
    qwen_prefill_attention_k_rope_stage_catapult(seq_len, tile_config, k_cache);
}

#pragma hls_design ccore
#pragma hls_ccore_type sequential
void qwen_prefill_attention_context_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t q_proj_buffer[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    const catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth],
    const catapult_fp_t v_cache[kPrefillSeqCapacity * kKvWidth],
    catapult_fp_t context_buffer[kPrefillSeqCapacity][llm_accel::kHiddenSize]) {
  const int query_tile = max_int(1, tile_config.query);

  for (int query_begin = 0; query_begin < seq_len; query_begin += query_tile) {
    const int query_end = min_int(seq_len, query_begin + query_tile);
    prefill_attention_context_block_fp(
        q_proj_buffer,
        k_cache,
        v_cache,
        seq_len,
        query_begin,
        query_end,
        tile_config,
        context_buffer + query_begin);
  }
}

void qwen_prefill_attention_output_projection_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t context_buffer[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    const packed_w4_t o_packed_weights[kHiddenSize * kHiddenSize / 2],
    const catapult_fp_t o_scales[kHiddenSize],
    catapult_fp_t output_sequence[kPrefillSeqCapacity * kHiddenSize]) {
  const int query_tile = max_int(1, tile_config.query);

  for (int query_begin = 0; query_begin < seq_len; query_begin += query_tile) {
    const int query_end = min_int(seq_len, query_begin + query_tile);
ATTN_OUTPUT_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int query_index = query_begin; query_index < query_end; ++query_index) {
      const catapult_fp_t* context_token = context_buffer[query_index];
      catapult_fp_t* output_token = output_sequence + query_index * kHiddenSize;
      project_hidden_token_tilewise_fp(
          context_token,
          nullptr,
          o_packed_weights,
          nullptr,
          o_scales,
          fp_zero(),
          tile_config.hidden_proj,
          false,
          output_token);
    }
  }
}

void qwen_prefill_attention_context_output_stage_catapult(
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t q_proj_buffer[kPrefillSeqCapacity][llm_accel::kHiddenSize],
    const catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth],
    const catapult_fp_t v_cache[kPrefillSeqCapacity * kKvWidth],
    const packed_w4_t o_packed_weights[kHiddenSize * kHiddenSize / 2],
    const catapult_fp_t o_scales[kHiddenSize],
    catapult_fp_t output_sequence[kPrefillSeqCapacity * kHiddenSize]) {
  const int query_tile = max_int(1, tile_config.query);
  catapult_fp_t context_buffer[kPrefillQueryCapacity][llm_accel::kHiddenSize];

  for (int query_begin = 0; query_begin < seq_len; query_begin += query_tile) {
    const int query_end = min_int(seq_len, query_begin + query_tile);

    prefill_attention_context_block_fp(
        q_proj_buffer,
        k_cache,
        v_cache,
        seq_len,
        query_begin,
        query_end,
        tile_config,
        context_buffer);

ATTN_CONTEXT_OUTPUT_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int query_index = query_begin; query_index < query_end; ++query_index) {
      const catapult_fp_t* context_token = context_buffer[query_index - query_begin];
      catapult_fp_t* output_token = output_sequence + query_index * kHiddenSize;
      project_hidden_token_tilewise_fp(
          context_token,
          nullptr,
          o_packed_weights,
          nullptr,
          o_scales,
          fp_zero(),
          tile_config.hidden_proj,
          false,
          output_token);
    }
  }
}

void qwen_prefill_attention_q_context_output_stage_catapult(
    const catapult_fp_t input_sequence[kPrefillSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t input_layernorm_weight[kHiddenSize],
    catapult_fp_t rms_eps,
    const packed_w4_t q_packed_weights[kHiddenSize * kHiddenSize / 2],
    const catapult_fp_t q_bias[kHiddenSize],
    const catapult_fp_t q_scales[kHiddenSize],
    const catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth],
    const catapult_fp_t v_cache[kPrefillSeqCapacity * kKvWidth],
    const packed_w4_t o_packed_weights[kHiddenSize * kHiddenSize / 2],
    const catapult_fp_t o_scales[kHiddenSize],
    catapult_fp_t output_sequence[kPrefillSeqCapacity * kHiddenSize]) {
  const int query_tile = max_int(1, tile_config.query);

  for (int query_begin = 0; query_begin < seq_len; query_begin += query_tile) {
    const int query_end = min_int(seq_len, query_begin + query_tile);
    catapult_fp_t q_proj_tile[kPrefillQueryCapacity][llm_accel::kHiddenSize];
    catapult_fp_t context_tile[kPrefillQueryCapacity][llm_accel::kHiddenSize];

ATNN_Q_STREAM_LOOP:
#pragma hls_pipeline_init_interval 2
    for (int query_index = query_begin; query_index < query_end; ++query_index) {
      const catapult_fp_t* input_token = input_sequence + query_index * llm_accel::kHiddenSize;
      const catapult_fp_t mean_square = fp_mul_op(rmsnorm_square_sum_fp(input_token), fp_const(1.0f / 1536.0f));
      const catapult_fp_t inv_rms = approx_rsqrt_fp(fp_add_op(mean_square, rms_eps));
      catapult_fp_t* q_proj_token = q_proj_tile[query_index - query_begin];

      project_hidden_token_tilewise_fp(
          input_token,
          input_layernorm_weight,
          q_packed_weights,
          q_bias,
          q_scales,
          inv_rms,
          tile_config.hidden_proj,
          true,
          q_proj_token);

      for (int head_base = 0; head_base < kNumAttentionHeads; head_base += tile_config.query_heads_parallel) {
        const int head_end = min_int(kNumAttentionHeads, head_base + tile_config.query_heads_parallel);
#pragma hls_unroll yes
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace_fp(q_proj_token + head * kHeadDim, query_index);
        }
      }
    }

    prefill_attention_context_query_tile_fp(
        q_proj_tile,
        k_cache,
        v_cache,
        seq_len,
        query_begin,
        query_end,
        tile_config,
        context_tile);

ATNN_Q_CONTEXT_OUTPUT_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int query_index = query_begin; query_index < query_end; ++query_index) {
      const catapult_fp_t* context_token = context_tile[query_index - query_begin];
      catapult_fp_t* output_token = output_sequence + query_index * kHiddenSize;
      project_hidden_token_tilewise_fp(
          context_token,
          nullptr,
          o_packed_weights,
          nullptr,
          o_scales,
          fp_zero(),
          tile_config.hidden_proj,
          false,
          output_token);
    }
  }
}

KernelStatus qwen_prefill_attention_kernel_catapult(
    const catapult_fp_t input_sequence[kPrefillSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const catapult_fp_t input_layernorm_weight[kHiddenSize],
    catapult_fp_t rms_eps,
    const packed_w4_t q_packed_weights[kHiddenSize * kHiddenSize / 2],
    const packed_w4_t k_packed_weights[kKvWidth * kHiddenSize / 2],
    const packed_w4_t v_packed_weights[kKvWidth * kHiddenSize / 2],
    const packed_w4_t o_packed_weights[kHiddenSize * kHiddenSize / 2],
    const catapult_fp_t q_bias[kHiddenSize],
    const catapult_fp_t k_bias[kKvWidth],
    const catapult_fp_t v_bias[kKvWidth],
    const catapult_fp_t q_scales[kHiddenSize],
    const catapult_fp_t k_scales[kKvWidth],
    const catapult_fp_t v_scales[kKvWidth],
    const catapult_fp_t o_scales[kHiddenSize],
    catapult_fp_t k_cache[kPrefillSeqCapacity * kKvWidth],
    catapult_fp_t v_cache[kPrefillSeqCapacity * kKvWidth],
    catapult_fp_t output_sequence[kPrefillSeqCapacity * kHiddenSize]) {
  if (seq_len <= 0 || tile_config.seq <= 0 || tile_config.query <= 0 || tile_config.key <= 0 || tile_config.hidden_proj <= 0 ||
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

  qwen_prefill_attention_kv_cache_stage_catapult(
      input_sequence,
      seq_len,
      tile_config,
      input_layernorm_weight,
      rms_eps,
      k_packed_weights,
      v_packed_weights,
      k_bias,
      v_bias,
      k_scales,
      v_scales,
      k_cache,
      v_cache);
  qwen_prefill_attention_q_context_output_stage_catapult(
      input_sequence,
      seq_len,
      tile_config,
      input_layernorm_weight,
      rms_eps,
      q_packed_weights,
      q_bias,
      q_scales,
      k_cache,
      v_cache,
      o_packed_weights,
      o_scales,
      output_sequence);

  return {true, 0};
}

#endif

}  // namespace llm_accel