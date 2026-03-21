#include "qwen_prefill_attention_kernel.h"
#include <ac_channel.h>

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
constexpr int kMaxDdrPortBitWidth = 256;
constexpr int kCatapultFpBitWidth = 32;
constexpr int kPackedW4BitWidth = 8;
constexpr int kMaxFpWordsPerBeat = kMaxDdrPortBitWidth / kCatapultFpBitWidth;
constexpr int kMaxPackedWordsPerBeat = kMaxDdrPortBitWidth / kPackedW4BitWidth;
constexpr int kContextTokenWordCount = llm_accel::kHiddenSize / kMaxFpWordsPerBeat;
constexpr int kContextKvWordCount = kKvWidth / kMaxFpWordsPerBeat;
constexpr int kHeadWordCount = llm_accel::kHeadDim / kMaxFpWordsPerBeat;
constexpr int kContextHeadGroupSize = llm_accel::kDefaultPrefillAttentionQueryHeadsParallel;
constexpr int kContextHeadGroupScoreWordCount =
  (kContextHeadGroupSize + kMaxFpWordsPerBeat - 1) / kMaxFpWordsPerBeat;
constexpr int kScoreExportHeadGroupCount =
  llm_accel::kNumAttentionHeads / llm_accel::kDefaultPrefillAttentionQueryHeadsParallel;
constexpr int kScoreExportKeyTileCount =
  llm_accel::kDefaultPrefillSeqTile / llm_accel::kDefaultPrefillAttentionKeyTile;
constexpr int kHiddenProjFpWordCount = kProjectionTileCapacity / kMaxFpWordsPerBeat;
constexpr int kHiddenProjPackedTileSize = kProjectionTileCapacity * kProjectionTileCapacity / 2;
constexpr int kHiddenProjPackedWordCount = kHiddenProjPackedTileSize / kMaxPackedWordsPerBeat;

static_assert(
  llm_accel::kNumAttentionHeads % llm_accel::kDefaultPrefillAttentionQueryHeadsParallel == 0,
  "Score export top expects even head grouping.");
static_assert(
  llm_accel::kDefaultPrefillSeqTile % llm_accel::kDefaultPrefillAttentionKeyTile == 0,
  "Score export top expects even key tiling.");

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

float decode_int4_weight(llm_accel::packed_w4_t packed_value, bool high_nibble) {
  const int nibble = high_nibble ? static_cast<int>((packed_value >> 4) & 0xF) : static_cast<int>(packed_value & 0xF);
  const int signed_nibble = nibble >= 8 ? nibble - 16 : nibble;
  return static_cast<float>(signed_nibble);
}
#ifdef __SYNTHESIS__

using catapult_fp_t = llm_accel::prefill_catapult_fp_t;
using catapult_fp_bits_t = ac_int<kCatapultFpBitWidth, false>;

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

template <typename Values>
catapult_fp_t reduce_sum_128_fp(const Values& values);

template <typename Input, typename LaneSquare>
void rmsnorm_square_chunk_fp(
  const Input& input,
  int base_index,
  LaneSquare& lane_square);

template <typename Weight, typename ScaledWeight>
void rmsnorm_scale_weight_chunk_fp(
  const Weight& weight,
  const catapult_fp_t& inv_rms,
  int base_index,
  ScaledWeight& scaled_weight);

template <typename Input, typename ScaledWeight, typename Output>
void rmsnorm_apply_scale_chunk_fp(
  const Input& input,
  const ScaledWeight& scaled_weight,
  int base_index,
  Output& output);

template <typename PackedWeights, typename Scales>
catapult_fp_t dequantized_weight_fp(
  const PackedWeights& packed_weights,
  const Scales& scales,
  int out_index,
  int in_index,
  int in_dim);

template <typename InputTile, typename PackedWeights, typename Scales, typename LaneProducts>
void weighted_chunk_128_fp(
  const InputTile& input_tile,
  const PackedWeights& packed_weights,
  const Scales& scales,
  int out_index,
  int in_index_base,
  int in_dim,
  int lane_extent,
  LaneProducts& lane_products) {
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
  return llm_accel::prefill_catapult_fp_add(lhs, rhs);
}

inline catapult_fp_t fp_sub_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return llm_accel::prefill_catapult_fp_sub(lhs, rhs);
}

inline catapult_fp_t fp_mul_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return llm_accel::prefill_catapult_fp_mul(lhs, rhs);
}

inline catapult_fp_t fp_div_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return llm_accel::prefill_catapult_fp_div(lhs, rhs);
}

inline catapult_fp_t fp_mac_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs, const catapult_fp_t& acc) {
  return llm_accel::prefill_catapult_fp_mac(lhs, rhs, acc);
}

inline catapult_fp_t fp_sqrt_op(const catapult_fp_t& value) {
  return llm_accel::prefill_catapult_fp_sqrt(value);
}

inline bool fp_eq_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return llm_accel::prefill_catapult_fp_eq(lhs, rhs);
}

inline bool fp_lt_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return llm_accel::prefill_catapult_fp_lt(lhs, rhs);
}

inline bool fp_gt_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return llm_accel::prefill_catapult_fp_gt(lhs, rhs);
}

inline bool fp_le_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs) {
  return llm_accel::prefill_catapult_fp_le(lhs, rhs);
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

void approx_sincos_fp(const catapult_fp_t& angle, catapult_fp_t& sin_value, catapult_fp_t& cos_value) {
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
  sin_value = sin_poly;
  cos_value = fp_mul_op(cos_sign, cos_poly);
}

template <typename Input>
catapult_fp_t rmsnorm_square_sum_fp(
    const Input& input) {
  catapult_fp_t square_sum = fp_zero();
  for (int chunk_index = 0; chunk_index < kRmsNormChunkCount; ++chunk_index) {
    catapult_fp_t lane_square[kParallelMacLaneCount];
    rmsnorm_square_chunk_fp(input, chunk_index * kParallelMacLaneCount, lane_square);
    square_sum = fp_add_op(square_sum, reduce_sum_128_fp(lane_square));
  }
  return square_sum;
}

template <typename Weight, typename ScaledWeight>
void rmsnorm_scale_weight_fp(
    const Weight& weight,
    const catapult_fp_t& inv_rms,
    ScaledWeight& scaled_weight) {
  for (int chunk_index = 0; chunk_index < kRmsNormChunkCount; ++chunk_index) {
    rmsnorm_scale_weight_chunk_fp(weight, inv_rms, chunk_index * kParallelMacLaneCount, scaled_weight);
  }
}

template <typename Input, typename ScaledWeight, typename Output>
void rmsnorm_apply_scale_fp(
    const Input& input,
    const ScaledWeight& scaled_weight,
    Output& output) {
  for (int chunk_index = 0; chunk_index < kRmsNormChunkCount; ++chunk_index) {
    rmsnorm_apply_scale_chunk_fp(input, scaled_weight, chunk_index * kParallelMacLaneCount, output);
  }
}

template <typename Input, typename Weight, typename Output>
void rmsnorm_token_fp(
    const Input& input,
    const Weight& weight,
    const catapult_fp_t& rms_eps,
    Output& output) {
  catapult_fp_t scaled_weight[llm_accel::kHiddenSize];
  const catapult_fp_t mean_square = fp_mul_op(rmsnorm_square_sum_fp(input), fp_const(1.0f / 1536.0f));
  const catapult_fp_t inv_rms = approx_rsqrt_fp(fp_add_op(mean_square, rms_eps));
  rmsnorm_scale_weight_fp(weight, inv_rms, scaled_weight);
  rmsnorm_apply_scale_fp(input, scaled_weight, output);
}

template <typename Head>
void apply_rope_inplace_fp(Head head, int token_index) {
  catapult_fp_t inv_freq = fp_one();
  const catapult_fp_t token_index_fp = fp_const_int(token_index);
  const catapult_fp_t rope_step = fp_const(0.8058421877614801f);
#pragma hls_unroll yes
  for (int pair = 0; pair < llm_accel::kHeadDim / 2; ++pair) {
    const catapult_fp_t angle = fp_mul_op(token_index_fp, inv_freq);
    catapult_fp_t sinv = fp_zero();
    catapult_fp_t cosv = fp_one();
    approx_sincos_fp(angle, sinv, cosv);
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

template <typename PackedWeights, typename Scales>
catapult_fp_t dequantized_weight_fp(
    const PackedWeights& packed_weights,
    const Scales& scales,
    int out_index,
    int in_index,
    int in_dim) {
  const int flat_index = out_index * in_dim + in_index;
  const llm_accel::packed_w4_t packed_value = packed_weights[flat_index / 2];
  const bool high_nibble = (flat_index & 1) != 0;
  return fp_mul_op(decode_int4_weight_fp(packed_value, high_nibble), scales[out_index]);
}

template <typename Values>
catapult_fp_t reduce_sum_128_fp(const Values& values) {
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

template <typename Input, typename LaneSquare>
void rmsnorm_square_chunk_fp(
    const Input& input,
    int base_index,
    LaneSquare& lane_square) {
#pragma hls_unroll yes
  for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
    const catapult_fp_t value = input[base_index + lane];
    lane_square[lane] = fp_mul_op(value, value);
  }
}

template <typename Weight, typename ScaledWeight>
void rmsnorm_scale_weight_chunk_fp(
    const Weight& weight,
    const catapult_fp_t& inv_rms,
    int base_index,
    ScaledWeight& scaled_weight) {
#pragma hls_unroll yes
  for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
    scaled_weight[base_index + lane] = fp_mul_op(weight[base_index + lane], inv_rms);
  }
}

template <typename Input, typename ScaledWeight, typename Output>
void rmsnorm_apply_scale_chunk_fp(
    const Input& input,
    const ScaledWeight& scaled_weight,
    int base_index,
    Output& output) {
#pragma hls_unroll yes
  for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
    const int index = base_index + lane;
    output[index] = fp_mul_op(input[index], scaled_weight[index]);
  }
}
  inline catapult_fp_t add4_fp(
      const catapult_fp_t& v0,
      const catapult_fp_t& v1,
      const catapult_fp_t& v2,
      const catapult_fp_t& v3) {
    return fp_add_op(fp_add_op(v0, v1), fp_add_op(v2, v3));
  }

  inline catapult_fp_t add8_fp(
      const catapult_fp_t& v0,
      const catapult_fp_t& v1,
      const catapult_fp_t& v2,
      const catapult_fp_t& v3,
      const catapult_fp_t& v4,
      const catapult_fp_t& v5,
      const catapult_fp_t& v6,
      const catapult_fp_t& v7) {
    return fp_add_op(add4_fp(v0, v1, v2, v3), add4_fp(v4, v5, v6, v7));
  }

  inline catapult_fp_t add16_fp(
      const catapult_fp_t& v0,
      const catapult_fp_t& v1,
      const catapult_fp_t& v2,
      const catapult_fp_t& v3,
      const catapult_fp_t& v4,
      const catapult_fp_t& v5,
      const catapult_fp_t& v6,
      const catapult_fp_t& v7,
      const catapult_fp_t& v8,
      const catapult_fp_t& v9,
      const catapult_fp_t& v10,
      const catapult_fp_t& v11,
      const catapult_fp_t& v12,
      const catapult_fp_t& v13,
      const catapult_fp_t& v14,
      const catapult_fp_t& v15) {
    return fp_add_op(
        add8_fp(v0, v1, v2, v3, v4, v5, v6, v7),
        add8_fp(v8, v9, v10, v11, v12, v13, v14, v15));
  }

  template <typename Lhs, typename Rhs>
  inline catapult_fp_t dot_product_chunk_8_fp(const Lhs& lhs, const Rhs& rhs, int base_index) {
    const catapult_fp_t p0 = fp_mul_op(lhs[base_index], rhs[base_index]);
    const catapult_fp_t p1 = fp_mul_op(lhs[base_index + 1], rhs[base_index + 1]);
    const catapult_fp_t p2 = fp_mul_op(lhs[base_index + 2], rhs[base_index + 2]);
    const catapult_fp_t p3 = fp_mul_op(lhs[base_index + 3], rhs[base_index + 3]);
    const catapult_fp_t p4 = fp_mul_op(lhs[base_index + 4], rhs[base_index + 4]);
    const catapult_fp_t p5 = fp_mul_op(lhs[base_index + 5], rhs[base_index + 5]);
    const catapult_fp_t p6 = fp_mul_op(lhs[base_index + 6], rhs[base_index + 6]);
    const catapult_fp_t p7 = fp_mul_op(lhs[base_index + 7], rhs[base_index + 7]);

    return add8_fp(p0, p1, p2, p3, p4, p5, p6, p7);
  }

template <typename Lhs, typename Rhs>
catapult_fp_t dot_product_128_fp(const Lhs& lhs, const Rhs& rhs) {
  const catapult_fp_t chunk0 = dot_product_chunk_8_fp(lhs, rhs, 0);
  const catapult_fp_t chunk1 = dot_product_chunk_8_fp(lhs, rhs, 8);
  const catapult_fp_t chunk2 = dot_product_chunk_8_fp(lhs, rhs, 16);
  const catapult_fp_t chunk3 = dot_product_chunk_8_fp(lhs, rhs, 24);
  const catapult_fp_t chunk4 = dot_product_chunk_8_fp(lhs, rhs, 32);
  const catapult_fp_t chunk5 = dot_product_chunk_8_fp(lhs, rhs, 40);
  const catapult_fp_t chunk6 = dot_product_chunk_8_fp(lhs, rhs, 48);
  const catapult_fp_t chunk7 = dot_product_chunk_8_fp(lhs, rhs, 56);
  const catapult_fp_t chunk8 = dot_product_chunk_8_fp(lhs, rhs, 64);
  const catapult_fp_t chunk9 = dot_product_chunk_8_fp(lhs, rhs, 72);
  const catapult_fp_t chunk10 = dot_product_chunk_8_fp(lhs, rhs, 80);
  const catapult_fp_t chunk11 = dot_product_chunk_8_fp(lhs, rhs, 88);
  const catapult_fp_t chunk12 = dot_product_chunk_8_fp(lhs, rhs, 96);
  const catapult_fp_t chunk13 = dot_product_chunk_8_fp(lhs, rhs, 104);
  const catapult_fp_t chunk14 = dot_product_chunk_8_fp(lhs, rhs, 112);
  const catapult_fp_t chunk15 = dot_product_chunk_8_fp(lhs, rhs, 120);

  return add16_fp(
      chunk0,
      chunk1,
      chunk2,
      chunk3,
      chunk4,
      chunk5,
      chunk6,
      chunk7,
      chunk8,
      chunk9,
      chunk10,
      chunk11,
      chunk12,
      chunk13,
      chunk14,
      chunk15);
}

template <typename AccumHead>
inline void zero_head_accum_128_fp(AccumHead& accum_head) {
#pragma hls_unroll no
  for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
    accum_head[dim] = fp_zero();
  }
}

template <typename InputTile, typename PackedWeights, typename Scales>
catapult_fp_t weighted_chunk_dot_fp(
    const InputTile& input_tile,
    const PackedWeights& packed_weights,
    const Scales& scales,
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

template <typename Vector, typename Accum>
void scaled_accum_128_fp(const catapult_fp_t& scale, const Vector& vector, Accum& accum) {
#pragma hls_unroll no
  for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
    accum[dim] = fp_mac_op(scale, vector[dim], accum[dim]);
  }
}

template <typename Accum, typename Output>
void scale_store_128_fp(const Accum& accum, const catapult_fp_t& scale, Output& output) {
#pragma hls_unroll no
  for (int dim = 0; dim < llm_accel::kHeadDim; ++dim) {
    output[dim] = fp_mul_op(accum[dim], scale);
  }
}

struct ContextKvTokenPacket {
  catapult_fp_t k_data[kKvWidth];
  catapult_fp_t v_data[kKvWidth];
};

struct ContextKvHeadPacket {
  catapult_fp_t k_data[llm_accel::kHeadDim];
  catapult_fp_t v_data[llm_accel::kHeadDim];
};

struct ContextKTokenPacket {
  catapult_fp_t k_data[kKvWidth];
};

struct ContextVTokenPacket {
  catapult_fp_t v_data[kKvWidth];
};

struct ContextHeadGroupScorePacket {
  catapult_fp_t data[kContextHeadGroupSize];
};

struct ContextFpWordPacket {
  catapult_fp_t data[kMaxFpWordsPerBeat];
};

struct ContextFpBitsWordPacket {
  catapult_fp_bits_t data[kMaxFpWordsPerBeat];
};

struct ContextValueHeadStatePacket {
  catapult_fp_t denom[kContextHeadGroupSize];
  catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim];
};

struct ContextKeyTileMetaPacket {
  int key_begin;
  int key_end;
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

void stream_context_score_packet_words(
  const ContextHeadGroupScorePacket& max_score_packet,
  ac_channel<ContextFpWordPacket>& max_score_word_chan);

void read_context_score_packet_words(
  ac_channel<ContextFpWordPacket>& max_score_word_chan,
  ContextHeadGroupScorePacket& max_score_packet);

void init_context_value_head_state_packet(
    int head_base,
    int head_end,
    catapult_fp_t denom[kContextHeadGroupSize],
    catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim]);

void compute_context_score_packet(
    const ContextQueryPacket& q_packet,
    const ContextKvTokenPacket& kv_packet,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    ContextHeadGroupScorePacket& packet) {
#pragma hls_array_partition variable=q_packet.data cyclic factor=kMaxFpWordsPerBeat dim=1
#pragma hls_array_partition variable=kv_packet.k_data cyclic factor=kMaxFpWordsPerBeat dim=1
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
    if (head_base + head_offset < head_end) {
      const int head = head_base + head_offset;
      const int kv_head = head / kNumGroups;
      const catapult_fp_t* q_head = q_packet.data + head * llm_accel::kHeadDim;
      const catapult_fp_t* k_head = kv_packet.k_data + kv_head * llm_accel::kHeadDim;
      const catapult_fp_t score = dot_product_128_fp(q_head, k_head);
      packet.data[head_offset] = fp_mul_op(score, attention_scaling);
    }
  }
}

void compute_context_score_packet(
    const ContextQueryPacket& q_packet,
    const ContextKTokenPacket& k_packet,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    ContextHeadGroupScorePacket& packet) {
#pragma hls_array_partition variable=q_packet.data cyclic factor=kMaxFpWordsPerBeat dim=1
#pragma hls_array_partition variable=k_packet.k_data cyclic factor=kMaxFpWordsPerBeat dim=1
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
    if (head_base + head_offset < head_end) {
      const int head = head_base + head_offset;
      const int kv_head = head / kNumGroups;
      const catapult_fp_t* q_head = q_packet.data + head * llm_accel::kHeadDim;
      const catapult_fp_t* k_head = k_packet.k_data + kv_head * llm_accel::kHeadDim;
      const catapult_fp_t score = dot_product_128_fp(q_head, k_head);
      packet.data[head_offset] = fp_mul_op(score, attention_scaling);
    }
  }
}

void compute_context_score_packet(
    const ContextQueryPacket& q_packet,
    const ContextKvHeadPacket& kv_packet,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    ContextHeadGroupScorePacket& packet) {
#pragma hls_array_partition variable=q_packet.data cyclic factor=kMaxFpWordsPerBeat dim=1
#pragma hls_array_partition variable=kv_packet.k_data cyclic factor=kMaxFpWordsPerBeat dim=1
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
    if (head_base + head_offset < head_end) {
      const int head = head_base + head_offset;
      const catapult_fp_t* q_head = q_packet.data + head * llm_accel::kHeadDim;
      const catapult_fp_t score = dot_product_128_fp(q_head, kv_packet.k_data);
      packet.data[head_offset] = fp_mul_op(score, attention_scaling);
    }
  }
}

void update_context_max_score_packet(
    const ContextHeadGroupScorePacket& packet,
    int head_base,
    int head_end,
    ContextHeadGroupScorePacket& max_score_packet) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
    if (head_base + head_offset < head_end && fp_gt_op(packet.data[head_offset], max_score_packet.data[head_offset])) {
      max_score_packet.data[head_offset] = packet.data[head_offset];
    }
  }
}

void accumulate_context_weighted_value_packet(
  const ContextHeadGroupScorePacket& exp_score_packet,
    const ContextVTokenPacket& v_packet,
    int head_base,
    int head_end,
    catapult_fp_t denom[kContextHeadGroupSize],
    catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim]) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
    if (head_base + head_offset < head_end) {
      const int head = head_base + head_offset;
      const int kv_head = head / kNumGroups;
      const catapult_fp_t* v_head = v_packet.v_data + kv_head * llm_accel::kHeadDim;
      denom[head_offset] = fp_add_op(denom[head_offset], exp_score_packet.data[head_offset]);
      scaled_accum_128_fp(exp_score_packet.data[head_offset], v_head, accum[head_offset]);
    }
  }
}

void accumulate_context_weighted_value_packet(
  const ContextHeadGroupScorePacket& exp_score_packet,
    const ContextKvTokenPacket& kv_packet,
    int head_base,
    int head_end,
    catapult_fp_t denom[kContextHeadGroupSize],
    catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim]) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
    if (head_base + head_offset < head_end) {
      const int head = head_base + head_offset;
      const int kv_head = head / kNumGroups;
      const catapult_fp_t* v_head = kv_packet.v_data + kv_head * llm_accel::kHeadDim;
      denom[head_offset] = fp_add_op(denom[head_offset], exp_score_packet.data[head_offset]);
      scaled_accum_128_fp(exp_score_packet.data[head_offset], v_head, accum[head_offset]);
    }
  }
}

void accumulate_context_weighted_value_packet(
  const ContextHeadGroupScorePacket& exp_score_packet,
    const ContextKvHeadPacket& kv_packet,
    int head_base,
    int head_end,
    catapult_fp_t denom[kContextHeadGroupSize],
    catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim]) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
    if (head_base + head_offset < head_end) {
      denom[head_offset] = fp_add_op(denom[head_offset], exp_score_packet.data[head_offset]);
      scaled_accum_128_fp(exp_score_packet.data[head_offset], kv_packet.v_data, accum[head_offset]);
    }
  }
}

void stream_context_v_token_packet_words(
    const ContextVTokenPacket& v_packet,
    ac_channel<ContextFpWordPacket>& value_word_chan) {
  for (int word_index = 0; word_index < kContextKvWordCount; ++word_index) {
    ContextFpWordPacket word_packet;

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      word_packet.data[index] = v_packet.v_data[word_index * kMaxFpWordsPerBeat + index];
    }
    value_word_chan.write(word_packet);
  }
}

void stream_context_k_token_packet_words(
    const ContextKTokenPacket& k_packet,
    ac_channel<ContextFpWordPacket>& key_word_chan) {
  for (int word_index = 0; word_index < kContextKvWordCount; ++word_index) {
    ContextFpWordPacket word_packet;

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      word_packet.data[index] = k_packet.k_data[word_index * kMaxFpWordsPerBeat + index];
    }
    key_word_chan.write(word_packet);
  }
}

void read_context_k_token_packet_words(
    ac_channel<ContextFpWordPacket>& key_word_chan,
    ContextKTokenPacket& k_packet) {
  for (int word_index = 0; word_index < kContextKvWordCount; ++word_index) {
    const ContextFpWordPacket word_packet = key_word_chan.read();

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      k_packet.k_data[word_index * kMaxFpWordsPerBeat + index] = word_packet.data[index];
    }
  }
}

void read_context_v_token_packet_words(
    ac_channel<ContextFpWordPacket>& value_word_chan,
    ContextVTokenPacket& v_packet) {
  for (int word_index = 0; word_index < kContextKvWordCount; ++word_index) {
    const ContextFpWordPacket word_packet = value_word_chan.read();

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      v_packet.v_data[word_index * kMaxFpWordsPerBeat + index] = word_packet.data[index];
    }
  }
}

void read_context_kv_token_packet_words(
    ac_channel<ContextFpWordPacket>& key_word_chan,
    ac_channel<ContextFpWordPacket>& value_word_chan,
    ContextKvTokenPacket& kv_packet) {
  for (int word_index = 0; word_index < kContextKvWordCount; ++word_index) {
    const ContextFpWordPacket key_word_packet = key_word_chan.read();
    const ContextFpWordPacket value_word_packet = value_word_chan.read();

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      const int data_index = word_index * kMaxFpWordsPerBeat + index;
      kv_packet.k_data[data_index] = key_word_packet.data[index];
      kv_packet.v_data[data_index] = value_word_packet.data[index];
    }
  }
}

void read_context_kv_head_packet_words(
    ac_channel<ContextFpWordPacket>& key_word_chan,
    ac_channel<ContextFpWordPacket>& value_word_chan,
    ContextKvHeadPacket& kv_packet) {
  for (int word_index = 0; word_index < kHeadWordCount; ++word_index) {
    const ContextFpWordPacket key_word_packet = key_word_chan.read();
    const ContextFpWordPacket value_word_packet = value_word_chan.read();

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      const int data_index = word_index * kMaxFpWordsPerBeat + index;
      kv_packet.k_data[data_index] = key_word_packet.data[index];
      kv_packet.v_data[data_index] = value_word_packet.data[index];
    }
  }
}

void stream_context_value_key_packet_words(
    const ContextQueryPacket& q_packet,
    ac_channel<ContextFpWordPacket>& source_key_word_chan,
    ac_channel<ContextFpWordPacket>& source_value_word_chan,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    const ContextHeadGroupScorePacket& max_score_packet,
    ac_channel<ContextFpWordPacket>& exp_score_word_chan,
    ac_channel<ContextFpWordPacket>& value_word_chan) {
  ContextKTokenPacket k_packet;
  ContextVTokenPacket v_packet;
  ContextHeadGroupScorePacket score_packet;
  ContextHeadGroupScorePacket exp_score_packet;

  read_context_k_token_packet_words(source_key_word_chan, k_packet);
  read_context_v_token_packet_words(source_value_word_chan, v_packet);
  compute_context_score_packet(q_packet, k_packet, head_base, head_end, attention_scaling, score_packet);

#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
    if (head_base + head_offset < head_end) {
      exp_score_packet.data[head_offset] = approx_exp_fp(fp_sub_op(score_packet.data[head_offset], max_score_packet.data[head_offset]));
    } else {
      exp_score_packet.data[head_offset] = fp_zero();
    }
  }

  stream_context_score_packet_words(exp_score_packet, exp_score_word_chan);
  stream_context_v_token_packet_words(v_packet, value_word_chan);
}

void accumulate_context_value_key_packet_words(
    int head_base,
    int head_end,
    ac_channel<ContextFpWordPacket>& exp_score_word_chan,
    ac_channel<ContextFpWordPacket>& value_word_chan,
  catapult_fp_t denom[kContextHeadGroupSize],
  catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim]) {
  ContextHeadGroupScorePacket exp_score_packet;
  ContextVTokenPacket v_packet;

  read_context_score_packet_words(exp_score_word_chan, exp_score_packet);
  read_context_v_token_packet_words(value_word_chan, v_packet);
  accumulate_context_weighted_value_packet(
      exp_score_packet,
      v_packet,
      head_base,
      head_end,
      denom,
      accum);
}

void stream_context_value_key_tasks(
    const ContextQueryPacket& q_packet,
    int key_begin,
    int key_end,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    const ContextHeadGroupScorePacket& max_score_packet,
  ac_channel<ContextFpWordPacket>& source_key_word_chan,
  ac_channel<ContextFpWordPacket>& source_value_word_chan,
    ac_channel<ContextFpWordPacket>& exp_score_word_chan,
    ac_channel<ContextFpWordPacket>& value_word_chan) {
ATTN_CONTEXT_VALUE_WEIGHT_KEY_LOOP:
#pragma hls_pipeline_init_interval 2
  for (int key_index = key_begin; key_index < key_end; ++key_index) {
  (void)key_index;
    stream_context_value_key_packet_words(
        q_packet,
    source_key_word_chan,
    source_value_word_chan,
        head_base,
        head_end,
        attention_scaling,
        max_score_packet,
        exp_score_word_chan,
        value_word_chan);
  }
}

void accumulate_context_value_key_tasks(
    int key_begin,
    int key_end,
    int head_base,
    int head_end,
    ac_channel<ContextFpWordPacket>& exp_score_word_chan,
    ac_channel<ContextFpWordPacket>& value_word_chan,
    catapult_fp_t denom[kContextHeadGroupSize],
    catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim]) {
ATTN_CONTEXT_VALUE_ACCUM_KEY_LOOP:
#pragma hls_pipeline_init_interval 2
  for (int key_index = key_begin; key_index < key_end; ++key_index) {
    (void)key_index;
    accumulate_context_value_key_packet_words(
        head_base,
        head_end,
        exp_score_word_chan,
        value_word_chan,
        denom,
        accum);
  }
}

void process_context_max_score_tile(
    const ContextQueryPacket& q_packet,
    int key_begin,
    int key_end,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
    ac_channel<ContextFpWordPacket>& key_word_chan,
    ContextHeadGroupScorePacket& max_score_packet) {
ATTN_CONTEXT_MAX_KEY_LOOP:
#pragma hls_pipeline_init_interval 2
  for (int key_index = key_begin; key_index < key_end; ++key_index) {
    (void)key_index;
    ContextKTokenPacket k_packet;
    ContextHeadGroupScorePacket score_packet;

    read_context_k_token_packet_words(key_word_chan, k_packet);
    compute_context_score_packet(q_packet, k_packet, head_base, head_end, attention_scaling, score_packet);
    update_context_max_score_packet(score_packet, head_base, head_end, max_score_packet);
  }
}

void process_context_value_tile(
    const ContextQueryPacket& q_packet,
    int key_begin,
    int key_end,
    int head_base,
    int head_end,
    catapult_fp_t attention_scaling,
  const ContextHeadGroupScorePacket& max_score_packet,
    ac_channel<ContextFpWordPacket>& source_key_word_chan,
    ac_channel<ContextFpWordPacket>& source_value_word_chan,
    catapult_fp_t denom[kContextHeadGroupSize],
    catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim]) {
ATTN_CONTEXT_VALUE_TILE_LOOP:
#pragma hls_unroll no
#pragma hls_pipeline_init_interval 6
  for (int key_index = key_begin; key_index < key_end; ++key_index) {
    (void)key_index;
    ContextKvHeadPacket kv_packet;
    ContextHeadGroupScorePacket score_packet;
    ContextHeadGroupScorePacket exp_score_packet;

    read_context_kv_head_packet_words(source_key_word_chan, source_value_word_chan, kv_packet);
    compute_context_score_packet(q_packet, kv_packet, head_base, head_end, attention_scaling, score_packet);

#pragma hls_unroll yes
    for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
      if (head_base + head_offset < head_end) {
        exp_score_packet.data[head_offset] =
            approx_exp_fp(fp_sub_op(score_packet.data[head_offset], max_score_packet.data[head_offset]));
      } else {
        exp_score_packet.data[head_offset] = fp_zero();
      }
    }

    accumulate_context_weighted_value_packet(
        exp_score_packet,
        kv_packet,
        head_base,
        head_end,
        denom,
        accum);
  }
}

void compute_context_max_score_tile_tasks(
    const ContextQueryPacket& q_packet,
    int head_base,
    int head_end,
    int key_tile_count,
    ac_channel<ContextKeyTileMetaPacket>& key_tile_meta_chan,
    ac_channel<ContextFpWordPacket>& key_word_chan,
  ContextHeadGroupScorePacket& max_score_packet) {
  const catapult_fp_t attention_scaling = fp_const(0.08838834764831845f);

  for (int tile_slot = 0; tile_slot < key_tile_count; ++tile_slot) {
    const ContextKeyTileMetaPacket meta_packet = key_tile_meta_chan.read();
    process_context_max_score_tile(
        q_packet,
        meta_packet.key_begin,
        meta_packet.key_end,
        head_base,
        head_end,
        attention_scaling,
        key_word_chan,
        max_score_packet);
  }
}

void compute_context_value_tile_tasks(
    const ContextQueryPacket& q_packet,
    int head_base,
    int head_end,
    int key_tile_count,
    ac_channel<ContextKeyTileMetaPacket>& key_tile_meta_chan,
  const ContextHeadGroupScorePacket& max_score_packet,
    ac_channel<ContextFpWordPacket>& source_key_word_chan,
    ac_channel<ContextFpWordPacket>& source_value_word_chan,
    catapult_fp_t denom[kContextHeadGroupSize],
    catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim]) {
  const catapult_fp_t attention_scaling = fp_const(0.08838834764831845f);

  for (int tile_slot = 0; tile_slot < key_tile_count; ++tile_slot) {
    const ContextKeyTileMetaPacket meta_packet = key_tile_meta_chan.read();
    process_context_value_tile(
        q_packet,
        meta_packet.key_begin,
        meta_packet.key_end,
        head_base,
        head_end,
        attention_scaling,
        max_score_packet,
        source_key_word_chan,
        source_value_word_chan,
        denom,
        accum);
  }
}

void init_context_query_meta_packet(
    int query_index,
    int query_offset,
    ContextQueryMetaPacket& packet) {
  packet.query_index = query_index;
  packet.query_offset = query_offset;
}

void init_context_result_meta_packet(
    int query_offset,
    ContextResultMetaPacket& packet) {
  packet.query_offset = query_offset;
}

void init_context_key_tile_meta_packet(
    int key_begin,
    int key_end,
    ContextKeyTileMetaPacket& packet) {
  packet.key_begin = key_begin;
  packet.key_end = key_end;
}

int count_context_key_tiles(
    int seq_len,
    int query_index,
    int key_tile) {
  const int query_limit = min_int(seq_len, query_index + 1);
  return (query_limit + key_tile - 1) / key_tile;
}

void stream_context_key_tile_meta_packets(
    int seq_len,
    int query_index,
    int key_tile,
    ac_channel<ContextKeyTileMetaPacket>& key_tile_meta_chan) {
  for (int key_begin = 0; key_begin <= query_index && key_begin < seq_len; key_begin += key_tile) {
    ContextKeyTileMetaPacket meta_packet;
    const int query_limit = query_index + 1;
    const int key_end = min_int(seq_len, min_int(query_limit, key_begin + key_tile));

    init_context_key_tile_meta_packet(key_begin, key_end, meta_packet);
    key_tile_meta_chan.write(meta_packet);
  }
}

void stream_context_query_packet_words(
    const ContextQueryPacket& query_packet,
    ac_channel<ContextFpWordPacket>& query_word_chan) {
  for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
    ContextFpWordPacket word_packet;

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      word_packet.data[index] = query_packet.data[word_index * kMaxFpWordsPerBeat + index];
    }
    query_word_chan.write(word_packet);
  }
}

void read_context_query_packet_words(
    ac_channel<ContextFpWordPacket>& query_word_chan,
    ContextQueryPacket& query_packet) {
  for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
    const ContextFpWordPacket word_packet = query_word_chan.read();

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      query_packet.data[word_index * kMaxFpWordsPerBeat + index] = word_packet.data[index];
    }
  }
}

void stream_context_result_packet_words(
    const ContextTokenPacket& context_packet,
    ac_channel<ContextFpWordPacket>& context_word_chan) {
  for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
    ContextFpWordPacket word_packet;

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      word_packet.data[index] = context_packet.data[word_index * kMaxFpWordsPerBeat + index];
    }
    context_word_chan.write(word_packet);
  }
}

void read_context_result_packet_words(
    ac_channel<ContextFpWordPacket>& context_word_chan,
    ContextTokenPacket& context_packet) {
  for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
    const ContextFpWordPacket word_packet = context_word_chan.read();

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      context_packet.data[word_index * kMaxFpWordsPerBeat + index] = word_packet.data[index];
    }
  }
}

void stream_context_score_packet_words(
    const ContextHeadGroupScorePacket& max_score_packet,
    ac_channel<ContextFpWordPacket>& max_score_word_chan) {
  for (int word_index = 0; word_index < kContextHeadGroupScoreWordCount; ++word_index) {
    ContextFpWordPacket word_packet;

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      const int head_index = word_index * kMaxFpWordsPerBeat + index;
      word_packet.data[index] = head_index < kContextHeadGroupSize ? max_score_packet.data[head_index] : fp_zero();
    }

    max_score_word_chan.write(word_packet);
  }
}

void read_context_score_packet_words(
    ac_channel<ContextFpWordPacket>& max_score_word_chan,
    ContextHeadGroupScorePacket& max_score_packet) {
  for (int word_index = 0; word_index < kContextHeadGroupScoreWordCount; ++word_index) {
    const ContextFpWordPacket word_packet = max_score_word_chan.read();

#pragma hls_unroll yes
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      const int head_index = word_index * kMaxFpWordsPerBeat + index;
      if (head_index < kContextHeadGroupSize) {
        max_score_packet.data[head_index] = word_packet.data[index];
      }
    }
  }
}

void init_context_max_score_packet(
    int head_base,
    int head_end,
    ContextHeadGroupScorePacket& max_score_packet) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
    if (head_offset < head_end - head_base) {
      max_score_packet.data[head_offset] = fp_const(-1.0e30f);
    } else {
      max_score_packet.data[head_offset] = fp_zero();
    }
  }
}

void init_context_value_head_state_packet(
    int head_base,
    int head_end,
    catapult_fp_t denom[kContextHeadGroupSize],
    catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim]) {
#pragma hls_unroll yes
  for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
    if (head_offset < head_end - head_base) {
      denom[head_offset] = fp_zero();
      zero_head_accum_128_fp(accum[head_offset]);
    } else {
      denom[head_offset] = fp_zero();
      zero_head_accum_128_fp(accum[head_offset]);
    }
  }
}

void compute_context_max_score_head_state_packet(
    const ContextQueryPacket& q_packet,
    int seq_len,
    int query_index,
    int key_tile,
    int head_base,
    int head_end,
    ac_channel<ContextFpWordPacket>& key_word_chan,
  ContextHeadGroupScorePacket& max_score_packet) {
  const int query_limit = min_int(seq_len, query_index + 1);

  for (int key_begin = 0; key_begin < query_limit; key_begin += key_tile) {
    const int key_end = min_int(seq_len, min_int(query_limit, key_begin + key_tile));
    process_context_max_score_tile(
        q_packet,
        key_begin,
        key_end,
        head_base,
        head_end,
        fp_const(0.08838834764831845f),
        key_word_chan,
        max_score_packet);
  }
}

void compute_context_value_head_state_packet(
    const ContextQueryPacket& q_packet,
  const ContextHeadGroupScorePacket& max_score_packet,
    int seq_len,
    int query_index,
    int key_tile,
    int head_base,
    int head_end,
    ac_channel<ContextFpWordPacket>& source_key_word_chan,
    ac_channel<ContextFpWordPacket>& source_value_word_chan,
    catapult_fp_t denom[kContextHeadGroupSize],
    catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim]) {
  const int query_limit = min_int(seq_len, query_index + 1);

  for (int key_begin = 0; key_begin < query_limit; key_begin += key_tile) {
    const int key_end = min_int(seq_len, min_int(query_limit, key_begin + key_tile));
    process_context_value_tile(
        q_packet,
        key_begin,
        key_end,
        head_base,
        head_end,
        fp_const(0.08838834764831845f),
        max_score_packet,
        source_key_word_chan,
        source_value_word_chan,
        denom,
        accum);
  }
}

void store_context_head_state_packet(
    const catapult_fp_t denom[kContextHeadGroupSize],
    const catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim],
    int head_base,
    int head_end,
  ContextTokenPacket& context_packet) {
#pragma hls_unroll yes
  for (int head = head_base; head < head_end; ++head) {
    const int head_offset = head - head_base;
  catapult_fp_t* context_head = context_packet.data + head * llm_accel::kHeadDim;
    const catapult_fp_t inv_denom = fp_gt_op(denom[head_offset], fp_zero())
        ? approx_reciprocal_fp(denom[head_offset])
        : fp_zero();
    scale_store_128_fp(accum[head_offset], inv_denom, context_head);
  }
}

void stream_context_head_state_words(
    const catapult_fp_t denom[kContextHeadGroupSize],
    const catapult_fp_t accum[kContextHeadGroupSize][llm_accel::kHeadDim],
    int head_base,
    int head_end,
    ac_channel<ContextFpWordPacket>& context_word_chan) {
  for (int head = head_base; head < head_end; ++head) {
    const int head_offset = head - head_base;
    const catapult_fp_t inv_denom = fp_gt_op(denom[head_offset], fp_zero())
        ? approx_reciprocal_fp(denom[head_offset])
        : fp_zero();

    for (int word_index = 0; word_index < kHeadWordCount; ++word_index) {
      ContextFpWordPacket word_packet;

#pragma hls_unroll yes
      for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
        const int dim = word_index * kMaxFpWordsPerBeat + index;
        word_packet.data[index] = fp_mul_op(accum[head_offset][dim], inv_denom);
      }

      context_word_chan.write(word_packet);
    }
  }
}

void prefill_attention_context_max_score_head_group_stage_fp(
    const ContextQueryPacket& q_packet,
    int seq_len,
    int query_index,
    int key_tile,
    int head_base,
    int head_end,
    ac_channel<ContextFpWordPacket>& key_word_chan,
  ContextHeadGroupScorePacket& max_score_packet) {
  init_context_max_score_packet(head_base, head_end, max_score_packet);
  compute_context_max_score_head_state_packet(
      q_packet,
      seq_len,
      query_index,
      key_tile,
      head_base,
      head_end,
      key_word_chan,
      max_score_packet);
}

void prefill_attention_context_value_head_group_stage_fp(
    const ContextQueryPacket& q_packet,
    int seq_len,
    int query_index,
    int key_tile,
    int head_base,
    int head_end,
  const ContextHeadGroupScorePacket& max_score_packet,
    ac_channel<ContextFpWordPacket>& source_key_word_chan,
    ac_channel<ContextFpWordPacket>& source_value_word_chan,
    ac_channel<ContextFpWordPacket>& context_word_chan) {
  catapult_fp_t head_state_denom[kContextHeadGroupSize];
  catapult_fp_t head_state_accum[kContextHeadGroupSize][llm_accel::kHeadDim];

#pragma hls_array_partition variable=head_state_denom complete dim=1
#pragma hls_array_partition variable=head_state_accum complete dim=1
#pragma hls_array_partition variable=head_state_accum complete dim=2

  init_context_value_head_state_packet(head_base, head_end, head_state_denom, head_state_accum);
  compute_context_value_head_state_packet(
      q_packet,
      max_score_packet,
      seq_len,
      query_index,
      key_tile,
      head_base,
      head_end,
      source_key_word_chan,
      source_value_word_chan,
        head_state_denom,
        head_state_accum);
  stream_context_head_state_words(head_state_denom, head_state_accum, head_base, head_end, context_word_chan);
}

void prefill_attention_context_query_value_fp(
    const ContextQueryPacket& q_packet,
    int seq_len,
    int query_index,
    int key_tile,
    int query_heads_parallel,
    ac_channel<ContextFpWordPacket>& max_score_word_chan,
    ac_channel<ContextFpWordPacket>& source_key_word_chan,
    ac_channel<ContextFpWordPacket>& source_value_word_chan,
    ac_channel<ContextFpWordPacket>& context_word_chan) {
  for (int head_base = 0; head_base < llm_accel::kNumAttentionHeads; head_base += query_heads_parallel) {
    const int head_end = min_int(llm_accel::kNumAttentionHeads, head_base + query_heads_parallel);
    ContextHeadGroupScorePacket head_group_max_score_packet;

    read_context_score_packet_words(max_score_word_chan, head_group_max_score_packet);

    prefill_attention_context_value_head_group_stage_fp(
        q_packet,
        seq_len,
        query_index,
        key_tile,
        head_base,
        head_end,
        head_group_max_score_packet,
        source_key_word_chan,
        source_value_word_chan,
        context_word_chan);
  }
}

void stream_context_query_packet_word_channel(
    ac_channel<ContextFpWordPacket>& source_query_word_chan,
    ac_channel<ContextFpWordPacket>& destination_query_word_chan) {
  for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
    destination_query_word_chan.write(source_query_word_chan.read());
  }
}

void stream_context_score_word_channel(
    ac_channel<ContextFpWordPacket>& source_score_word_chan,
    ac_channel<ContextFpWordPacket>& destination_score_word_chan) {
  for (int head_group = 0; head_group < kScoreExportHeadGroupCount; ++head_group) {
    for (int word_index = 0; word_index < kContextHeadGroupScoreWordCount; ++word_index) {
      destination_score_word_chan.write(source_score_word_chan.read());
    }
  }
}

void stream_context_score_stage_key_words_for_query(
    int seq_len,
    int query_index,
    int key_tile,
    int query_heads_parallel,
  ac_channel<ContextFpWordPacket>& source_key_word_chan,
    ac_channel<ContextFpWordPacket>& key_word_chan) {
  for (int head_base = 0; head_base < llm_accel::kNumAttentionHeads; head_base += query_heads_parallel) {
    const int query_limit = min_int(seq_len, query_index + 1);
    for (int key_begin = 0; key_begin < query_limit; key_begin += key_tile) {
      const int key_end = min_int(query_limit, key_begin + key_tile);
      for (int key_index = key_begin; key_index < key_end; ++key_index) {
        (void)key_index;
        for (int word_index = 0; word_index < kContextKvWordCount; ++word_index) {
          key_word_chan.write(source_key_word_chan.read());
        }
      }
    }
  }
}

void stream_context_value_stage_kv_words_for_query(
    int seq_len,
    int query_index,
  int key_tile,
  int query_heads_parallel,
    ac_channel<ContextFpWordPacket>& source_key_word_chan,
    ac_channel<ContextFpWordPacket>& source_value_word_chan,
    ac_channel<ContextFpWordPacket>& key_word_chan,
    ac_channel<ContextFpWordPacket>& value_word_chan) {
  for (int head_base = 0; head_base < llm_accel::kNumAttentionHeads; head_base += query_heads_parallel) {
    const int query_limit = min_int(seq_len, query_index + 1);
    for (int key_begin = 0; key_begin < query_limit; key_begin += key_tile) {
      const int key_end = min_int(query_limit, key_begin + key_tile);
      for (int key_index = key_begin; key_index < key_end; ++key_index) {
        (void)key_index;
        const int kv_head = head_base / kNumGroups;
        for (int word_index = 0; word_index < kContextKvWordCount; ++word_index) {
          const ContextFpWordPacket key_word_packet = source_key_word_chan.read();
          const ContextFpWordPacket value_word_packet = source_value_word_chan.read();
          if (word_index / kHeadWordCount == kv_head) {
            key_word_chan.write(key_word_packet);
            value_word_chan.write(value_word_packet);
          }
        }
      }
    }
  }
}

#pragma hls_design block
void stream_context_score_stage_inputs(
    int seq_len,
    int query_count,
  int key_tile,
  int query_heads_parallel,
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan,
    ac_channel<ContextFpWordPacket>& source_key_word_chan,
    ac_channel<ContextQueryMetaPacket>& score_query_meta_chan,
    ac_channel<ContextFpWordPacket>& score_query_word_chan,
    ac_channel<ContextFpWordPacket>& score_key_word_chan) {
  const int normalized_key_tile = max_int(1, key_tile);
  const int normalized_query_heads_parallel = min_int(kContextHeadGroupSize, max_int(1, query_heads_parallel));

  for (int query_slot = 0; query_slot < query_count; ++query_slot) {
    const ContextQueryMetaPacket meta_packet = query_meta_chan.read();
    score_query_meta_chan.write(meta_packet);
    stream_context_query_packet_word_channel(query_word_chan, score_query_word_chan);
    stream_context_score_stage_key_words_for_query(
        seq_len,
        meta_packet.query_index,
        normalized_key_tile,
        normalized_query_heads_parallel,
      source_key_word_chan,
        score_key_word_chan);
  }
}

#pragma hls_design block
void stream_context_value_stage_inputs(
    int seq_len,
    int query_count,
  int key_tile,
  int query_heads_parallel,
    ac_channel<ContextQueryMetaPacket>& score_meta_chan,
    ac_channel<ContextFpWordPacket>& score_query_word_chan,
    ac_channel<ContextFpWordPacket>& max_score_word_chan,
    ac_channel<ContextFpWordPacket>& source_key_word_chan,
    ac_channel<ContextFpWordPacket>& source_value_word_chan,
    ac_channel<ContextQueryMetaPacket>& value_meta_chan,
    ac_channel<ContextFpWordPacket>& value_query_word_chan,
    ac_channel<ContextFpWordPacket>& value_max_score_word_chan,
    ac_channel<ContextFpWordPacket>& value_key_word_chan,
    ac_channel<ContextFpWordPacket>& value_source_word_chan) {
  const int normalized_key_tile = max_int(1, key_tile);
  const int normalized_query_heads_parallel = min_int(kContextHeadGroupSize, max_int(1, query_heads_parallel));

  for (int query_slot = 0; query_slot < query_count; ++query_slot) {
    const ContextQueryMetaPacket meta_packet = score_meta_chan.read();
    value_meta_chan.write(meta_packet);
    stream_context_query_packet_word_channel(score_query_word_chan, value_query_word_chan);
    stream_context_score_word_channel(max_score_word_chan, value_max_score_word_chan);
    stream_context_value_stage_kv_words_for_query(
        seq_len,
        meta_packet.query_index,
      normalized_key_tile,
      normalized_query_heads_parallel,
        source_key_word_chan,
        source_value_word_chan,
        value_key_word_chan,
        value_source_word_chan);
  }
}

void prefill_attention_context_score_stream_stage_fp_core(
    int seq_len,
    int query_count,
    int key_tile,
    int query_heads_parallel,
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan,
  ac_channel<ContextFpWordPacket>& key_word_chan,
    ac_channel<ContextQueryMetaPacket>& score_meta_chan,
    ac_channel<ContextFpWordPacket>& score_query_word_chan,
    ac_channel<ContextFpWordPacket>& max_score_word_chan) {
  const int normalized_key_tile = max_int(1, key_tile);
  const int normalized_query_heads_parallel = min_int(kContextHeadGroupSize, max_int(1, query_heads_parallel));
  const catapult_fp_t attention_scaling = fp_const(kAttentionScaling);

  for (int query_slot = 0; query_slot < query_count; ++query_slot) {
    const ContextQueryMetaPacket meta_packet = query_meta_chan.read();
    ContextFpWordPacket query_word_buffer[kContextTokenWordCount];

    for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
      const ContextFpWordPacket word_packet = query_word_chan.read();

      query_word_buffer[word_index] = word_packet;
      score_query_word_chan.write(word_packet);
    }

    const int query_limit = min_int(seq_len, meta_packet.query_index + 1);
    for (int head_base = 0; head_base < llm_accel::kNumAttentionHeads; head_base += normalized_query_heads_parallel) {
      const int head_end = min_int(llm_accel::kNumAttentionHeads, head_base + normalized_query_heads_parallel);
      ContextHeadGroupScorePacket head_group_max_score_packet;

#pragma hls_unroll yes
      for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
        head_group_max_score_packet.data[head_offset] =
            head_base + head_offset < head_end ? fp_const(-1.0e30f) : fp_zero();
      }
      for (int key_begin = 0; key_begin < query_limit; key_begin += normalized_key_tile) {
        const int key_end = min_int(query_limit, key_begin + normalized_key_tile);
ATTN_CONTEXT_SCORE_KEY_LOOP:
        for (int key_index = key_begin; key_index < key_end; ++key_index) {
          (void)key_index;
          catapult_fp_t partial_score[kContextHeadGroupSize];

#pragma hls_unroll yes
          for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
            partial_score[head_offset] = fp_zero();
          }

          for (int word_index = 0; word_index < kContextKvWordCount; ++word_index) {
            const ContextFpWordPacket word_packet = key_word_chan.read();
            const int kv_head = word_index / kHeadWordCount;
            const int head_word_offset = word_index % kHeadWordCount;

            for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
              const int head = head_base + head_offset;
              if (head < head_end && head / kNumGroups == kv_head) {
                const ContextFpWordPacket query_word_packet =
                    query_word_buffer[head * kHeadWordCount + head_word_offset];
                catapult_fp_t word_sum = fp_zero();

#pragma hls_unroll yes
                for (int lane = 0; lane < kMaxFpWordsPerBeat; ++lane) {
                  const catapult_fp_t q_value = query_word_packet.data[lane];
                  word_sum = fp_mac_op(q_value, word_packet.data[lane], word_sum);
                }
                partial_score[head_offset] = fp_add_op(partial_score[head_offset], word_sum);
              }
            }
          }

#pragma hls_unroll yes
          for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
            const int head = head_base + head_offset;
            if (head < head_end) {
              const catapult_fp_t score = fp_mul_op(partial_score[head_offset], attention_scaling);
              if (fp_gt_op(score, head_group_max_score_packet.data[head_offset])) {
                head_group_max_score_packet.data[head_offset] = score;
              }
            }
          }
        }
      }

      stream_context_score_packet_words(head_group_max_score_packet, max_score_word_chan);
    }

    score_meta_chan.write(meta_packet);
  }
}

void prefill_attention_context_score_stream_top_catapult(
    int seq_len,
    int query_count,
    const llm_accel::PrefillAttentionTileConfig& tile_config,
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan,
    ac_channel<ContextFpWordPacket>& key_word_chan,
    ac_channel<ContextQueryMetaPacket>& score_meta_chan,
    ac_channel<ContextFpWordPacket>& score_query_word_chan,
    ac_channel<ContextFpWordPacket>& max_score_word_chan) {
  prefill_attention_context_score_stream_stage_fp_core(
      seq_len,
      query_count,
      max_int(1, tile_config.key),
      max_int(1, tile_config.query_heads_parallel),
      query_meta_chan,
      query_word_chan,
      key_word_chan,
      score_meta_chan,
      score_query_word_chan,
      max_score_word_chan);
}

void prefill_attention_context_score_stream_rtl_export_top_core(
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan,
    ac_channel<ContextFpWordPacket>& key_word_chan,
    ac_channel<ContextQueryMetaPacket>& score_meta_chan,
    ac_channel<ContextFpWordPacket>& score_query_word_chan,
    ac_channel<ContextFpWordPacket>& max_score_word_chan) {
  const ContextQueryMetaPacket meta_packet = query_meta_chan.read();
  ContextFpBitsWordPacket query_word_buffer[kContextTokenWordCount];
  ContextFpBitsWordPacket key_word_buffer[kContextKvWordCount];
  const catapult_fp_t attention_scaling = fp_const(kAttentionScaling);

  for (int word_index = 0; word_index < kContextTokenWordCount; ++word_index) {
    const ContextFpWordPacket word_packet = query_word_chan.read();

#pragma hls_unroll yes
    for (int lane = 0; lane < kMaxFpWordsPerBeat; ++lane) {
      query_word_buffer[word_index].data[lane] = word_packet.data[lane].data_ac_int();
    }
    score_query_word_chan.write(word_packet);
  }

  score_meta_chan.write(meta_packet);

  for (int head_group = 0; head_group < kScoreExportHeadGroupCount; ++head_group) {
    const int head_base = head_group * llm_accel::kDefaultPrefillAttentionQueryHeadsParallel;
    ContextHeadGroupScorePacket head_group_max_score_packet;

#pragma hls_unroll yes
    for (int head_offset = 0; head_offset < kContextHeadGroupSize; ++head_offset) {
      head_group_max_score_packet.data[head_offset] = fp_const(-1.0e30f);
    }

    for (int key_tile = 0; key_tile < kScoreExportKeyTileCount; ++key_tile) {
      const int key_begin = key_tile * llm_accel::kDefaultPrefillAttentionKeyTile;
ATTN_CONTEXT_SCORE_EXPORT_KEY_LOOP:
      for (int key_offset = 0; key_offset < llm_accel::kDefaultPrefillAttentionKeyTile; ++key_offset) {
        const int key_index = key_begin + key_offset;
        (void)key_index;
        for (int word_index = 0; word_index < kContextKvWordCount; ++word_index) {
          const ContextFpWordPacket word_packet = key_word_chan.read();

#pragma hls_unroll yes
          for (int lane = 0; lane < kMaxFpWordsPerBeat; ++lane) {
            key_word_buffer[word_index].data[lane] = word_packet.data[lane].data_ac_int();
          }
        }

        for (int head_offset = 0;
             head_offset < llm_accel::kDefaultPrefillAttentionQueryHeadsParallel;
             ++head_offset) {
          const int head = head_base + head_offset;
          const int kv_head = head / kNumGroups;
          const int query_word_base = head * kHeadWordCount;
          const int key_word_base = kv_head * kHeadWordCount;
          catapult_fp_t partial_score = fp_zero();

          for (int head_word_offset = 0;
               head_word_offset < kHeadWordCount;
               ++head_word_offset) {
            const ContextFpBitsWordPacket query_word_packet =
                query_word_buffer[query_word_base + head_word_offset];
            const ContextFpBitsWordPacket key_word_packet =
                key_word_buffer[key_word_base + head_word_offset];
            catapult_fp_t word_sum = fp_zero();

            for (int lane = 0; lane < kMaxFpWordsPerBeat; ++lane) {
              catapult_fp_t q_value;
              catapult_fp_t k_value;

              q_value.set_data(query_word_packet.data[lane]);
              k_value.set_data(key_word_packet.data[lane]);
              word_sum = fp_mac_op(q_value, k_value, word_sum);
            }
            partial_score = fp_add_op(partial_score, word_sum);
          }

          const catapult_fp_t score = fp_mul_op(partial_score, attention_scaling);
          if (fp_gt_op(score, head_group_max_score_packet.data[head_offset])) {
            head_group_max_score_packet.data[head_offset] = score;
          }
        }
      }
    }

    stream_context_score_packet_words(head_group_max_score_packet, max_score_word_chan);
  }
}

#pragma hls_design block
void prefill_attention_context_score_stream_rtl_export_top_catapult(
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan,
    ac_channel<ContextFpWordPacket>& key_word_chan,
    ac_channel<ContextQueryMetaPacket>& score_meta_chan,
    ac_channel<ContextFpWordPacket>& score_query_word_chan,
    ac_channel<ContextFpWordPacket>& max_score_word_chan) {
  prefill_attention_context_score_stream_rtl_export_top_core(
      query_meta_chan,
      query_word_chan,
      key_word_chan,
      score_meta_chan,
      score_query_word_chan,
      max_score_word_chan);
}

#pragma hls_design block
void prefill_attention_context_score_stream_stage_fp(
    int seq_len,
    int query_count,
  int key_tile,
  int query_heads_parallel,
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan,
    ac_channel<ContextFpWordPacket>& key_word_chan,
    ac_channel<ContextQueryMetaPacket>& score_meta_chan,
    ac_channel<ContextFpWordPacket>& score_query_word_chan,
    ac_channel<ContextFpWordPacket>& max_score_word_chan) {
  prefill_attention_context_score_stream_stage_fp_core(
      seq_len,
      query_count,
      key_tile,
      query_heads_parallel,
      query_meta_chan,
      query_word_chan,
      key_word_chan,
      score_meta_chan,
      score_query_word_chan,
      max_score_word_chan);
}

#pragma hls_design block
void prefill_attention_context_value_stream_stage_fp(
    int seq_len,
    int query_count,
  int key_tile,
  int query_heads_parallel,
    ac_channel<ContextQueryMetaPacket>& score_meta_chan,
    ac_channel<ContextFpWordPacket>& score_query_word_chan,
    ac_channel<ContextFpWordPacket>& max_score_word_chan,
  ac_channel<ContextFpWordPacket>& source_key_word_chan,
  ac_channel<ContextFpWordPacket>& source_value_word_chan,
    ac_channel<ContextResultMetaPacket>& context_meta_chan,
    ac_channel<ContextFpWordPacket>& context_word_chan) {
  const int normalized_key_tile = max_int(1, key_tile);
  const int normalized_query_heads_parallel = min_int(kContextHeadGroupSize, max_int(1, query_heads_parallel));

  for (int query_slot = 0; query_slot < query_count; ++query_slot) {
    const ContextQueryMetaPacket meta_packet = score_meta_chan.read();
    ContextQueryPacket query_packet;
    ContextResultMetaPacket result_meta_packet;

    read_context_query_packet_words(score_query_word_chan, query_packet);
    prefill_attention_context_query_value_fp(
        query_packet,
        seq_len,
        meta_packet.query_index,
        normalized_key_tile,
        normalized_query_heads_parallel,
        max_score_word_chan,
        source_key_word_chan,
        source_value_word_chan,
        context_word_chan);
    init_context_result_meta_packet(meta_packet.query_offset, result_meta_packet);
    context_meta_chan.write(result_meta_packet);
  }
}

void qwen_prefill_attention_context_query_tile_stream_core(
    int seq_len,
    int query_count,
    int key_tile,
    int query_heads_parallel,
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan,
    ac_channel<ContextFpWordPacket>& score_stage_key_word_source_chan,
    ac_channel<ContextFpWordPacket>& value_stage_key_word_source_chan,
    ac_channel<ContextFpWordPacket>& value_stage_source_word_source_chan,
    ac_channel<ContextResultMetaPacket>& context_meta_chan,
    ac_channel<ContextFpWordPacket>& context_word_chan) {
      static ac_channel<ContextQueryMetaPacket> score_stage_query_meta_chan;
      static ac_channel<ContextFpWordPacket> score_stage_query_word_chan;
      static ac_channel<ContextFpWordPacket> score_stage_key_word_chan;
    static ac_channel<ContextQueryMetaPacket> score_meta_chan;
    static ac_channel<ContextFpWordPacket> score_query_word_chan;
    static ac_channel<ContextFpWordPacket> max_score_word_chan;
      static ac_channel<ContextQueryMetaPacket> value_stage_meta_chan;
      static ac_channel<ContextFpWordPacket> value_stage_query_word_chan;
      static ac_channel<ContextFpWordPacket> value_stage_max_score_word_chan;
      static ac_channel<ContextFpWordPacket> value_stage_key_word_chan;
      static ac_channel<ContextFpWordPacket> value_stage_source_word_chan;

      stream_context_score_stage_inputs(
        seq_len,
        query_count,
        key_tile,
        query_heads_parallel,
        query_meta_chan,
        query_word_chan,
        score_stage_key_word_source_chan,
        score_stage_query_meta_chan,
        score_stage_query_word_chan,
        score_stage_key_word_chan);

    prefill_attention_context_score_stream_stage_fp(
        seq_len,
        query_count,
      key_tile,
      query_heads_parallel,
        score_stage_query_meta_chan,
        score_stage_query_word_chan,
        score_stage_key_word_chan,
        score_meta_chan,
        score_query_word_chan,
        max_score_word_chan);
      stream_context_value_stage_inputs(
        seq_len,
        query_count,
        key_tile,
        query_heads_parallel,
        score_meta_chan,
        score_query_word_chan,
        max_score_word_chan,
        value_stage_key_word_source_chan,
        value_stage_source_word_source_chan,
        value_stage_meta_chan,
        value_stage_query_word_chan,
        value_stage_max_score_word_chan,
        value_stage_key_word_chan,
        value_stage_source_word_chan);
    prefill_attention_context_value_stream_stage_fp(
        seq_len,
        query_count,
      key_tile,
      query_heads_parallel,
        value_stage_meta_chan,
        value_stage_query_word_chan,
        value_stage_max_score_word_chan,
        value_stage_key_word_chan,
        value_stage_source_word_chan,
        context_meta_chan,
        context_word_chan);
  }

template <bool HasBias, int OutDim, int InDim, typename InputToken, typename PackedWeights, typename Bias, typename Scales, typename OutputToken>
void project_tiled_token_fp_impl(
  const InputToken& input_token,
  const PackedWeights& packed_weights,
  const Bias& bias,
  const Scales& scales,
    int out_tile,
    int in_tile,
  OutputToken& output) {
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

#endif

}  // namespace

#pragma hls_design top
#pragma hls_pipeline_init_interval 1
#pragma hls_resource seq_len:rsc variables="seq_len" map_to_module="[DirectInput]"
#pragma hls_resource query_count:rsc variables="query_count" map_to_module="[DirectInput]"
#pragma hls_resource key_tile:rsc variables="key_tile" map_to_module="[DirectInput]"
#pragma hls_resource query_heads_parallel:rsc variables="query_heads_parallel" map_to_module="[DirectInput]"
void qwen_prefill_attention_context_query_tile_stream_catapult(
    int seq_len,
    int query_count,
    int key_tile,
    int query_heads_parallel,
    ac_channel<ContextQueryMetaPacket>& query_meta_chan,
    ac_channel<ContextFpWordPacket>& query_word_chan,
    ac_channel<ContextFpWordPacket>& score_stage_key_word_source_chan,
    ac_channel<ContextFpWordPacket>& value_stage_key_word_source_chan,
    ac_channel<ContextFpWordPacket>& value_stage_source_word_source_chan,
    ac_channel<ContextResultMetaPacket>& context_meta_chan,
    ac_channel<ContextFpWordPacket>& context_word_chan) {
  qwen_prefill_attention_context_query_tile_stream_core(
      seq_len,
      query_count,
      key_tile,
      query_heads_parallel,
      query_meta_chan,
      query_word_chan,
      score_stage_key_word_source_chan,
      value_stage_key_word_source_chan,
      value_stage_source_word_source_chan,
      context_meta_chan,
      context_word_chan);
}

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

void read_hidden_proj_fp_tile_packet(
    ac_channel<HiddenProjFpWordPacket>& channel,
    HiddenProjFpTilePacket& packet) {
  for (int word_index = 0; word_index < kHiddenProjFpWordCount; ++word_index) {
    const HiddenProjFpWordPacket word_packet = channel.read();
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      packet.data[word_index * kMaxFpWordsPerBeat + index] = word_packet.data[index];
    }
  }
}

template <typename PacketType>
void read_hidden_proj_fp_tile_words(
    ac_channel<HiddenProjFpWordPacket>& channel,
    PacketType& packet) {
  for (int word_index = 0; word_index < kHiddenProjFpWordCount; ++word_index) {
    const HiddenProjFpWordPacket word_packet = channel.read();
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      packet.data[word_index * kMaxFpWordsPerBeat + index] = word_packet.data[index];
    }
  }
}

void write_hidden_proj_fp_tile_packet(
    const HiddenProjFpTilePacket& packet,
    ac_channel<HiddenProjFpWordPacket>& channel) {
  for (int word_index = 0; word_index < kHiddenProjFpWordCount; ++word_index) {
    HiddenProjFpWordPacket word_packet;
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      word_packet.data[index] = packet.data[word_index * kMaxFpWordsPerBeat + index];
    }
    channel.write(word_packet);
  }
}

template <typename PacketType>
void write_hidden_proj_fp_tile_words(
    const PacketType& packet,
    ac_channel<HiddenProjFpWordPacket>& channel) {
  for (int word_index = 0; word_index < kHiddenProjFpWordCount; ++word_index) {
    HiddenProjFpWordPacket word_packet;
    for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
      word_packet.data[index] = packet.data[word_index * kMaxFpWordsPerBeat + index];
    }
    channel.write(word_packet);
  }
}

void read_hidden_proj_packed_weight_tile_packet(
    ac_channel<HiddenProjPackedWeightWordPacket>& channel,
    HiddenProjPackedWeightTilePacket& packet) {
  for (int word_index = 0; word_index < kHiddenProjPackedWordCount; ++word_index) {
    const HiddenProjPackedWeightWordPacket word_packet = channel.read();
    for (int index = 0; index < kMaxPackedWordsPerBeat; ++index) {
      packet.data[word_index * kMaxPackedWordsPerBeat + index] = word_packet.data[index];
    }
  }
}

inline prefill_catapult_fp_t hidden_proj_input_value(
    prefill_catapult_fp_t input_value,
    const prefill_catapult_fp_t* input_layernorm_weight,
    int index,
    prefill_catapult_fp_t inv_rms,
    bool apply_rmsnorm) {
  if (apply_rmsnorm && input_layernorm_weight != 0) {
    input_value = input_value * inv_rms * input_layernorm_weight[index];
  }
  return input_value;
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
  prefill_catapult_fp_t input_tile[kProjectionTileCapacity];
  prefill_catapult_fp_t partial_sum[kProjectionTileCapacity];

  for (int out_base = 0; out_base < kHiddenSize; out_base += proj_tile) {
    const int out_extent = min_int(proj_tile, kHiddenSize - out_base);

    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      partial_sum[out_offset] = bias != 0 ? bias[out_base + out_offset] : prefill_catapult_fp_t(0.0f);
    }

    for (int in_base = 0; in_base < kHiddenSize; in_base += proj_tile) {
      const int in_extent = min_int(proj_tile, kHiddenSize - in_base);

      for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
        prefill_catapult_fp_t input_value = hidden_proj_input_value(
            input_token[in_base + in_offset],
            input_layernorm_weight,
            in_base + in_offset,
            inv_rms,
            apply_rmsnorm);
        input_tile[in_offset] = input_value;
      }

      for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
        const int out_index = out_base + out_offset;
        prefill_catapult_fp_t accum = partial_sum[out_offset];
        for (int lane_base = 0; lane_base < in_extent; lane_base += kParallelMacLaneCount) {
          const int lane_extent = min_int(kParallelMacLaneCount, in_extent - lane_base);
          accum = fp_add_op(
              accum,
              weighted_chunk_dot_fp(
                  input_tile + lane_base,
                  packed_weights,
                  scales,
                  out_index,
                  in_base + lane_base,
                  kHiddenSize,
                  lane_extent));
        }
        partial_sum[out_offset] = accum;
      }
    }

    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      output_token[out_base + out_offset] = partial_sum[out_offset];
    }
  }
#else
  const int proj_tile = min_int(kProjectionTileCapacity, max_int(1, tile_span));

  for (int out_base = 0; out_base < kHiddenSize; out_base += proj_tile) {
    const int out_extent = min_int(proj_tile, kHiddenSize - out_base);

    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      const int out_index = out_base + out_offset;
      prefill_catapult_fp_t accum = bias != 0 ? bias[out_index] : prefill_catapult_fp_t(0.0f);
      for (int in_index = 0; in_index < kHiddenSize; ++in_index) {
        prefill_catapult_fp_t input_value = hidden_proj_input_value(
            input_token[in_index],
            input_layernorm_weight,
            in_index,
            inv_rms,
            apply_rmsnorm);
        const int flat_index = out_index * kHiddenSize + in_index;
        const packed_w4_t packed_value = packed_weights[flat_index / 2];
        const bool high_nibble = (flat_index & 1) != 0;
        accum += input_value * decode_int4_weight(packed_value, high_nibble) * scales[out_index];
      }
      output_token[out_index] = accum;
    }
  }
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

  read_hidden_proj_fp_tile_packet(input_tile_chan, input_tile_packet);
  read_hidden_proj_fp_tile_packet(input_layernorm_weight_tile_chan, input_layernorm_weight_tile_packet);
  read_hidden_proj_packed_weight_tile_packet(packed_weight_tile_chan, packed_weight_tile_packet);
  read_hidden_proj_fp_tile_words(scale_tile_chan, scale_tile_packet);
  read_hidden_proj_fp_tile_words(partial_sum_tile_in_chan, partial_sum_tile_packet);

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

  write_hidden_proj_fp_tile_words(partial_sum_tile_packet, partial_sum_tile_out_chan);
}

#ifdef __SYNTHESIS__

void qwen_prefill_attention_kv_cache_stage_catapult(
  CatapultConstTensorView<catapult_fp_t> input_sequence,
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
  CatapultConstTensorView<catapult_fp_t> input_layernorm_weight,
    catapult_fp_t rms_eps,
  CatapultConstTensorView<packed_w4_t> k_packed_weights,
  CatapultConstTensorView<packed_w4_t> v_packed_weights,
  CatapultConstTensorView<catapult_fp_t> k_bias,
  CatapultConstTensorView<catapult_fp_t> v_bias,
  CatapultConstTensorView<catapult_fp_t> k_scales,
  CatapultConstTensorView<catapult_fp_t> v_scales,
  CatapultTensorView<catapult_fp_t> k_cache,
  CatapultTensorView<catapult_fp_t> v_cache) {
  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
ATNN_KV_STREAM_LOOP:
#pragma hls_pipeline_init_interval 2
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      catapult_fp_t normalized_token[llm_accel::kHiddenSize];
      catapult_fp_t* k_proj_token = k_cache.data + token_index * kKvWidth;
      catapult_fp_t* v_proj_token = v_cache.data + token_index * kKvWidth;

      rmsnorm_token_fp(input_sequence.data + token_index * llm_accel::kHiddenSize, input_layernorm_weight.data, rms_eps, normalized_token);
        project_tiled_token_fp_impl<true, kKvWidth, llm_accel::kHiddenSize>(
          normalized_token,
          k_packed_weights.data,
          k_bias.data,
          k_scales.data,
          tile_config.kv_proj,
          tile_config.hidden_proj,
          k_proj_token);
        project_tiled_token_fp_impl<true, kKvWidth, llm_accel::kHiddenSize>(
          normalized_token,
          v_packed_weights.data,
          v_bias.data,
          v_scales.data,
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

void qwen_prefill_attention_q_context_output_stage_catapult(
  CatapultConstTensorView<catapult_fp_t> input_sequence,
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
  CatapultConstTensorView<catapult_fp_t> input_layernorm_weight,
    catapult_fp_t rms_eps,
  CatapultConstTensorView<packed_w4_t> q_packed_weights,
  CatapultConstTensorView<catapult_fp_t> q_bias,
  CatapultConstTensorView<catapult_fp_t> q_scales,
  CatapultConstTensorView<catapult_fp_t> k_cache,
  CatapultConstTensorView<catapult_fp_t> v_cache,
  CatapultConstTensorView<packed_w4_t> o_packed_weights,
  CatapultConstTensorView<catapult_fp_t> o_scales,
  CatapultTensorView<catapult_fp_t> output_sequence) {
  const int query_tile = max_int(1, tile_config.query);

  for (int query_begin = 0; query_begin < seq_len; query_begin += query_tile) {
    const int query_end = min_int(seq_len, query_begin + query_tile);
    static ac_channel<ContextQueryMetaPacket> query_meta_chan;
    static ac_channel<ContextFpWordPacket> query_word_chan;
    static ac_channel<ContextFpWordPacket> score_stage_key_word_source_chan;
    static ac_channel<ContextFpWordPacket> value_stage_key_word_source_chan;
    static ac_channel<ContextFpWordPacket> value_stage_source_word_source_chan;
    static ac_channel<ContextResultMetaPacket> context_meta_chan;
    static ac_channel<ContextFpWordPacket> context_word_chan;
    catapult_fp_t q_proj_tile[kPrefillQueryCapacity][llm_accel::kHiddenSize];
    catapult_fp_t context_tile[kPrefillQueryCapacity][llm_accel::kHiddenSize];
    const int query_count = query_end - query_begin;

Q_CONTEXT_Q_PROJ_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int query_index = query_begin; query_index < query_end; ++query_index) {
      const catapult_fp_t* input_token = input_sequence.data + query_index * llm_accel::kHiddenSize;
      const catapult_fp_t mean_square = fp_mul_op(rmsnorm_square_sum_fp(input_token), fp_const(1.0f / 1536.0f));
      const catapult_fp_t inv_rms = approx_rsqrt_fp(fp_add_op(mean_square, rms_eps));
      catapult_fp_t* q_proj_token = q_proj_tile[query_index - query_begin];

      project_hidden_token_tilewise_fp(
          input_token,
          input_layernorm_weight.data,
          q_packed_weights.data,
          q_bias.data,
          q_scales.data,
          inv_rms,
          tile_config.hidden_proj,
          true,
          q_proj_token);

      const int query_heads_parallel = min_int(kContextHeadGroupSize, max_int(1, tile_config.query_heads_parallel));
      for (int head_base = 0; head_base < kNumAttentionHeads; head_base += query_heads_parallel) {
        const int head_end = min_int(kNumAttentionHeads, head_base + query_heads_parallel);
#pragma hls_unroll yes
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace_fp(q_proj_token + head * kHeadDim, query_index);
        }
      }

      ContextQueryPacket q_packet;
      ContextQueryMetaPacket meta_packet;
      const int query_offset = query_index - query_begin;

#pragma hls_unroll yes
      for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
        q_packet.data[dim] = q_proj_token[dim];
      }
      init_context_query_meta_packet(query_index, query_offset, meta_packet);
      query_meta_chan.write(meta_packet);
      stream_context_query_packet_words(q_packet, query_word_chan);

      const int key_tile = max_int(1, tile_config.key);
      const int query_limit = min_int(seq_len, query_index + 1);
      for (int head_base = 0; head_base < llm_accel::kNumAttentionHeads; head_base += query_heads_parallel) {
        (void)head_base;
        for (int key_begin = 0; key_begin < query_limit; key_begin += key_tile) {
          const int key_end = min_int(query_limit, key_begin + key_tile);
          for (int key_index = key_begin; key_index < key_end; ++key_index) {
            ContextKTokenPacket k_packet;
            ContextVTokenPacket v_packet;
#pragma hls_unroll yes
            for (int dim = 0; dim < kKvWidth; ++dim) {
              k_packet.k_data[dim] = k_cache.data[key_index * kKvWidth + dim];
              v_packet.v_data[dim] = v_cache.data[key_index * kKvWidth + dim];
            }
            stream_context_k_token_packet_words(k_packet, score_stage_key_word_source_chan);
            stream_context_k_token_packet_words(k_packet, value_stage_key_word_source_chan);
            stream_context_v_token_packet_words(v_packet, value_stage_source_word_source_chan);
          }
        }
      }
    }
    qwen_prefill_attention_context_query_tile_stream_catapult(
        seq_len,
        query_count,
      max_int(1, tile_config.key),
      min_int(kContextHeadGroupSize, max_int(1, tile_config.query_heads_parallel)),
        query_meta_chan,
        query_word_chan,
        score_stage_key_word_source_chan,
        value_stage_key_word_source_chan,
        value_stage_source_word_source_chan,
        context_meta_chan,
        context_word_chan);
    for (int query_slot = 0; query_slot < query_count; ++query_slot) {
      const ContextResultMetaPacket meta_packet = context_meta_chan.read();
      ContextTokenPacket context_packet;
      read_context_result_packet_words(context_word_chan, context_packet);
#pragma hls_unroll yes
      for (int dim = 0; dim < llm_accel::kHiddenSize; ++dim) {
        context_tile[meta_packet.query_offset][dim] = context_packet.data[dim];
      }
    }

Q_CONTEXT_OUTPUT_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int query_index = query_begin; query_index < query_end; ++query_index) {
      const catapult_fp_t* context_token = context_tile[query_index - query_begin];
      catapult_fp_t* output_token = output_sequence.data + query_index * kHiddenSize;
      project_hidden_token_tilewise_fp(
          context_token,
          nullptr,
          o_packed_weights.data,
          nullptr,
          o_scales.data,
          fp_zero(),
          tile_config.hidden_proj,
          false,
          output_token);
    }
  }
}

KernelStatus qwen_prefill_attention_kernel_catapult(
  CatapultConstTensorView<catapult_fp_t> input_sequence,
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
  CatapultConstTensorView<catapult_fp_t> input_layernorm_weight,
    catapult_fp_t rms_eps,
  CatapultConstTensorView<packed_w4_t> q_packed_weights,
  CatapultConstTensorView<packed_w4_t> k_packed_weights,
  CatapultConstTensorView<packed_w4_t> v_packed_weights,
  CatapultConstTensorView<packed_w4_t> o_packed_weights,
  CatapultConstTensorView<catapult_fp_t> q_bias,
  CatapultConstTensorView<catapult_fp_t> k_bias,
  CatapultConstTensorView<catapult_fp_t> v_bias,
  CatapultConstTensorView<catapult_fp_t> q_scales,
  CatapultConstTensorView<catapult_fp_t> k_scales,
  CatapultConstTensorView<catapult_fp_t> v_scales,
  CatapultConstTensorView<catapult_fp_t> o_scales,
  CatapultTensorView<catapult_fp_t> k_cache,
  CatapultTensorView<catapult_fp_t> v_cache,
  CatapultTensorView<catapult_fp_t> output_sequence) {
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
      {input_sequence.data},
      seq_len,
      tile_config,
      {input_layernorm_weight.data},
      rms_eps,
      {k_packed_weights.data},
      {v_packed_weights.data},
      {k_bias.data},
      {v_bias.data},
      {k_scales.data},
      {v_scales.data},
      {k_cache.data},
      {v_cache.data});
  qwen_prefill_attention_q_context_output_stage_catapult(
      {input_sequence.data},
      seq_len,
      tile_config,
      {input_layernorm_weight.data},
      rms_eps,
      {q_packed_weights.data},
      {q_bias.data},
      {q_scales.data},
      {k_cache.data},
      {v_cache.data},
      {o_packed_weights.data},
      {o_scales.data},
      {output_sequence.data});

  return {true, 0};
}

#endif

}  // namespace llm_accel
