#include "../common/llm_accel_types.h"
#include "../common/qwen2_model_config.h"
#include "../include/ac_channel.h"
#include "qwen_catapult_fp.h"

#ifdef __SYNTHESIS__
#include "../include/ac_int.h"
#include "../include/ac_std_float.h"
#include "../include/ccs_dw_fp_lib.h"
#endif

namespace {

constexpr int kProjectionTileCapacity = llm_accel::kAttentionMacCols;
constexpr int kProjectionPackedTileSize = kProjectionTileCapacity * kProjectionTileCapacity / 2;

#ifdef __SYNTHESIS__

using catapult_fp_t = llm_accel::prefill_catapult_fp_t;

constexpr int kFpIeeeCompliance = 0;
constexpr int kKvWidth = llm_accel::kNumKeyValueHeads * llm_accel::kHeadDim;
constexpr int kParallelMacLaneCount = llm_accel::kAttentionMacCols;
constexpr int kRmsNormChunkCount = llm_accel::kHiddenSize / kParallelMacLaneCount;

#define HLS_DO_1(M, base) M(base)
#define HLS_DO_2(M, base) HLS_DO_1(M, base) M((base) + 1)
#define HLS_DO_4(M, base) HLS_DO_2(M, base) HLS_DO_2(M, (base) + 2)
#define HLS_DO_8(M, base) HLS_DO_4(M, base) HLS_DO_4(M, (base) + 4)
#define HLS_DO_16(M, base) HLS_DO_8(M, base) HLS_DO_8(M, (base) + 8)
#define HLS_DO_32(M, base) HLS_DO_16(M, base) HLS_DO_16(M, (base) + 16)
#define HLS_DO_64(M, base) HLS_DO_32(M, base) HLS_DO_32(M, (base) + 32)

inline int min_int(int lhs, int rhs) {
  return lhs < rhs ? lhs : rhs;
}

inline int max_int(int lhs, int rhs) {
  return lhs > rhs ? lhs : rhs;
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

catapult_fp_t reduce_sum_64_fp(catapult_fp_t values[kParallelMacLaneCount]) {
  catapult_fp_t stage32[kParallelMacLaneCount / 2];
  catapult_fp_t stage16[kParallelMacLaneCount / 4];
  catapult_fp_t stage8[kParallelMacLaneCount / 8];
  catapult_fp_t stage4[kParallelMacLaneCount / 16];
  catapult_fp_t stage2[kParallelMacLaneCount / 32];

#define REDUCE_STAGE32(i) stage32[i] = fp_add_op(values[(i) * 2], values[(i) * 2 + 1]);
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

catapult_fp_t decode_int4_weight_fp(llm_accel::packed_w4_t packed_value, bool high_nibble) {
  const int nibble = high_nibble ? static_cast<int>((packed_value >> 4) & 0xF) : static_cast<int>(packed_value & 0xF);
  const int signed_nibble = nibble >= 8 ? nibble - 16 : nibble;
  return fp_const_int(signed_nibble);
}

int extract_int4_weight_nibble(llm_accel::packed_w4_t packed_value, bool high_nibble) {
  return high_nibble ? static_cast<int>((packed_value >> 4) & 0xF) : static_cast<int>(packed_value & 0xF);
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

catapult_fp_t weighted_chunk_dot_64_fp(
    const catapult_fp_t* input_tile,
    const llm_accel::packed_w4_t* packed_weights,
    const catapult_fp_t* scales,
    int out_index,
    int in_index_base,
    int in_dim,
    int lane_extent) {
  catapult_fp_t lane_products[kParallelMacLaneCount];

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

  return reduce_sum_64_fp(lane_products);
}

template <bool HasBias>
void project_kv_token_fp_impl(
    const catapult_fp_t input_token[llm_accel::kHiddenSize],
    const llm_accel::packed_w4_t packed_weights[kKvWidth * llm_accel::kHiddenSize / 2],
    const catapult_fp_t bias[kKvWidth],
    const catapult_fp_t scales[kKvWidth],
    catapult_fp_t output[kKvWidth]) {
  catapult_fp_t input_tile[kProjectionTileCapacity];
  catapult_fp_t partial_sum[kProjectionTileCapacity];

  for (int out_base = 0; out_base < kKvWidth; out_base += llm_accel::kAttentionMacCols) {
    const int out_extent = min_int(llm_accel::kAttentionMacCols, kKvWidth - out_base);

#pragma hls_pipeline_init_interval 1
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      partial_sum[out_offset] = HasBias ? bias[out_base + out_offset] : fp_zero();
    }

    for (int in_base = 0; in_base < llm_accel::kHiddenSize; in_base += llm_accel::kAttentionMacCols) {
      const int in_extent = min_int(llm_accel::kAttentionMacCols, llm_accel::kHiddenSize - in_base);

#pragma hls_pipeline_init_interval 1
      for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
        input_tile[in_offset] = input_token[in_base + in_offset];
      }

#pragma hls_pipeline_init_interval 2
      for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
        const int out_index = out_base + out_offset;
        partial_sum[out_offset] = fp_add_op(
            partial_sum[out_offset],
            weighted_chunk_dot_64_fp(
                input_tile,
                packed_weights,
                scales,
                out_index,
                in_base,
                llm_accel::kHiddenSize,
                in_extent));
      }
    }

#pragma hls_pipeline_init_interval 1
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      output[out_base + out_offset] = partial_sum[out_offset];
    }
  }
}

void project_kv_token_bias_fp(
    const catapult_fp_t input_token[llm_accel::kHiddenSize],
    const llm_accel::packed_w4_t packed_weights[kKvWidth * llm_accel::kHiddenSize / 2],
    const catapult_fp_t bias[kKvWidth],
    const catapult_fp_t scales[kKvWidth],
    catapult_fp_t output[kKvWidth]) {
  project_kv_token_fp_impl<true>(input_token, packed_weights, bias, scales, output);
}

catapult_fp_t rmsnorm_square_sum_fp(const catapult_fp_t input[llm_accel::kHiddenSize]) {
  catapult_fp_t square_sum = fp_zero();

  for (int chunk_index = 0; chunk_index < kRmsNormChunkCount; ++chunk_index) {
    catapult_fp_t lane_square[kParallelMacLaneCount];

#pragma hls_unroll yes
    for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
      const int index = chunk_index * kParallelMacLaneCount + lane;
      const catapult_fp_t value = input[index];
      lane_square[lane] = fp_mul_op(value, value);
    }

    square_sum = fp_add_op(square_sum, reduce_sum_64_fp(lane_square));
  }

  return square_sum;
}

void rmsnorm_token_fp(
    const catapult_fp_t input[llm_accel::kHiddenSize],
    const catapult_fp_t weight[llm_accel::kHiddenSize],
    const catapult_fp_t& rms_eps,
    catapult_fp_t output[llm_accel::kHiddenSize]) {
  const catapult_fp_t mean_square = fp_mul_op(rmsnorm_square_sum_fp(input), fp_const(1.0f / 1536.0f));
  const catapult_fp_t inv_rms = approx_rsqrt_fp(fp_add_op(mean_square, rms_eps));

#pragma hls_unroll yes
  for (int index = 0; index < llm_accel::kHiddenSize; ++index) {
    output[index] = fp_mul_op(input[index], fp_mul_op(weight[index], inv_rms));
  }
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

void load_projection_input_tile_fp(
    const catapult_fp_t input_token[llm_accel::kHiddenSize],
    int in_base,
    int lane_extent,
    catapult_fp_t input_tile[kProjectionTileCapacity]) {
#pragma hls_unroll yes
  for (int lane = 0; lane < kProjectionTileCapacity; ++lane) {
    input_tile[lane] = lane < lane_extent ? input_token[in_base + lane] : fp_zero();
  }
}

void load_projection_weight_tile_fp(
    const llm_accel::packed_w4_t packed_weights[kKvWidth * llm_accel::kHiddenSize / 2],
    int out_base,
    int out_extent,
    int in_base,
    int lane_extent,
    llm_accel::packed_w4_t packed_weight_tile[kProjectionPackedTileSize]) {
#pragma hls_pipeline_init_interval 1
  for (int out_offset = 0; out_offset < kProjectionTileCapacity; ++out_offset) {
    for (int lane_pair = 0; lane_pair < kProjectionTileCapacity / 2; ++lane_pair) {
      int low_nibble = 0;
      int high_nibble = 0;

      if (out_offset < out_extent) {
        const int lane0 = lane_pair * 2;
        const int lane1 = lane0 + 1;

        if (lane0 < lane_extent) {
          const int flat_index0 = (out_base + out_offset) * llm_accel::kHiddenSize + in_base + lane0;
          low_nibble = extract_int4_weight_nibble(packed_weights[flat_index0 / 2], (flat_index0 & 1) != 0);
        }

        if (lane1 < lane_extent) {
          const int flat_index1 = (out_base + out_offset) * llm_accel::kHiddenSize + in_base + lane1;
          high_nibble = extract_int4_weight_nibble(packed_weights[flat_index1 / 2], (flat_index1 & 1) != 0);
        }
      }

      packed_weight_tile[out_offset * (kProjectionTileCapacity / 2) + lane_pair] =
          static_cast<llm_accel::packed_w4_t>((high_nibble << 4) | low_nibble);
    }
  }
}

#endif

}  // namespace

namespace llm_accel {

constexpr int kKvTileCapacity = ::kProjectionTileCapacity;
constexpr int kKvPackedTileSize = ::kProjectionPackedTileSize;
constexpr int kMaxDdrPortBitWidth = 256;
constexpr int kCatapultFpBitWidth = 32;
constexpr int kPackedW4BitWidth = 8;
constexpr int kMaxFpWordsPerBeat = kMaxDdrPortBitWidth / kCatapultFpBitWidth;
constexpr int kMaxPackedWordsPerBeat = kMaxDdrPortBitWidth / kPackedW4BitWidth;
constexpr int kKvFpWordCount = kKvTileCapacity / kMaxFpWordsPerBeat;
constexpr int kKvPackedWordCount = kKvPackedTileSize / kMaxPackedWordsPerBeat;

struct KvFpTilePacket {
  prefill_catapult_fp_t data[kKvTileCapacity];
};

struct KvPackedTilePacket {
  packed_w4_t data[kKvPackedTileSize];
};

struct KvScaleTilePacket {
  prefill_catapult_fp_t data[kKvTileCapacity];
};

struct KvPartialTilePacket {
  prefill_catapult_fp_t data[kKvTileCapacity];
};

struct KvFpWordPacket {
  prefill_catapult_fp_t data[kMaxFpWordsPerBeat];
};

struct KvPackedWordPacket {
  packed_w4_t data[kMaxPackedWordsPerBeat];
};

void load_kv_fp_tile_packet(
    const prefill_catapult_fp_t* source,
    int base,
    int extent,
    KvFpTilePacket* packet) {
  for (int index = 0; index < kKvTileCapacity; ++index) {
    packet->data[index] = index < extent ? source[base + index] : prefill_catapult_fp_t(0.0f);
  }
}

void load_kv_packed_weight_tile_pair_packet(
    const packed_w4_t* k_packed_weights,
    const packed_w4_t* v_packed_weights,
    int out_base,
    int out_extent,
    int in_base,
    int lane_extent,
    KvPackedTilePacket* k_packet,
    KvPackedTilePacket* v_packet) {
#ifdef __SYNTHESIS__
  load_projection_weight_tile_fp(k_packed_weights, out_base, out_extent, in_base, lane_extent, k_packet->data);
  load_projection_weight_tile_fp(v_packed_weights, out_base, out_extent, in_base, lane_extent, v_packet->data);
#else
  (void)k_packed_weights;
  (void)v_packed_weights;
  (void)out_base;
  (void)out_extent;
  (void)in_base;
  (void)lane_extent;
  (void)k_packet;
  (void)v_packet;
#endif
}

void load_kv_scale_tile_pair_packet(
    const prefill_catapult_fp_t* k_scales,
    const prefill_catapult_fp_t* v_scales,
    int out_base,
    int out_extent,
    KvScaleTilePacket* k_packet,
    KvScaleTilePacket* v_packet) {
  for (int index = 0; index < kKvTileCapacity; ++index) {
    if (index < out_extent) {
      k_packet->data[index] = k_scales[out_base + index];
      v_packet->data[index] = v_scales[out_base + index];
    } else {
      k_packet->data[index] = prefill_catapult_fp_t(0.0f);
      v_packet->data[index] = prefill_catapult_fp_t(0.0f);
    }
  }
}

void init_kv_partial_tile_pair_packet(
    const prefill_catapult_fp_t* k_bias,
    const prefill_catapult_fp_t* v_bias,
    int out_base,
    int out_extent,
    KvPartialTilePacket* k_packet,
    KvPartialTilePacket* v_packet) {
  for (int index = 0; index < kKvTileCapacity; ++index) {
    if (index < out_extent) {
      k_packet->data[index] = k_bias[out_base + index];
      v_packet->data[index] = v_bias[out_base + index];
    } else {
      k_packet->data[index] = prefill_catapult_fp_t(0.0f);
      v_packet->data[index] = prefill_catapult_fp_t(0.0f);
    }
  }
}

void store_kv_partial_tile_pair_packet(
    const KvPartialTilePacket& k_packet,
    const KvPartialTilePacket& v_packet,
    int out_extent,
    prefill_catapult_fp_t* k_proj_token,
    prefill_catapult_fp_t* v_proj_token,
    int out_base) {
  for (int index = 0; index < out_extent; ++index) {
    k_proj_token[out_base + index] = k_packet.data[index];
    v_proj_token[out_base + index] = v_packet.data[index];
  }
}

void load_kv_fp_word_packet(
    const prefill_catapult_fp_t* source,
    int base,
    KvFpWordPacket* packet) {
  for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
    packet->data[index] = source[base + index];
  }
}

void store_kv_fp_word_packet(
    const KvFpWordPacket& packet,
    prefill_catapult_fp_t* destination,
    int base) {
  for (int index = 0; index < kMaxFpWordsPerBeat; ++index) {
    destination[base + index] = packet.data[index];
  }
}

void load_kv_packed_word_packet(
    const packed_w4_t* source,
    int base,
    KvPackedWordPacket* packet) {
  for (int index = 0; index < kMaxPackedWordsPerBeat; ++index) {
    packet->data[index] = source[base + index];
  }
}

void store_kv_packed_word_packet(
    const KvPackedWordPacket& packet,
    packed_w4_t* destination,
    int base) {
  for (int index = 0; index < kMaxPackedWordsPerBeat; ++index) {
    destination[base + index] = packet.data[index];
  }
}

void read_kv_fp_tile_packet(
    ac_channel<KvFpWordPacket>& channel,
    KvFpTilePacket* packet) {
  for (int word_index = 0; word_index < kKvFpWordCount; ++word_index) {
    const KvFpWordPacket word_packet = channel.read();
    store_kv_fp_word_packet(word_packet, packet->data, word_index * kMaxFpWordsPerBeat);
  }
}

void read_kv_fp_tile_words(
    ac_channel<KvFpWordPacket>& channel,
    prefill_catapult_fp_t* destination) {
  for (int word_index = 0; word_index < kKvFpWordCount; ++word_index) {
    const KvFpWordPacket word_packet = channel.read();
    store_kv_fp_word_packet(word_packet, destination, word_index * kMaxFpWordsPerBeat);
  }
}

void write_kv_fp_tile_packet(
    const KvFpTilePacket& packet,
    ac_channel<KvFpWordPacket>& channel) {
  for (int word_index = 0; word_index < kKvFpWordCount; ++word_index) {
    KvFpWordPacket word_packet;
    load_kv_fp_word_packet(packet.data, word_index * kMaxFpWordsPerBeat, &word_packet);
    channel.write(word_packet);
  }
}

void write_kv_fp_tile_words(
    const prefill_catapult_fp_t* source,
    ac_channel<KvFpWordPacket>& channel) {
  for (int word_index = 0; word_index < kKvFpWordCount; ++word_index) {
    KvFpWordPacket word_packet;
    load_kv_fp_word_packet(source, word_index * kMaxFpWordsPerBeat, &word_packet);
    channel.write(word_packet);
  }
}

void read_kv_packed_tile_packet(
    ac_channel<KvPackedWordPacket>& channel,
    KvPackedTilePacket* packet) {
  for (int word_index = 0; word_index < kKvPackedWordCount; ++word_index) {
    const KvPackedWordPacket word_packet = channel.read();
    store_kv_packed_word_packet(word_packet, packet->data, word_index * kMaxPackedWordsPerBeat);
  }
}

void qwen_prefill_attention_kv_tile_array_core(
    const prefill_catapult_fp_t input_tile[kKvTileCapacity],
    const prefill_catapult_fp_t input_layernorm_weight_tile[kKvTileCapacity],
    prefill_catapult_fp_t inv_rms,
    const packed_w4_t k_packed_weights_tile[kKvPackedTileSize],
    const packed_w4_t v_packed_weights_tile[kKvPackedTileSize],
    const prefill_catapult_fp_t k_scales_tile[kKvTileCapacity],
    const prefill_catapult_fp_t v_scales_tile[kKvTileCapacity],
    int lane_extent,
    int out_extent,
    prefill_catapult_fp_t k_partial_sum_tile[kKvTileCapacity],
    prefill_catapult_fp_t v_partial_sum_tile[kKvTileCapacity]) {
#ifdef __SYNTHESIS__
  catapult_fp_t normalized_input_tile[kKvTileCapacity];

#pragma hls_unroll yes
  for (int lane = 0; lane < kKvTileCapacity; ++lane) {
    if (lane < lane_extent) {
      normalized_input_tile[lane] =
          fp_mul_op(input_tile[lane], fp_mul_op(input_layernorm_weight_tile[lane], inv_rms));
    } else {
      normalized_input_tile[lane] = fp_zero();
    }
  }

#pragma hls_unroll no
#pragma hls_pipeline_init_interval 2
  for (int out_offset = 0; out_offset < kKvTileCapacity; ++out_offset) {
    if (out_offset < out_extent) {
      k_partial_sum_tile[out_offset] = fp_add_op(
          k_partial_sum_tile[out_offset],
          weighted_chunk_dot_64_fp(
              normalized_input_tile,
              k_packed_weights_tile,
              k_scales_tile,
              out_offset,
              0,
              kKvTileCapacity,
              lane_extent));
      v_partial_sum_tile[out_offset] = fp_add_op(
          v_partial_sum_tile[out_offset],
          weighted_chunk_dot_64_fp(
              normalized_input_tile,
              v_packed_weights_tile,
              v_scales_tile,
              out_offset,
              0,
              kKvTileCapacity,
              lane_extent));
    }
  }
#else
  (void)input_tile;
  (void)input_layernorm_weight_tile;
  (void)inv_rms;
  (void)k_packed_weights_tile;
  (void)v_packed_weights_tile;
  (void)k_scales_tile;
  (void)v_scales_tile;
  (void)lane_extent;
  (void)out_extent;
  (void)k_partial_sum_tile;
  (void)v_partial_sum_tile;
#endif
}

#pragma hls_design top
#pragma hls_pipeline_init_interval 1
#pragma hls_resource inv_rms:rsc variables="inv_rms" map_to_module="[DirectInput]"
#pragma hls_resource lane_extent:rsc variables="lane_extent" map_to_module="[DirectInput]"
#pragma hls_resource out_extent:rsc variables="out_extent" map_to_module="[DirectInput]"
void qwen_prefill_attention_kv_tile_stream_catapult(
    ac_channel<KvFpWordPacket>& input_tile_chan,
    ac_channel<KvFpWordPacket>& input_layernorm_weight_tile_chan,
    ac_channel<KvPackedWordPacket>& k_packed_weight_tile_chan,
    ac_channel<KvPackedWordPacket>& v_packed_weight_tile_chan,
    ac_channel<KvFpWordPacket>& k_scale_tile_chan,
    ac_channel<KvFpWordPacket>& v_scale_tile_chan,
    ac_channel<KvFpWordPacket>& k_partial_sum_tile_in_chan,
    ac_channel<KvFpWordPacket>& v_partial_sum_tile_in_chan,
    prefill_catapult_fp_t inv_rms,
    int lane_extent,
    int out_extent,
    ac_channel<KvFpWordPacket>& k_partial_sum_tile_out_chan,
    ac_channel<KvFpWordPacket>& v_partial_sum_tile_out_chan) {
#ifdef __SYNTHESIS__
  KvFpTilePacket input_tile_packet;
  KvFpTilePacket input_layernorm_weight_tile_packet;
  KvPackedTilePacket k_packed_weight_tile_packet;
  KvPackedTilePacket v_packed_weight_tile_packet;
  KvScaleTilePacket k_scale_tile_packet;
  KvScaleTilePacket v_scale_tile_packet;
  KvPartialTilePacket k_partial_sum_tile_packet;
  KvPartialTilePacket v_partial_sum_tile_packet;

  read_kv_fp_tile_packet(input_tile_chan, &input_tile_packet);
  read_kv_fp_tile_packet(input_layernorm_weight_tile_chan, &input_layernorm_weight_tile_packet);
  read_kv_packed_tile_packet(k_packed_weight_tile_chan, &k_packed_weight_tile_packet);
  read_kv_packed_tile_packet(v_packed_weight_tile_chan, &v_packed_weight_tile_packet);
  read_kv_fp_tile_words(k_scale_tile_chan, k_scale_tile_packet.data);
  read_kv_fp_tile_words(v_scale_tile_chan, v_scale_tile_packet.data);
  read_kv_fp_tile_words(k_partial_sum_tile_in_chan, k_partial_sum_tile_packet.data);
  read_kv_fp_tile_words(v_partial_sum_tile_in_chan, v_partial_sum_tile_packet.data);

  qwen_prefill_attention_kv_tile_array_core(
      input_tile_packet.data,
      input_layernorm_weight_tile_packet.data,
      inv_rms,
      k_packed_weight_tile_packet.data,
      v_packed_weight_tile_packet.data,
      k_scale_tile_packet.data,
      v_scale_tile_packet.data,
      lane_extent,
      out_extent,
      k_partial_sum_tile_packet.data,
      v_partial_sum_tile_packet.data);

  write_kv_fp_tile_words(k_partial_sum_tile_packet.data, k_partial_sum_tile_out_chan);
  write_kv_fp_tile_words(v_partial_sum_tile_packet.data, v_partial_sum_tile_out_chan);
#else
  (void)input_tile_chan;
  (void)input_layernorm_weight_tile_chan;
  (void)k_packed_weight_tile_chan;
  (void)v_packed_weight_tile_chan;
  (void)k_scale_tile_chan;
  (void)v_scale_tile_chan;
  (void)k_partial_sum_tile_in_chan;
  (void)v_partial_sum_tile_in_chan;
  (void)inv_rms;
  (void)lane_extent;
  (void)out_extent;
  (void)k_partial_sum_tile_out_chan;
  (void)v_partial_sum_tile_out_chan;
#endif
}

void qwen_prefill_attention_kv_cache_stage_catapult(
    const prefill_catapult_fp_t input_sequence[kPrefillCatapultSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    const prefill_catapult_fp_t input_layernorm_weight[kHiddenSize],
    prefill_catapult_fp_t rms_eps,
    const packed_w4_t k_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const packed_w4_t v_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2],
    const prefill_catapult_fp_t k_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_bias[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t k_scales[kPrefillCatapultKvWidth],
    const prefill_catapult_fp_t v_scales[kPrefillCatapultKvWidth],
    prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth],
    prefill_catapult_fp_t v_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth]) {
#ifndef __SYNTHESIS__
  if (seq_len <= 0 || seq_len > kPrefillCatapultSeqCapacity) {
    return;
  }

  if (tile_config.query != kAttentionMacRows || tile_config.key != kAttentionMacCols ||
      tile_config.hidden_proj != kAttentionMacCols || tile_config.kv_proj != kAttentionMacCols ||
      tile_config.head_dim != kAttentionMacCols) {
    return;
  }
#endif

#ifdef __SYNTHESIS__

  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);

ATNN_KV_STREAM_LOOP:
#pragma hls_pipeline_init_interval 2
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      const catapult_fp_t* input_token = input_sequence + token_index * kHiddenSize;
      const catapult_fp_t mean_square = fp_mul_op(rmsnorm_square_sum_fp(input_token), fp_const(1.0f / 1536.0f));
      const catapult_fp_t inv_rms = approx_rsqrt_fp(fp_add_op(mean_square, rms_eps));
      catapult_fp_t* k_proj_token = k_cache + token_index * kKvWidth;
      catapult_fp_t* v_proj_token = v_cache + token_index * kKvWidth;

      for (int out_base = 0; out_base < kKvWidth; out_base += kKvTileCapacity) {
        const int out_extent = min_int(kKvTileCapacity, kKvWidth - out_base);
        KvScaleTilePacket k_scale_tile_packet;
        KvScaleTilePacket v_scale_tile_packet;
        KvPartialTilePacket k_partial_sum_tile_packet;
        KvPartialTilePacket v_partial_sum_tile_packet;

        load_kv_scale_tile_pair_packet(
            k_scales,
            v_scales,
            out_base,
            out_extent,
            &k_scale_tile_packet,
            &v_scale_tile_packet);
        init_kv_partial_tile_pair_packet(
            k_bias,
            v_bias,
            out_base,
            out_extent,
            &k_partial_sum_tile_packet,
            &v_partial_sum_tile_packet);

        for (int in_base = 0; in_base < kHiddenSize; in_base += kKvTileCapacity) {
          const int in_extent = min_int(kKvTileCapacity, kHiddenSize - in_base);
          KvFpTilePacket input_tile_packet;
          KvFpTilePacket layernorm_tile_packet;
          KvPackedTilePacket k_packed_weight_tile_packet;
          KvPackedTilePacket v_packed_weight_tile_packet;

          load_kv_fp_tile_packet(input_token, in_base, in_extent, &input_tile_packet);
          load_kv_fp_tile_packet(input_layernorm_weight, in_base, in_extent, &layernorm_tile_packet);
          load_kv_packed_weight_tile_pair_packet(
              k_packed_weights,
              v_packed_weights,
              out_base,
              out_extent,
              in_base,
              in_extent,
              &k_packed_weight_tile_packet,
              &v_packed_weight_tile_packet);

          qwen_prefill_attention_kv_tile_array_core(
              input_tile_packet.data,
              layernorm_tile_packet.data,
              inv_rms,
              k_packed_weight_tile_packet.data,
              v_packed_weight_tile_packet.data,
              k_scale_tile_packet.data,
              v_scale_tile_packet.data,
              in_extent,
              out_extent,
              k_partial_sum_tile_packet.data,
              v_partial_sum_tile_packet.data);
        }

        store_kv_partial_tile_pair_packet(
            k_partial_sum_tile_packet,
            v_partial_sum_tile_packet,
            out_extent,
            k_proj_token,
            v_proj_token,
            out_base);
      }

      for (int head_base = 0; head_base < kNumKeyValueHeads; head_base += tile_config.kv_heads_parallel) {
        const int head_end = min_int(kNumKeyValueHeads, head_base + tile_config.kv_heads_parallel);

#pragma hls_unroll yes
        for (int head = head_base; head < head_end; ++head) {
          apply_rope_inplace_fp(k_proj_token + head * kHeadDim, token_index);
        }
      }
    }
  }
#else
  (void)input_sequence;
  (void)seq_len;
  (void)tile_config;
  (void)input_layernorm_weight;
  (void)rms_eps;
  (void)k_packed_weights;
  (void)v_packed_weights;
  (void)k_bias;
  (void)v_bias;
  (void)k_scales;
  (void)v_scales;
  (void)k_cache;
  (void)v_cache;
#endif
}

}  // namespace llm_accel