#include "qwen_prefill_mlp_kernel.h"

#ifdef __SYNTHESIS__
#include "../include/ac_int.h"
#include "../include/ac_std_float.h"
#include "../include/ccs_dw_fp_lib.h"
#endif

namespace {

constexpr int kMlpHiddenTileCapacity = llm_accel::kDefaultPrefillMLPHiddenTile;
constexpr int kMlpFfTileCapacity = llm_accel::kDefaultPrefillMLPFFTile;
constexpr int kProjectionTileCapacity =
    kMlpHiddenTileCapacity > kMlpFfTileCapacity ? kMlpHiddenTileCapacity : kMlpFfTileCapacity;
constexpr int kMlpPackedWeightWords = llm_accel::kIntermediateSize * llm_accel::kHiddenSize / 2;

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

void project_tiled(
    const llm_accel::scalar_t* input_token,
    const llm_accel::packed_w4_t* packed_weights,
    const llm_accel::scalar_t* scales,
    int in_dim,
    int out_dim,
    int in_tile,
    int out_tile,
    llm_accel::scalar_t* output) {
  llm_accel::scalar_t input_tile[kProjectionTileCapacity];
  llm_accel::scalar_t partial_sum[kProjectionTileCapacity];

  for (int out_base = 0; out_base < out_dim; out_base += out_tile) {
    const int out_extent = min_int(out_tile, out_dim - out_base);
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      partial_sum[out_offset] = 0.0f;
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

inline float silu(float value) {
  return value / (1.0f + approx_exp(-value));
}

}  // namespace

namespace llm_accel {

KernelStatus qwen_prefill_mlp_kernel(
    const scalar_t* attention_residual,
    int seq_len,
  const PrefillMLPTileConfig& tile_config,
    const scalar_t* post_attention_layernorm_weight,
    scalar_t rms_eps,
    const packed_w4_t* gate_packed_weights,
    const packed_w4_t* up_packed_weights,
    const packed_w4_t* down_packed_weights,
    const scalar_t* gate_scales,
    const scalar_t* up_scales,
    const scalar_t* down_scales,
    scalar_t* output_sequence) {
  if (attention_residual == nullptr || output_sequence == nullptr || seq_len <= 0 ||
      post_attention_layernorm_weight == nullptr || gate_packed_weights == nullptr || up_packed_weights == nullptr ||
      down_packed_weights == nullptr || gate_scales == nullptr || up_scales == nullptr || down_scales == nullptr ||
      tile_config.seq <= 0 || tile_config.hidden <= 0 || tile_config.ff <= 0 ||
      tile_config.hidden > kMlpHiddenTileCapacity || tile_config.ff > kMlpFfTileCapacity) {
    return {false, 1};
  }

  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      const scalar_t* attention_token = attention_residual + token_index * kHiddenSize;
      scalar_t* output_token = output_sequence + token_index * kHiddenSize;

      scalar_t post_norm[kHiddenSize];
      scalar_t gate_proj[kIntermediateSize];
      scalar_t up_proj[kIntermediateSize];
      scalar_t silu_mul[kIntermediateSize];
      scalar_t down_proj[kHiddenSize];

      rmsnorm_token(attention_token, post_attention_layernorm_weight, rms_eps, post_norm);
      project_tiled(
          post_norm,
          gate_packed_weights,
          gate_scales,
          kHiddenSize,
          kIntermediateSize,
          tile_config.hidden,
          tile_config.ff,
          gate_proj);
      project_tiled(
          post_norm,
          up_packed_weights,
          up_scales,
          kHiddenSize,
          kIntermediateSize,
          tile_config.hidden,
          tile_config.ff,
          up_proj);
      for (int index = 0; index < kIntermediateSize; ++index) {
        silu_mul[index] = silu(gate_proj[index]) * up_proj[index];
      }
      project_tiled(
          silu_mul,
          down_packed_weights,
          down_scales,
          kIntermediateSize,
          kHiddenSize,
          tile_config.ff,
          tile_config.hidden,
          down_proj);
      for (int index = 0; index < kHiddenSize; ++index) {
        output_token[index] = attention_token[index] + down_proj[index];
      }
    }
  }

  return {true, 0};
}

#ifdef __SYNTHESIS__
namespace {

using catapult_fp_t = llm_accel::prefill_catapult_fp_t;

constexpr int kParallelMacLaneCount = llm_accel::kTileN;

inline catapult_fp_t fp_add_op(const catapult_fp_t& lhs, const catapult_fp_t& rhs);

template <int PairCount, int Index>
struct PairwiseSumBuilder {
  static void run(const catapult_fp_t* input, catapult_fp_t* output) {
    output[Index] = fp_add_op(input[Index * 2], input[Index * 2 + 1]);
    PairwiseSumBuilder<PairCount, Index + 1>::run(input, output);
  }
};

template <int PairCount>
struct PairwiseSumBuilder<PairCount, PairCount> {
  static void run(const catapult_fp_t*, catapult_fp_t*) {}
};

template <int LaneCount>
struct ReductionTree {
  static catapult_fp_t run(const catapult_fp_t* input) {
    catapult_fp_t partial[LaneCount / 2];
    PairwiseSumBuilder<LaneCount / 2, 0>::run(input, partial);
    return ReductionTree<LaneCount / 2>::run(partial);
  }
};

template <>
struct ReductionTree<1> {
  static catapult_fp_t run(const catapult_fp_t* input) {
    return input[0];
  }
};

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
#pragma hls_unroll no
  for (int step = 0; step < 6; ++step) {
    if (!fp_lt_op(scaled, neg_one)) {
      break;
    }
    scaled = fp_mul_op(scaled, half);
    ++half_steps;
  }

  catapult_fp_t term = fp_one();
  catapult_fp_t series = fp_one();
#pragma hls_unroll no
  for (int degree = 1; degree <= 6; ++degree) {
    term = fp_mul_op(term, fp_div_op(scaled, fp_const_int(degree)));
    series = fp_add_op(series, term);
  }

#pragma hls_unroll no
  for (int step = 0; step < half_steps; ++step) {
    series = fp_mul_op(series, series);
  }
  return series;
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
  return ReductionTree<kParallelMacLaneCount>::run(values);
}

inline catapult_fp_t silu_fp(const catapult_fp_t& value);

catapult_fp_t weighted_chunk_dot_fp(
    const catapult_fp_t* input_tile,
    const llm_accel::packed_w4_t* packed_weights,
    const catapult_fp_t* scales,
    int out_index,
    int in_index_base,
    int in_dim,
    int lane_extent) {
  catapult_fp_t accum = fp_zero();
  for (int lane = 0; lane < lane_extent; ++lane) {
    accum = fp_add_op(
        accum,
        fp_mul_op(
            input_tile[lane],
            dequantized_weight_fp(packed_weights, scales, out_index, in_index_base + lane, in_dim)));
  }
  return accum;
}

void silu_mul_block_128_fp(
    const catapult_fp_t* gate_proj,
    const catapult_fp_t* up_proj,
    int lane_extent,
    catapult_fp_t* silu_mul) {
#pragma hls_unroll no
  for (int lane = 0; lane < lane_extent; ++lane) {
    silu_mul[lane] = fp_mul_op(silu_fp(gate_proj[lane]), up_proj[lane]);
  }
}

void residual_add_128_fp(
    const catapult_fp_t* attention_token,
    const catapult_fp_t* down_proj,
    int lane_extent,
    catapult_fp_t* output_token) {
#pragma hls_unroll no
  for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
    if (lane < lane_extent) {
      output_token[lane] = fp_add_op(attention_token[lane], down_proj[lane]);
    }
  }
}

void rmsnorm_token_fp(
    const catapult_fp_t* input,
    const catapult_fp_t* weight,
    const catapult_fp_t& rms_eps,
    catapult_fp_t* output) {
  catapult_fp_t mean_square = fp_zero();
#pragma hls_unroll no
  for (int base = 0; base < llm_accel::kHiddenSize; base += kParallelMacLaneCount) {
    catapult_fp_t lane_square[kParallelMacLaneCount];
#pragma hls_unroll no
    for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
      const catapult_fp_t value = input[base + lane];
      lane_square[lane] = fp_mul_op(value, value);
    }
    mean_square = fp_add_op(mean_square, reduce_sum_128_fp(lane_square));
  }
  mean_square = fp_div_op(mean_square, fp_const_int(llm_accel::kHiddenSize));
  const catapult_fp_t inv_rms = fp_div_op(fp_one(), fp_sqrt_op(fp_add_op(mean_square, rms_eps)));
#pragma hls_unroll no
  for (int base = 0; base < llm_accel::kHiddenSize; base += kParallelMacLaneCount) {
#pragma hls_unroll no
    for (int lane = 0; lane < kParallelMacLaneCount; ++lane) {
      output[base + lane] = fp_mul_op(fp_mul_op(input[base + lane], inv_rms), weight[base + lane]);
    }
  }
}

void project_tiled_fp(
    const catapult_fp_t* input_token,
    const llm_accel::packed_w4_t* packed_weights,
    const catapult_fp_t* scales,
    int in_dim,
    int out_dim,
    int in_tile,
    int out_tile,
    catapult_fp_t* output) {
  catapult_fp_t input_tile[kProjectionTileCapacity];
  catapult_fp_t partial_sum[kProjectionTileCapacity];

  for (int out_base = 0; out_base < out_dim; out_base += out_tile) {
    const int out_extent = min_int(out_tile, out_dim - out_base);
MLP_PROJ_INIT_LOOP:
#pragma hls_pipeline_init_interval 1
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      partial_sum[out_offset] = fp_zero();
    }

    for (int in_base = 0; in_base < in_dim; in_base += in_tile) {
      const int in_extent = min_int(in_tile, in_dim - in_base);
MLP_PROJ_LOAD_LOOP:
#pragma hls_pipeline_init_interval 1
      for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
        input_tile[in_offset] = input_token[in_base + in_offset];
      }
MLP_PROJ_ACCUM_LOOP:
#pragma hls_pipeline_init_interval 4
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
                  in_dim,
                  lane_extent));
        }
        partial_sum[out_offset] = accum;
      }
    }

MLP_PROJ_STORE_LOOP:
#pragma hls_pipeline_init_interval 1
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      output[out_base + out_offset] = partial_sum[out_offset];
    }
  }
}

void project_tiled_range_fp(
    const catapult_fp_t* input_token,
    const llm_accel::packed_w4_t* packed_weights,
    const catapult_fp_t* scales,
    int in_dim,
    int out_index_base,
    int out_extent,
    int in_tile,
    catapult_fp_t* output_tile) {
  catapult_fp_t input_tile[kProjectionTileCapacity];
  catapult_fp_t partial_sum[kProjectionTileCapacity];

MLP_PROJ_RANGE_INIT_LOOP:
#pragma hls_pipeline_init_interval 1
  for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
    partial_sum[out_offset] = fp_zero();
  }

  for (int in_base = 0; in_base < in_dim; in_base += in_tile) {
    const int in_extent = min_int(in_tile, in_dim - in_base);
MLP_PROJ_RANGE_LOAD_LOOP:
#pragma hls_pipeline_init_interval 1
    for (int in_offset = 0; in_offset < in_extent; ++in_offset) {
      input_tile[in_offset] = input_token[in_base + in_offset];
    }
MLP_PROJ_RANGE_ACCUM_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      const int out_index = out_index_base + out_offset;
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
                in_dim,
                lane_extent));
      }
      partial_sum[out_offset] = accum;
    }
  }

MLP_PROJ_RANGE_STORE_LOOP:
#pragma hls_pipeline_init_interval 1
  for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
    output_tile[out_offset] = partial_sum[out_offset];
  }
}

void down_project_accumulate_tile_fp(
    const catapult_fp_t* silu_tile,
    int ff_index_base,
    int ff_extent,
    int hidden_tile,
    bool initialize,
    const llm_accel::packed_w4_t* down_packed_weights,
    const catapult_fp_t* down_scales,
    catapult_fp_t* down_accum) {
  catapult_fp_t partial_sum[kProjectionTileCapacity];

  for (int out_base = 0; out_base < llm_accel::kHiddenSize; out_base += hidden_tile) {
    const int out_extent = min_int(hidden_tile, llm_accel::kHiddenSize - out_base);
MLP_DOWN_ACCUM_INIT_LOOP:
#pragma hls_pipeline_init_interval 1
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      partial_sum[out_offset] = initialize ? fp_zero() : down_accum[out_base + out_offset];
    }
MLP_DOWN_ACCUM_LOOP:
#pragma hls_pipeline_init_interval 4
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      const int out_index = out_base + out_offset;
      catapult_fp_t accum = partial_sum[out_offset];
      for (int in_offset = 0; in_offset < ff_extent; in_offset += kParallelMacLaneCount) {
        const int lane_extent = min_int(kParallelMacLaneCount, ff_extent - in_offset);
        accum = fp_add_op(
            accum,
            weighted_chunk_dot_fp(
                silu_tile + in_offset,
                down_packed_weights,
                down_scales,
                out_index,
                ff_index_base + in_offset,
                llm_accel::kIntermediateSize,
                lane_extent));
      }
      partial_sum[out_offset] = accum;
    }
MLP_DOWN_ACCUM_STORE_LOOP:
#pragma hls_pipeline_init_interval 1
    for (int out_offset = 0; out_offset < out_extent; ++out_offset) {
      down_accum[out_base + out_offset] = partial_sum[out_offset];
    }
  }
}

inline catapult_fp_t silu_fp(const catapult_fp_t& value) {
  return fp_div_op(value, fp_add_op(fp_one(), approx_exp_fp(fp_sub_op(fp_zero(), value))));
}

}  // namespace

KernelStatus qwen_prefill_mlp_kernel_catapult(
    const prefill_catapult_fp_t attention_residual[kPrefillCatapultSeqCapacity * kHiddenSize],
    int seq_len,
    const PrefillMLPTileConfig& tile_config,
    const prefill_catapult_fp_t post_attention_layernorm_weight[kHiddenSize],
    prefill_catapult_fp_t rms_eps,
  const packed_w4_t gate_packed_weights[kMlpPackedWeightWords],
  const packed_w4_t up_packed_weights[kMlpPackedWeightWords],
  const packed_w4_t down_packed_weights[kMlpPackedWeightWords],
  const prefill_catapult_fp_t gate_scales[kIntermediateSize],
  const prefill_catapult_fp_t up_scales[kIntermediateSize],
  const prefill_catapult_fp_t down_scales[kHiddenSize],
    prefill_catapult_fp_t output_sequence[kPrefillCatapultSeqCapacity * kHiddenSize]) {
  if (seq_len <= 0 || tile_config.seq <= 0 || tile_config.hidden <= 0 || tile_config.ff <= 0 ||
      tile_config.hidden > kMlpHiddenTileCapacity || tile_config.ff > kMlpFfTileCapacity) {
    return {false, 1};
  }

  const int seq_tile = max_int(1, tile_config.seq);

  for (int token_begin = 0; token_begin < seq_len; token_begin += seq_tile) {
    const int token_end = min_int(seq_len, token_begin + seq_tile);
MLP_TOKEN_LOOP:
#pragma hls_pipeline_init_interval 8
    for (int token_index = token_begin; token_index < token_end; ++token_index) {
      const prefill_catapult_fp_t* attention_token = attention_residual + token_index * kHiddenSize;
      prefill_catapult_fp_t* output_token = output_sequence + token_index * kHiddenSize;

      prefill_catapult_fp_t post_norm[kHiddenSize];
      prefill_catapult_fp_t down_accum[kHiddenSize];
      prefill_catapult_fp_t gate_tile[kProjectionTileCapacity];
      prefill_catapult_fp_t up_tile[kProjectionTileCapacity];
      prefill_catapult_fp_t silu_tile[kProjectionTileCapacity];

      rmsnorm_token_fp(attention_token, post_attention_layernorm_weight, rms_eps, post_norm);

      for (int ff_base = 0; ff_base < kIntermediateSize; ff_base += tile_config.ff) {
        const int ff_extent = min_int(tile_config.ff, kIntermediateSize - ff_base);
        project_tiled_range_fp(
            post_norm,
            gate_packed_weights,
            gate_scales,
            kHiddenSize,
            ff_base,
            ff_extent,
            tile_config.hidden,
            gate_tile);
        project_tiled_range_fp(
            post_norm,
            up_packed_weights,
            up_scales,
            kHiddenSize,
            ff_base,
            ff_extent,
            tile_config.hidden,
            up_tile);

        for (int base = 0; base < ff_extent; base += kParallelMacLaneCount) {
          const int lane_extent = min_int(kParallelMacLaneCount, ff_extent - base);
          silu_mul_block_128_fp(gate_tile + base, up_tile + base, lane_extent, silu_tile + base);
        }

        down_project_accumulate_tile_fp(
            silu_tile,
            ff_base,
            ff_extent,
            tile_config.hidden,
            ff_base == 0,
            down_packed_weights,
            down_scales,
            down_accum);
      }

      for (int base = 0; base < kHiddenSize; base += kParallelMacLaneCount) {
        const int lane_extent = min_int(kParallelMacLaneCount, kHiddenSize - base);
        residual_add_128_fp(attention_token + base, down_accum + base, lane_extent, output_token + base);
      }
    }
  }

  return {true, 0};
}
#endif

}  // namespace llm_accel