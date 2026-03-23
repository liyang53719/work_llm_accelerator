#include "qwen_prefill_attention_stream_top.h"

namespace llm_accel {
namespace {

inline prefill_catapult_fp_t stream_fp_zero() {
  return prefill_catapult_fp_t(0.0f);
}

int ceil_div_int(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

void read_fp_words(
    ac_channel<PrefillStreamFpWordPacket>& channel,
    prefill_catapult_fp_t* destination,
    int element_count) {
  const int packet_count = ceil_div_int(element_count, kPrefillStreamFpWordsPerPacket);

READ_FP_PACKETS:
#pragma hls_unroll no
#pragma hls_pipeline_init_interval 1
  for (int packet_index = 0; packet_index < packet_count; ++packet_index) {
    const PrefillStreamFpWordPacket packet = channel.read();

READ_FP_WORDS:
#pragma hls_unroll yes
    for (int word_index = 0; word_index < kPrefillStreamFpWordsPerPacket; ++word_index) {
      const int flat_index = packet_index * kPrefillStreamFpWordsPerPacket + word_index;
      if (flat_index < element_count) {
        destination[flat_index] = packet.data[word_index];
      }
    }
  }
}

void read_packed_words(
    ac_channel<PrefillStreamPackedWordPacket>& channel,
    packed_w4_t* destination,
    int element_count) {
  const int packet_count = ceil_div_int(element_count, kPrefillStreamPackedWordsPerPacket);

READ_PACKED_PACKETS:
#pragma hls_unroll no
#pragma hls_pipeline_init_interval 1
  for (int packet_index = 0; packet_index < packet_count; ++packet_index) {
    const PrefillStreamPackedWordPacket packet = channel.read();

READ_PACKED_WORDS:
#pragma hls_unroll yes
    for (int word_index = 0; word_index < kPrefillStreamPackedWordsPerPacket; ++word_index) {
      const int flat_index = packet_index * kPrefillStreamPackedWordsPerPacket + word_index;
      if (flat_index < element_count) {
        destination[flat_index] = packet.data[word_index];
      }
    }
  }
}

void write_fp_words(
    const prefill_catapult_fp_t* source,
    int element_count,
    ac_channel<PrefillStreamFpWordPacket>& channel) {
  const int packet_count = ceil_div_int(element_count, kPrefillStreamFpWordsPerPacket);

WRITE_FP_PACKETS:
#pragma hls_unroll no
#pragma hls_pipeline_init_interval 1
  for (int packet_index = 0; packet_index < packet_count; ++packet_index) {
    PrefillStreamFpWordPacket packet;

WRITE_FP_WORDS:
#pragma hls_unroll yes
    for (int word_index = 0; word_index < kPrefillStreamFpWordsPerPacket; ++word_index) {
      const int flat_index = packet_index * kPrefillStreamFpWordsPerPacket + word_index;
      packet.data[word_index] = flat_index < element_count ? source[flat_index] : stream_fp_zero();
    }

    channel.write(packet);
  }
}

bool valid_attention_stream_config(int seq_len) {
  return seq_len > 0 && seq_len <= kPrefillCatapultSeqCapacity;
}

}  // namespace

#ifndef QWEN_HLS_GLUE_INLINE_CHILD_TOPS
#pragma hls_design top
#endif
KernelStatus qwen_prefill_attention_stream_top_catapult(
    int seq_len,
    prefill_catapult_fp_t rms_eps,
    ac_channel<PrefillStreamFpWordPacket>& input_sequence_chan,
    ac_channel<PrefillStreamFpWordPacket>& input_layernorm_weight_chan,
    ac_channel<PrefillStreamPackedWordPacket>& q_packed_weight_chan,
    ac_channel<PrefillStreamPackedWordPacket>& k_packed_weight_chan,
    ac_channel<PrefillStreamPackedWordPacket>& v_packed_weight_chan,
    ac_channel<PrefillStreamPackedWordPacket>& o_packed_weight_chan,
    ac_channel<PrefillStreamFpWordPacket>& q_bias_chan,
    ac_channel<PrefillStreamFpWordPacket>& k_bias_chan,
    ac_channel<PrefillStreamFpWordPacket>& v_bias_chan,
    ac_channel<PrefillStreamFpWordPacket>& q_scale_chan,
    ac_channel<PrefillStreamFpWordPacket>& k_scale_chan,
    ac_channel<PrefillStreamFpWordPacket>& v_scale_chan,
    ac_channel<PrefillStreamFpWordPacket>& o_scale_chan,
    ac_channel<PrefillStreamFpWordPacket>& k_cache_out_chan,
    ac_channel<PrefillStreamFpWordPacket>& v_cache_out_chan,
    ac_channel<PrefillStreamFpWordPacket>& output_sequence_chan) {
  PrefillAttentionTileConfig tile_config = default_prefill_tile_config().attention;
  tile_config.head_dim = kHeadDim;
  if (!valid_attention_stream_config(seq_len)) {
    return {false, 2};
  }

  prefill_catapult_fp_t input_sequence[kPrefillCatapultSeqCapacity * kHiddenSize];
  prefill_catapult_fp_t input_layernorm_weight[kHiddenSize];
  packed_w4_t q_packed_weights[kHiddenSize * kHiddenSize / 2];
  packed_w4_t k_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2];
  packed_w4_t v_packed_weights[kPrefillCatapultKvWidth * kHiddenSize / 2];
  packed_w4_t o_packed_weights[kHiddenSize * kHiddenSize / 2];
  prefill_catapult_fp_t q_bias[kHiddenSize];
  prefill_catapult_fp_t k_bias[kPrefillCatapultKvWidth];
  prefill_catapult_fp_t v_bias[kPrefillCatapultKvWidth];
  prefill_catapult_fp_t q_scales[kHiddenSize];
  prefill_catapult_fp_t k_scales[kPrefillCatapultKvWidth];
  prefill_catapult_fp_t v_scales[kPrefillCatapultKvWidth];
  prefill_catapult_fp_t o_scales[kHiddenSize];
  prefill_catapult_fp_t k_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth];
  prefill_catapult_fp_t v_cache[kPrefillCatapultSeqCapacity * kPrefillCatapultKvWidth];
  prefill_catapult_fp_t output_sequence[kPrefillCatapultSeqCapacity * kHiddenSize];

  read_fp_words(input_sequence_chan, input_sequence, seq_len * kHiddenSize);
  read_fp_words(input_layernorm_weight_chan, input_layernorm_weight, kHiddenSize);
  read_packed_words(q_packed_weight_chan, q_packed_weights, kHiddenSize * kHiddenSize / 2);
  read_packed_words(k_packed_weight_chan, k_packed_weights, kPrefillCatapultKvWidth * kHiddenSize / 2);
  read_packed_words(v_packed_weight_chan, v_packed_weights, kPrefillCatapultKvWidth * kHiddenSize / 2);
  read_packed_words(o_packed_weight_chan, o_packed_weights, kHiddenSize * kHiddenSize / 2);
  read_fp_words(q_bias_chan, q_bias, kHiddenSize);
  read_fp_words(k_bias_chan, k_bias, kPrefillCatapultKvWidth);
  read_fp_words(v_bias_chan, v_bias, kPrefillCatapultKvWidth);
  read_fp_words(q_scale_chan, q_scales, kHiddenSize);
  read_fp_words(k_scale_chan, k_scales, kPrefillCatapultKvWidth);
  read_fp_words(v_scale_chan, v_scales, kPrefillCatapultKvWidth);
  read_fp_words(o_scale_chan, o_scales, kHiddenSize);

  const KernelStatus status = qwen_prefill_attention_kernel_catapult(
      {input_sequence},
      seq_len,
      tile_config,
      {input_layernorm_weight},
      rms_eps,
      {q_packed_weights},
      {k_packed_weights},
      {v_packed_weights},
      {o_packed_weights},
      {q_bias},
      {k_bias},
      {v_bias},
      {q_scales},
      {k_scales},
      {v_scales},
      {o_scales},
      {k_cache},
      {v_cache},
      {output_sequence});
  if (!status.ok) {
    return status;
  }

  write_fp_words(k_cache, seq_len * kPrefillCatapultKvWidth, k_cache_out_chan);
  write_fp_words(v_cache, seq_len * kPrefillCatapultKvWidth, v_cache_out_chan);
  write_fp_words(output_sequence, seq_len * kHiddenSize, output_sequence_chan);
  return status;
}

}  // namespace llm_accel
