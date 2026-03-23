#include "qwen_prefill_mlp_stream_top.h"

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

bool valid_mlp_stream_config(int seq_len) {
  return seq_len > 0 && seq_len <= kPrefillCatapultSeqCapacity;
}

}  // namespace

#pragma hls_design top
KernelStatus qwen_prefill_mlp_stream_top_catapult(
    int seq_len,
    prefill_catapult_fp_t rms_eps,
    ac_channel<PrefillStreamFpWordPacket>& attention_residual_chan,
    ac_channel<PrefillStreamFpWordPacket>& post_attention_layernorm_weight_chan,
    ac_channel<PrefillStreamPackedWordPacket>& gate_packed_weight_chan,
    ac_channel<PrefillStreamPackedWordPacket>& up_packed_weight_chan,
    ac_channel<PrefillStreamPackedWordPacket>& down_packed_weight_chan,
    ac_channel<PrefillStreamFpWordPacket>& gate_scale_chan,
    ac_channel<PrefillStreamFpWordPacket>& up_scale_chan,
    ac_channel<PrefillStreamFpWordPacket>& down_scale_chan,
    ac_channel<PrefillStreamFpWordPacket>& output_sequence_chan) {
  const PrefillMLPTileConfig tile_config = default_prefill_tile_config().mlp;
  if (!valid_mlp_stream_config(seq_len)) {
    return {false, 2};
  }

  prefill_catapult_fp_t attention_residual[kPrefillCatapultSeqCapacity * kHiddenSize];
  prefill_catapult_fp_t post_attention_layernorm_weight[kHiddenSize];
  packed_w4_t gate_packed_weights[kIntermediateSize * kHiddenSize / 2];
  packed_w4_t up_packed_weights[kIntermediateSize * kHiddenSize / 2];
  packed_w4_t down_packed_weights[kIntermediateSize * kHiddenSize / 2];
  prefill_catapult_fp_t gate_scales[kIntermediateSize];
  prefill_catapult_fp_t up_scales[kIntermediateSize];
  prefill_catapult_fp_t down_scales[kHiddenSize];
  prefill_catapult_fp_t output_sequence[kPrefillCatapultSeqCapacity * kHiddenSize];

  read_fp_words(attention_residual_chan, attention_residual, seq_len * kHiddenSize);
  read_fp_words(post_attention_layernorm_weight_chan, post_attention_layernorm_weight, kHiddenSize);
  read_packed_words(gate_packed_weight_chan, gate_packed_weights, kIntermediateSize * kHiddenSize / 2);
  read_packed_words(up_packed_weight_chan, up_packed_weights, kIntermediateSize * kHiddenSize / 2);
  read_packed_words(down_packed_weight_chan, down_packed_weights, kIntermediateSize * kHiddenSize / 2);
  read_fp_words(gate_scale_chan, gate_scales, kIntermediateSize);
  read_fp_words(up_scale_chan, up_scales, kIntermediateSize);
  read_fp_words(down_scale_chan, down_scales, kHiddenSize);

  const KernelStatus status = qwen_prefill_mlp_kernel_catapult(
      attention_residual,
      seq_len,
      tile_config,
      post_attention_layernorm_weight,
      rms_eps,
      gate_packed_weights,
      up_packed_weights,
      down_packed_weights,
      gate_scales,
      up_scales,
      down_scales,
      output_sequence);
  if (!status.ok) {
    return status;
  }

  write_fp_words(output_sequence, seq_len * kHiddenSize, output_sequence_chan);
  return status;
}

}  // namespace llm_accel
