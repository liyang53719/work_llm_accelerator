#include "qwen_prefill_attention_kernel.h"
#include "qwen_prefill_mlp_kernel.h"

#include <vector>

#ifndef __SYNTHESIS__

namespace llm_accel {
namespace {

constexpr int kHostMlpStreamCoreHiddenTile = 128;
constexpr int kHostMlpStreamCoreFfTile = 128;
constexpr int kHostMlpStreamCorePackedTileWords =
    kHostMlpStreamCoreHiddenTile * kHostMlpStreamCoreFfTile / 2;

inline int ceil_div_int(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

inline int decode_packed_weight(const packed_w4_t* packed_weights, int flat_index) {
  const packed_w4_t packed_value = packed_weights[flat_index / 2];
  const int nibble = (flat_index & 1) == 0 ? (packed_value & 0xF) : ((packed_value >> 4) & 0xF);
  return nibble >= 8 ? nibble - 16 : nibble;
}

inline void set_packed_weight(packed_w4_t* packed_weights, int flat_index, int value) {
  const packed_w4_t encoded = static_cast<packed_w4_t>(value & 0xF);
  packed_w4_t& packed_ref = packed_weights[flat_index / 2];
  if ((flat_index & 1) == 0) {
    packed_ref = static_cast<packed_w4_t>((packed_ref & 0xF0) | encoded);
  } else {
    packed_ref = static_cast<packed_w4_t>((packed_ref & 0x0F) | (encoded << 4));
  }
}

void read_fp_words(
    ac_channel<PrefillStreamFpWordPacket>& channel,
    prefill_catapult_fp_t* destination,
    int element_count) {
  const int packet_count = ceil_div_int(element_count, kPrefillStreamFpWordsPerPacket);
  for (int packet_index = 0; packet_index < packet_count; ++packet_index) {
    const PrefillStreamFpWordPacket packet = channel.read();
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
  for (int packet_index = 0; packet_index < packet_count; ++packet_index) {
    const PrefillStreamPackedWordPacket packet = channel.read();
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
  for (int packet_index = 0; packet_index < packet_count; ++packet_index) {
    PrefillStreamFpWordPacket packet = {};
    for (int word_index = 0; word_index < kPrefillStreamFpWordsPerPacket; ++word_index) {
      const int flat_index = packet_index * kPrefillStreamFpWordsPerPacket + word_index;
      packet.data[word_index] = flat_index < element_count ? source[flat_index] : prefill_catapult_fp_t(0.0f);
    }
    channel.write(packet);
  }
}

void unpack_gate_or_up_tile(
    const packed_w4_t* tile_words,
    int ff_base,
    int hidden_base,
    packed_w4_t* destination) {
  for (int ff_offset = 0; ff_offset < kHostMlpStreamCoreFfTile; ++ff_offset) {
    for (int hidden_offset = 0; hidden_offset < kHostMlpStreamCoreHiddenTile; ++hidden_offset) {
      const int tile_flat_index = ff_offset * kHostMlpStreamCoreHiddenTile + hidden_offset;
      const int destination_flat_index =
          (ff_base + ff_offset) * kHiddenSize + hidden_base + hidden_offset;
      set_packed_weight(destination, destination_flat_index, decode_packed_weight(tile_words, tile_flat_index));
    }
  }
}

void unpack_down_tile(
    const packed_w4_t* tile_words,
    int out_base,
    int ff_base,
    packed_w4_t* destination) {
  for (int out_offset = 0; out_offset < kHostMlpStreamCoreHiddenTile; ++out_offset) {
    for (int ff_offset = 0; ff_offset < kHostMlpStreamCoreFfTile; ++ff_offset) {
      const int tile_flat_index = out_offset * kHostMlpStreamCoreFfTile + ff_offset;
      const int destination_flat_index =
          (out_base + out_offset) * kIntermediateSize + ff_base + ff_offset;
      set_packed_weight(destination, destination_flat_index, decode_packed_weight(tile_words, tile_flat_index));
    }
  }
}

}  // namespace

KernelStatus qwen_prefill_attention_kernel_catapult(
    CatapultConstTensorView<prefill_catapult_fp_t> input_sequence,
    int seq_len,
    const PrefillAttentionTileConfig& tile_config,
    CatapultConstTensorView<prefill_catapult_fp_t> input_layernorm_weight,
    prefill_catapult_fp_t rms_eps,
    CatapultConstTensorView<packed_w4_t> q_packed_weights,
    CatapultConstTensorView<packed_w4_t> k_packed_weights,
    CatapultConstTensorView<packed_w4_t> v_packed_weights,
    CatapultConstTensorView<packed_w4_t> o_packed_weights,
    CatapultConstTensorView<prefill_catapult_fp_t> q_bias,
    CatapultConstTensorView<prefill_catapult_fp_t> k_bias,
    CatapultConstTensorView<prefill_catapult_fp_t> v_bias,
    CatapultConstTensorView<prefill_catapult_fp_t> q_scales,
    CatapultConstTensorView<prefill_catapult_fp_t> k_scales,
    CatapultConstTensorView<prefill_catapult_fp_t> v_scales,
    CatapultConstTensorView<prefill_catapult_fp_t> o_scales,
    CatapultTensorView<prefill_catapult_fp_t> k_cache,
    CatapultTensorView<prefill_catapult_fp_t> v_cache,
    CatapultTensorView<prefill_catapult_fp_t> output_sequence) {
  return qwen_prefill_attention_kernel(
      input_sequence.data,
      seq_len,
      tile_config,
      input_layernorm_weight.data,
      rms_eps,
      q_packed_weights.data,
      k_packed_weights.data,
      v_packed_weights.data,
      o_packed_weights.data,
      q_bias.data,
      k_bias.data,
      v_bias.data,
      q_scales.data,
      k_scales.data,
      v_scales.data,
      o_scales.data,
      k_cache.data,
      v_cache.data,
      output_sequence.data);
}

KernelStatus qwen_prefill_mlp_stream_core_catapult(
    int seq_len,
    prefill_catapult_fp_t rms_eps,
    ac_channel<PrefillStreamFpWordPacket>& attention_residual_chan,
    ac_channel<PrefillStreamFpWordPacket>& post_attention_layernorm_weight_chan,
    ac_channel<PrefillStreamPackedWordPacket>& gate_packed_weight_tile_chan,
    ac_channel<PrefillStreamPackedWordPacket>& up_packed_weight_tile_chan,
    ac_channel<PrefillStreamPackedWordPacket>& down_packed_weight_tile_chan,
    ac_channel<PrefillStreamFpWordPacket>& gate_scale_tile_chan,
    ac_channel<PrefillStreamFpWordPacket>& up_scale_tile_chan,
    ac_channel<PrefillStreamFpWordPacket>& down_scale_chan,
    ac_channel<PrefillStreamFpWordPacket>& output_sequence_chan) {
  if (seq_len <= 0 || seq_len > kPrefillCatapultSeqCapacity) {
    return {false, 2};
  }

  const PrefillMLPTileConfig tile_config = default_prefill_tile_config().mlp;

  std::vector<prefill_catapult_fp_t> attention_residual(
      static_cast<std::size_t>(kPrefillCatapultSeqCapacity) * kHiddenSize,
      prefill_catapult_fp_t(0.0f));
  std::vector<prefill_catapult_fp_t> post_attention_layernorm_weight(kHiddenSize, prefill_catapult_fp_t(0.0f));
  std::vector<prefill_catapult_fp_t> gate_scales(kIntermediateSize, prefill_catapult_fp_t(0.0f));
  std::vector<prefill_catapult_fp_t> up_scales(kIntermediateSize, prefill_catapult_fp_t(0.0f));
  std::vector<prefill_catapult_fp_t> down_scales(kHiddenSize, prefill_catapult_fp_t(0.0f));
  std::vector<packed_w4_t> gate_packed_weights(static_cast<std::size_t>(kIntermediateSize) * kHiddenSize / 2, 0);
  std::vector<packed_w4_t> up_packed_weights(static_cast<std::size_t>(kIntermediateSize) * kHiddenSize / 2, 0);
  std::vector<packed_w4_t> down_packed_weights(static_cast<std::size_t>(kIntermediateSize) * kHiddenSize / 2, 0);
  std::vector<prefill_catapult_fp_t> output_sequence(
      static_cast<std::size_t>(kPrefillCatapultSeqCapacity) * kHiddenSize,
      prefill_catapult_fp_t(0.0f));
  packed_w4_t packed_tile[kHostMlpStreamCorePackedTileWords] = {};
  prefill_catapult_fp_t scale_tile[kHostMlpStreamCoreFfTile] = {};

  read_fp_words(post_attention_layernorm_weight_chan, post_attention_layernorm_weight.data(), kHiddenSize);
  read_fp_words(down_scale_chan, down_scales.data(), kHiddenSize);

  for (int token_index = 0; token_index < seq_len; ++token_index) {
    read_fp_words(attention_residual_chan, attention_residual.data() + token_index * kHiddenSize, kHiddenSize);

    for (int ff_base = 0; ff_base < kIntermediateSize; ff_base += kHostMlpStreamCoreFfTile) {
      read_fp_words(gate_scale_tile_chan, scale_tile, kHostMlpStreamCoreFfTile);
      for (int ff_offset = 0; ff_offset < kHostMlpStreamCoreFfTile; ++ff_offset) {
        gate_scales[ff_base + ff_offset] = scale_tile[ff_offset];
      }

      read_fp_words(up_scale_tile_chan, scale_tile, kHostMlpStreamCoreFfTile);
      for (int ff_offset = 0; ff_offset < kHostMlpStreamCoreFfTile; ++ff_offset) {
        up_scales[ff_base + ff_offset] = scale_tile[ff_offset];
      }

      for (int hidden_base = 0; hidden_base < kHiddenSize; hidden_base += kHostMlpStreamCoreHiddenTile) {
        read_packed_words(gate_packed_weight_tile_chan, packed_tile, kHostMlpStreamCorePackedTileWords);
        unpack_gate_or_up_tile(packed_tile, ff_base, hidden_base, gate_packed_weights.data());

        read_packed_words(up_packed_weight_tile_chan, packed_tile, kHostMlpStreamCorePackedTileWords);
        unpack_gate_or_up_tile(packed_tile, ff_base, hidden_base, up_packed_weights.data());
      }

      for (int out_base = 0; out_base < kHiddenSize; out_base += kHostMlpStreamCoreHiddenTile) {
        read_packed_words(down_packed_weight_tile_chan, packed_tile, kHostMlpStreamCorePackedTileWords);
        unpack_down_tile(packed_tile, out_base, ff_base, down_packed_weights.data());
      }
    }
  }

  const KernelStatus status = qwen_prefill_mlp_kernel(
      attention_residual.data(),
      seq_len,
      tile_config,
      post_attention_layernorm_weight.data(),
      rms_eps,
      gate_packed_weights.data(),
      up_packed_weights.data(),
      down_packed_weights.data(),
      gate_scales.data(),
      up_scales.data(),
      down_scales.data(),
      output_sequence.data());
  if (!status.ok) {
    return status;
  }

  write_fp_words(output_sequence.data(), seq_len * kHiddenSize, output_sequence_chan);
  return status;
}

}  // namespace llm_accel

#endif