#include "qwen_prefill_host_wrapper.h"

#include <cstdint>
#include <vector>

#include "qwen_prefill_attention_kernel.h"
#include "qwen_prefill_glue_top_v1_catapult.h"
#include "qwen_prefill_mlp_kernel.h"
#include "qwen_prefill_top_wrapper.h"

using namespace llm_accel;

namespace {

PrefillAttentionTileConfig make_attention_tile_config(
    int seq_tile,
    int query_tile,
    int key_tile,
    int hidden_proj_tile,
    int kv_proj_tile,
    int head_dim_tile,
    int query_heads_parallel,
    int kv_heads_parallel) {
  return {
      seq_tile,
      query_tile,
      key_tile,
      hidden_proj_tile,
      kv_proj_tile,
      head_dim_tile,
      query_heads_parallel,
      kv_heads_parallel,
  };
}

PrefillMLPTileConfig make_mlp_tile_config(int seq_tile, int hidden_tile, int ff_tile) {
  return {
      seq_tile,
      hidden_tile,
      ff_tile,
  };
}

PrefillTileConfig make_prefill_tile_config(
    int attention_seq_tile,
    int attention_query_tile,
    int attention_key_tile,
    int attention_hidden_proj_tile,
    int attention_kv_proj_tile,
    int attention_head_dim_tile,
    int attention_query_heads_parallel,
    int attention_kv_heads_parallel,
    int mlp_seq_tile,
    int mlp_hidden_tile,
    int mlp_ff_tile) {
  return {
      make_attention_tile_config(
          attention_seq_tile,
          attention_query_tile,
          attention_key_tile,
          attention_hidden_proj_tile,
          attention_kv_proj_tile,
          attention_head_dim_tile,
          attention_query_heads_parallel,
          attention_kv_heads_parallel),
      make_mlp_tile_config(mlp_seq_tile, mlp_hidden_tile, mlp_ff_tile),
  };
}

int ceil_div_int(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

int decode_packed_weight(const std::uint8_t* packed_weights, int flat_index) {
  const std::uint8_t packed_value = packed_weights[flat_index / 2];
  const int nibble = (flat_index & 1) == 0 ? (packed_value & 0xF) : ((packed_value >> 4) & 0xF);
  return nibble >= 8 ? nibble - 16 : nibble;
}

void set_packed_weight(std::uint8_t* packed_weights, int flat_index, int value) {
  const std::uint8_t nibble = static_cast<std::uint8_t>(value & 0xF);
  std::uint8_t& packed_value = packed_weights[flat_index / 2];
  if ((flat_index & 1) == 0) {
    packed_value = static_cast<std::uint8_t>((packed_value & 0xF0) | nibble);
  } else {
    packed_value = static_cast<std::uint8_t>((packed_value & 0x0F) | (nibble << 4));
  }
}

void write_fp_words_to_channel(
    const float* source,
    int element_count,
    ac_channel<PrefillStreamFpWordPacket>& channel) {
  const int packet_count = ceil_div_int(element_count, kPrefillStreamFpWordsPerPacket);
  for (int packet_index = 0; packet_index < packet_count; ++packet_index) {
    PrefillStreamFpWordPacket packet;
    for (int word_index = 0; word_index < kPrefillStreamFpWordsPerPacket; ++word_index) {
      const int flat_index = packet_index * kPrefillStreamFpWordsPerPacket + word_index;
      packet.data[word_index] =
          flat_index < element_count ? prefill_catapult_fp_t(source[flat_index]) : prefill_catapult_fp_t(0.0f);
    }
    channel.write(packet);
  }
}

void write_packed_words_to_channel(
    const std::uint8_t* source,
    int element_count,
    ac_channel<PrefillStreamPackedWordPacket>& channel) {
  const int packet_count = ceil_div_int(element_count, kPrefillStreamPackedWordsPerPacket);
  for (int packet_index = 0; packet_index < packet_count; ++packet_index) {
    PrefillStreamPackedWordPacket packet;
    for (int word_index = 0; word_index < kPrefillStreamPackedWordsPerPacket; ++word_index) {
      const int flat_index = packet_index * kPrefillStreamPackedWordsPerPacket + word_index;
      packet.data[word_index] = flat_index < element_count ? source[flat_index] : 0;
    }
    channel.write(packet);
  }
}

void read_fp_words_from_channel(
    ac_channel<PrefillStreamFpWordPacket>& channel,
    float* destination,
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

void emit_mlp_stream_tiles(
    int seq_len,
    const float* post_attention_layernorm_weight,
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* down_packed_weights,
    const float* gate_scales,
    const float* up_scales,
    const float* down_scales,
    ac_channel<PrefillStreamFpWordPacket>& post_attention_layernorm_weight_chan,
    ac_channel<PrefillStreamPackedWordPacket>& gate_packed_weight_tile_chan,
    ac_channel<PrefillStreamPackedWordPacket>& up_packed_weight_tile_chan,
    ac_channel<PrefillStreamPackedWordPacket>& down_packed_weight_tile_chan,
    ac_channel<PrefillStreamFpWordPacket>& gate_scale_tile_chan,
    ac_channel<PrefillStreamFpWordPacket>& up_scale_tile_chan,
    ac_channel<PrefillStreamFpWordPacket>& down_scale_chan) {
  constexpr int kStreamCoreHiddenTile = 128;
  constexpr int kStreamCoreFfTile = 128;
  constexpr int kStreamCorePackedTileWords = kStreamCoreHiddenTile * kStreamCoreFfTile / 2;

  write_fp_words_to_channel(post_attention_layernorm_weight, kHiddenSize, post_attention_layernorm_weight_chan);
  write_fp_words_to_channel(down_scales, kHiddenSize, down_scale_chan);

  std::vector<std::uint8_t> packed_tile(kStreamCorePackedTileWords, 0);
  for (int token_index = 0; token_index < seq_len; ++token_index) {
    for (int ff_base = 0; ff_base < kIntermediateSize; ff_base += kStreamCoreFfTile) {
      write_fp_words_to_channel(gate_scales + ff_base, kStreamCoreFfTile, gate_scale_tile_chan);
      write_fp_words_to_channel(up_scales + ff_base, kStreamCoreFfTile, up_scale_tile_chan);

      for (int hidden_base = 0; hidden_base < kHiddenSize; hidden_base += kStreamCoreHiddenTile) {
        std::fill(packed_tile.begin(), packed_tile.end(), 0);
        for (int ff_offset = 0; ff_offset < kStreamCoreFfTile; ++ff_offset) {
          for (int hidden_offset = 0; hidden_offset < kStreamCoreHiddenTile; ++hidden_offset) {
            const int source_flat_index = (ff_base + ff_offset) * kHiddenSize + hidden_base + hidden_offset;
            const int tile_flat_index = ff_offset * kStreamCoreHiddenTile + hidden_offset;
            set_packed_weight(packed_tile.data(), tile_flat_index, decode_packed_weight(gate_packed_weights, source_flat_index));
          }
        }
        write_packed_words_to_channel(packed_tile.data(), kStreamCorePackedTileWords, gate_packed_weight_tile_chan);

        std::fill(packed_tile.begin(), packed_tile.end(), 0);
        for (int ff_offset = 0; ff_offset < kStreamCoreFfTile; ++ff_offset) {
          for (int hidden_offset = 0; hidden_offset < kStreamCoreHiddenTile; ++hidden_offset) {
            const int source_flat_index = (ff_base + ff_offset) * kHiddenSize + hidden_base + hidden_offset;
            const int tile_flat_index = ff_offset * kStreamCoreHiddenTile + hidden_offset;
            set_packed_weight(packed_tile.data(), tile_flat_index, decode_packed_weight(up_packed_weights, source_flat_index));
          }
        }
        write_packed_words_to_channel(packed_tile.data(), kStreamCorePackedTileWords, up_packed_weight_tile_chan);
      }

      for (int out_base = 0; out_base < kHiddenSize; out_base += kStreamCoreHiddenTile) {
        std::fill(packed_tile.begin(), packed_tile.end(), 0);
        for (int out_offset = 0; out_offset < kStreamCoreHiddenTile; ++out_offset) {
          for (int ff_offset = 0; ff_offset < kStreamCoreFfTile; ++ff_offset) {
            const int source_flat_index = (out_base + out_offset) * kIntermediateSize + ff_base + ff_offset;
            const int tile_flat_index = out_offset * kStreamCoreFfTile + ff_offset;
            set_packed_weight(packed_tile.data(), tile_flat_index, decode_packed_weight(down_packed_weights, source_flat_index));
          }
        }
        write_packed_words_to_channel(packed_tile.data(), kStreamCorePackedTileWords, down_packed_weight_tile_chan);
      }
    }
  }
}

}  // namespace

extern "C" int qwen_prefill_stub_forward(
    const float* input_sequence,
    int seq_len,
    int attention_seq_tile,
    int attention_query_tile,
    int attention_key_tile,
    int attention_hidden_proj_tile,
    int attention_kv_proj_tile,
    int attention_head_dim_tile,
    int attention_query_heads_parallel,
    int attention_kv_heads_parallel,
    int mlp_seq_tile,
    int mlp_hidden_tile,
    int mlp_ff_tile,
    float* output_sequence) {
  if (input_sequence == nullptr || output_sequence == nullptr || seq_len <= 0) {
    return 1;
  }

  const PrefillTileConfig tile_config = make_prefill_tile_config(
      attention_seq_tile,
      attention_query_tile,
      attention_key_tile,
      attention_hidden_proj_tile,
      attention_kv_proj_tile,
      attention_head_dim_tile,
      attention_query_heads_parallel,
      attention_kv_heads_parallel,
      mlp_seq_tile,
      mlp_hidden_tile,
      mlp_ff_tile);

  std::vector<float> attention_output(seq_len * kHiddenSize, 0.0f);
  std::vector<float> k_cache(seq_len * kNumKeyValueHeads * kHeadDim, 0.0f);
  std::vector<float> v_cache(seq_len * kNumKeyValueHeads * kHeadDim, 0.0f);
    std::vector<float> input_layernorm_weight(kHiddenSize, 1.0f);
    std::vector<float> post_attention_layernorm_weight(kHiddenSize, 1.0f);
    std::vector<packed_w4_t> q_weights(static_cast<std::size_t>(kHiddenSize) * kHiddenSize / 2, 0);
    std::vector<packed_w4_t> k_weights(static_cast<std::size_t>(kHiddenSize) * kNumKeyValueHeads * kHeadDim / 2, 0);
    std::vector<packed_w4_t> v_weights(static_cast<std::size_t>(kHiddenSize) * kNumKeyValueHeads * kHeadDim / 2, 0);
    std::vector<packed_w4_t> o_weights(static_cast<std::size_t>(kHiddenSize) * kHiddenSize / 2, 0);
    std::vector<packed_w4_t> gate_weights(static_cast<std::size_t>(kIntermediateSize) * kHiddenSize / 2, 0);
    std::vector<packed_w4_t> up_weights(static_cast<std::size_t>(kIntermediateSize) * kHiddenSize / 2, 0);
    std::vector<packed_w4_t> down_weights(static_cast<std::size_t>(kIntermediateSize) * kHiddenSize / 2, 0);
    std::vector<float> q_bias(kHiddenSize, 0.0f);
    std::vector<float> k_bias(kNumKeyValueHeads * kHeadDim, 0.0f);
    std::vector<float> v_bias(kNumKeyValueHeads * kHeadDim, 0.0f);
    std::vector<float> q_scales(kHiddenSize, 1.0f);
    std::vector<float> k_scales(kNumKeyValueHeads * kHeadDim, 1.0f);
    std::vector<float> v_scales(kNumKeyValueHeads * kHeadDim, 1.0f);
    std::vector<float> o_scales(kHiddenSize, 1.0f);
    std::vector<float> gate_scales(kIntermediateSize, 1.0f);
    std::vector<float> up_scales(kIntermediateSize, 1.0f);
    std::vector<float> down_scales(kHiddenSize, 1.0f);

  KernelStatus attention_status = qwen_prefill_attention_kernel(
      input_sequence,
      seq_len,
      tile_config.attention,
      input_layernorm_weight.data(),
      kRmsNormEps,
      q_weights.data(),
      k_weights.data(),
      v_weights.data(),
      o_weights.data(),
      q_bias.data(),
      k_bias.data(),
      v_bias.data(),
      q_scales.data(),
      k_scales.data(),
      v_scales.data(),
      o_scales.data(),
      k_cache.data(),
      v_cache.data(),
      attention_output.data());
  if (!attention_status.ok) {
    return attention_status.error_code;
  }

  KernelStatus mlp_status = qwen_prefill_mlp_kernel(
      attention_output.data(),
      seq_len,
      tile_config.mlp,
      post_attention_layernorm_weight.data(),
      kRmsNormEps,
      gate_weights.data(),
      up_weights.data(),
      down_weights.data(),
      gate_scales.data(),
      up_scales.data(),
      down_scales.data(),
      output_sequence);
  return mlp_status.ok ? 0 : mlp_status.error_code;
}

extern "C" int qwen_prefill_attention_smoke_forward(
    const float* input_sequence,
    int seq_len,
  int attention_seq_tile,
  int attention_query_tile,
  int attention_key_tile,
  int attention_hidden_proj_tile,
  int attention_kv_proj_tile,
  int attention_head_dim_tile,
  int attention_query_heads_parallel,
  int attention_kv_heads_parallel,
    const float* input_layernorm_weight,
    const std::uint8_t* q_packed_weights,
    const std::uint8_t* k_packed_weights,
    const std::uint8_t* v_packed_weights,
    const std::uint8_t* o_packed_weights,
    const float* q_bias,
    const float* k_bias,
    const float* v_bias,
    const float* q_scales,
    const float* k_scales,
    const float* v_scales,
    const float* o_scales,
    float* k_cache,
    float* v_cache,
    float* output_sequence) {
  const PrefillAttentionTileConfig tile_config = make_attention_tile_config(
      attention_seq_tile,
      attention_query_tile,
      attention_key_tile,
      attention_hidden_proj_tile,
      attention_kv_proj_tile,
      attention_head_dim_tile,
      attention_query_heads_parallel,
      attention_kv_heads_parallel);
  KernelStatus status = qwen_prefill_attention_kernel(
      input_sequence,
      seq_len,
      tile_config,
      input_layernorm_weight,
      kRmsNormEps,
      q_packed_weights,
      k_packed_weights,
      v_packed_weights,
      o_packed_weights,
      q_bias,
      k_bias,
      v_bias,
      q_scales,
      k_scales,
      v_scales,
      o_scales,
      k_cache,
      v_cache,
      output_sequence);
  return status.ok ? 0 : status.error_code;
}

extern "C" int qwen_prefill_mlp_smoke_forward(
    const float* attention_residual_sequence,
    int seq_len,
  int mlp_seq_tile,
  int mlp_hidden_tile,
  int mlp_ff_tile,
    const float* post_attention_layernorm_weight,
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* down_packed_weights,
    const float* gate_scales,
    const float* up_scales,
    const float* down_scales,
    float* output_sequence) {
  const PrefillMLPTileConfig tile_config = make_mlp_tile_config(mlp_seq_tile, mlp_hidden_tile, mlp_ff_tile);
  KernelStatus status = qwen_prefill_mlp_kernel(
      attention_residual_sequence,
      seq_len,
      tile_config,
      post_attention_layernorm_weight,
      kRmsNormEps,
      gate_packed_weights,
      up_packed_weights,
      down_packed_weights,
      gate_scales,
      up_scales,
      down_scales,
      output_sequence);
  return status.ok ? 0 : status.error_code;
}

extern "C" int qwen_prefill_layer_smoke_forward(
    const float* input_sequence,
    int seq_len,
  int attention_seq_tile,
  int attention_query_tile,
  int attention_key_tile,
  int attention_hidden_proj_tile,
  int attention_kv_proj_tile,
  int attention_head_dim_tile,
  int attention_query_heads_parallel,
  int attention_kv_heads_parallel,
  int mlp_seq_tile,
  int mlp_hidden_tile,
  int mlp_ff_tile,
    const float* input_layernorm_weight,
    const float* post_attention_layernorm_weight,
    const std::uint8_t* q_packed_weights,
    const std::uint8_t* k_packed_weights,
    const std::uint8_t* v_packed_weights,
    const std::uint8_t* o_packed_weights,
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* down_packed_weights,
    const float* q_bias,
    const float* k_bias,
    const float* v_bias,
    const float* q_scales,
    const float* k_scales,
    const float* v_scales,
    const float* o_scales,
    const float* gate_scales,
    const float* up_scales,
    const float* down_scales,
    float* k_cache,
    float* v_cache,
    float* output_sequence) {
  const PrefillTileConfig tile_config = make_prefill_tile_config(
      attention_seq_tile,
      attention_query_tile,
      attention_key_tile,
      attention_hidden_proj_tile,
      attention_kv_proj_tile,
      attention_head_dim_tile,
      attention_query_heads_parallel,
      attention_kv_heads_parallel,
      mlp_seq_tile,
      mlp_hidden_tile,
      mlp_ff_tile);
  std::vector<float> attention_output(static_cast<std::size_t>(seq_len) * kHiddenSize, 0.0f);
  KernelStatus attention_status = qwen_prefill_attention_kernel(
      input_sequence,
      seq_len,
      tile_config.attention,
      input_layernorm_weight,
      kRmsNormEps,
      q_packed_weights,
      k_packed_weights,
      v_packed_weights,
      o_packed_weights,
      q_bias,
      k_bias,
      v_bias,
      q_scales,
      k_scales,
      v_scales,
      o_scales,
      k_cache,
      v_cache,
      attention_output.data());
  if (!attention_status.ok) {
    return attention_status.error_code;
  }

  KernelStatus mlp_status = qwen_prefill_mlp_kernel(
      attention_output.data(),
      seq_len,
      tile_config.mlp,
      post_attention_layernorm_weight,
      kRmsNormEps,
      gate_packed_weights,
      up_packed_weights,
      down_packed_weights,
      gate_scales,
      up_scales,
      down_scales,
      output_sequence);
  return mlp_status.ok ? 0 : mlp_status.error_code;
}

extern "C" int qwen_prefill_glue_smoke_forward(
    const float* input_sequence,
    int seq_len,
    int attention_seq_tile,
    int attention_query_tile,
    int attention_key_tile,
    int attention_hidden_proj_tile,
    int attention_kv_proj_tile,
    int attention_head_dim_tile,
    int attention_query_heads_parallel,
    int attention_kv_heads_parallel,
    int mlp_seq_tile,
    int mlp_hidden_tile,
    int mlp_ff_tile,
    const float* input_layernorm_weight,
    const float* post_attention_layernorm_weight,
    const std::uint8_t* q_packed_weights,
    const std::uint8_t* k_packed_weights,
    const std::uint8_t* v_packed_weights,
    const std::uint8_t* o_packed_weights,
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* down_packed_weights,
    const float* q_bias,
    const float* k_bias,
    const float* v_bias,
    const float* q_scales,
    const float* k_scales,
    const float* v_scales,
    const float* o_scales,
    const float* gate_scales,
    const float* up_scales,
    const float* down_scales,
    float* k_cache,
    float* v_cache,
    float* output_sequence) {
  if (input_sequence == nullptr || input_layernorm_weight == nullptr || post_attention_layernorm_weight == nullptr ||
      q_packed_weights == nullptr || k_packed_weights == nullptr || v_packed_weights == nullptr ||
      o_packed_weights == nullptr || gate_packed_weights == nullptr || up_packed_weights == nullptr ||
      down_packed_weights == nullptr || q_bias == nullptr || k_bias == nullptr || v_bias == nullptr ||
      q_scales == nullptr || k_scales == nullptr || v_scales == nullptr || o_scales == nullptr ||
      gate_scales == nullptr || up_scales == nullptr || down_scales == nullptr || k_cache == nullptr ||
      v_cache == nullptr || output_sequence == nullptr || seq_len <= 0) {
    return 1;
  }

  const PrefillTileConfig tile_config = make_prefill_tile_config(
      attention_seq_tile,
      attention_query_tile,
      attention_key_tile,
      attention_hidden_proj_tile,
      attention_kv_proj_tile,
      attention_head_dim_tile,
      attention_query_heads_parallel,
      attention_kv_heads_parallel,
      mlp_seq_tile,
      mlp_hidden_tile,
      mlp_ff_tile);

  if (tile_config.attention.seq != default_prefill_tile_config().attention.seq ||
      tile_config.attention.query != default_prefill_tile_config().attention.query ||
      tile_config.attention.key != default_prefill_tile_config().attention.key ||
      tile_config.attention.hidden_proj != default_prefill_tile_config().attention.hidden_proj ||
      tile_config.attention.kv_proj != default_prefill_tile_config().attention.kv_proj ||
      tile_config.attention.head_dim != kHeadDim ||
      tile_config.attention.query_heads_parallel != default_prefill_tile_config().attention.query_heads_parallel ||
      tile_config.attention.kv_heads_parallel != default_prefill_tile_config().attention.kv_heads_parallel ||
      tile_config.mlp.seq != default_prefill_tile_config().mlp.seq ||
      tile_config.mlp.hidden != default_prefill_tile_config().mlp.hidden ||
      tile_config.mlp.ff != default_prefill_tile_config().mlp.ff) {
    return 2;
  }

  ac_channel<PrefillStreamFpWordPacket> input_sequence_chan;
  ac_channel<PrefillStreamFpWordPacket> input_layernorm_weight_chan;
  ac_channel<PrefillStreamPackedWordPacket> q_packed_weight_chan;
  ac_channel<PrefillStreamPackedWordPacket> k_packed_weight_chan;
  ac_channel<PrefillStreamPackedWordPacket> v_packed_weight_chan;
  ac_channel<PrefillStreamPackedWordPacket> o_packed_weight_chan;
  ac_channel<PrefillStreamFpWordPacket> q_bias_chan;
  ac_channel<PrefillStreamFpWordPacket> k_bias_chan;
  ac_channel<PrefillStreamFpWordPacket> v_bias_chan;
  ac_channel<PrefillStreamFpWordPacket> q_scale_chan;
  ac_channel<PrefillStreamFpWordPacket> k_scale_chan;
  ac_channel<PrefillStreamFpWordPacket> v_scale_chan;
  ac_channel<PrefillStreamFpWordPacket> o_scale_chan;
  ac_channel<PrefillStreamFpWordPacket> k_cache_out_chan;
  ac_channel<PrefillStreamFpWordPacket> v_cache_out_chan;
  ac_channel<PrefillStreamFpWordPacket> post_attention_layernorm_weight_chan;
  ac_channel<PrefillStreamPackedWordPacket> gate_packed_weight_tile_chan;
  ac_channel<PrefillStreamPackedWordPacket> up_packed_weight_tile_chan;
  ac_channel<PrefillStreamPackedWordPacket> down_packed_weight_tile_chan;
  ac_channel<PrefillStreamFpWordPacket> gate_scale_tile_chan;
  ac_channel<PrefillStreamFpWordPacket> up_scale_tile_chan;
  ac_channel<PrefillStreamFpWordPacket> down_scale_chan;
  ac_channel<PrefillStreamFpWordPacket> output_sequence_chan;

  write_fp_words_to_channel(input_sequence, seq_len * kHiddenSize, input_sequence_chan);
  write_fp_words_to_channel(input_layernorm_weight, kHiddenSize, input_layernorm_weight_chan);
  write_packed_words_to_channel(q_packed_weights, kHiddenSize * kHiddenSize / 2, q_packed_weight_chan);
  write_packed_words_to_channel(k_packed_weights, kPrefillCatapultKvWidth * kHiddenSize / 2, k_packed_weight_chan);
  write_packed_words_to_channel(v_packed_weights, kPrefillCatapultKvWidth * kHiddenSize / 2, v_packed_weight_chan);
  write_packed_words_to_channel(o_packed_weights, kHiddenSize * kHiddenSize / 2, o_packed_weight_chan);
  write_fp_words_to_channel(q_bias, kHiddenSize, q_bias_chan);
  write_fp_words_to_channel(k_bias, kPrefillCatapultKvWidth, k_bias_chan);
  write_fp_words_to_channel(v_bias, kPrefillCatapultKvWidth, v_bias_chan);
  write_fp_words_to_channel(q_scales, kHiddenSize, q_scale_chan);
  write_fp_words_to_channel(k_scales, kPrefillCatapultKvWidth, k_scale_chan);
  write_fp_words_to_channel(v_scales, kPrefillCatapultKvWidth, v_scale_chan);
  write_fp_words_to_channel(o_scales, kHiddenSize, o_scale_chan);

  emit_mlp_stream_tiles(
      seq_len,
      post_attention_layernorm_weight,
      gate_packed_weights,
      up_packed_weights,
      down_packed_weights,
      gate_scales,
      up_scales,
      down_scales,
      post_attention_layernorm_weight_chan,
      gate_packed_weight_tile_chan,
      up_packed_weight_tile_chan,
      down_packed_weight_tile_chan,
      gate_scale_tile_chan,
      up_scale_tile_chan,
      down_scale_chan);

  qwen_prefill_glue_top_v1_catapult(
      seq_len,
      prefill_catapult_fp_t(kRmsNormEps),
      input_sequence_chan,
      input_layernorm_weight_chan,
      q_packed_weight_chan,
      k_packed_weight_chan,
      v_packed_weight_chan,
      o_packed_weight_chan,
      q_bias_chan,
      k_bias_chan,
      v_bias_chan,
      q_scale_chan,
      k_scale_chan,
      v_scale_chan,
      o_scale_chan,
      k_cache_out_chan,
      v_cache_out_chan,
      post_attention_layernorm_weight_chan,
      gate_packed_weight_tile_chan,
      up_packed_weight_tile_chan,
      down_packed_weight_tile_chan,
      gate_scale_tile_chan,
      up_scale_tile_chan,
      down_scale_chan,
      output_sequence_chan);

  read_fp_words_from_channel(k_cache_out_chan, k_cache, seq_len * kPrefillCatapultKvWidth);
  read_fp_words_from_channel(v_cache_out_chan, v_cache, seq_len * kPrefillCatapultKvWidth);
  read_fp_words_from_channel(output_sequence_chan, output_sequence, seq_len * kHiddenSize);
  return 0;
}

extern "C" int qwen_prefill_top_smoke_forward(
    int layer_id,
    int seq_len,
  int attention_seq_tile,
  int attention_query_tile,
  int attention_key_tile,
  int attention_hidden_proj_tile,
  int attention_kv_proj_tile,
  int attention_head_dim_tile,
  int attention_query_heads_parallel,
  int attention_kv_heads_parallel,
  int mlp_seq_tile,
  int mlp_hidden_tile,
  int mlp_ff_tile,
    std::uint64_t input_sequence_addr,
    std::uint64_t output_sequence_addr,
    std::uint64_t layer_weights_base_addr,
    std::uint64_t layer_scales_base_addr,
    std::uint64_t k_cache_base_addr,
    std::uint64_t v_cache_base_addr,
    const std::uint8_t* weight_ddr,
    const float* scale_ddr,
    float* kv_cache_ddr,
    float* activation_ddr,
    std::uint8_t* weight_sram,
    float* kv_sram,
    std::int32_t* partial_sum_sram,
    float* softmax_sram,
    float* control_sram) {
  const std::uint64_t sequence_bytes = static_cast<std::uint64_t>(seq_len) * static_cast<std::uint64_t>(kHiddenSize) * sizeof(float);
  PrefillLayerDescriptor descriptor{
      layer_id,
      seq_len,
      make_prefill_tile_config(
          attention_seq_tile,
          attention_query_tile,
          attention_key_tile,
          attention_hidden_proj_tile,
          attention_kv_proj_tile,
          attention_head_dim_tile,
          attention_query_heads_parallel,
          attention_kv_heads_parallel,
          mlp_seq_tile,
          mlp_hidden_tile,
          mlp_ff_tile),
      input_sequence_addr,
      output_sequence_addr,
      layer_weights_base_addr,
      layer_scales_base_addr,
      k_cache_base_addr,
      v_cache_base_addr,
        output_sequence_addr + sequence_bytes,
  };
  PrefillTopLevelPorts ports{
      weight_ddr,
      scale_ddr,
      kv_cache_ddr,
      activation_ddr,
      weight_sram,
      kv_sram,
      partial_sum_sram,
      softmax_sram,
      control_sram,
  };
  KernelStatus status = qwen_prefill_top_wrapper(descriptor, ports);
  return status.ok ? 0 : status.error_code;
}