#pragma once

#include "../include/ac_channel.h"

#include "qwen_prefill_attention_kernel.h"

namespace llm_accel {

constexpr int kAttentionFpWordsPerPacket = 8;
constexpr int kAttentionPackedWordsPerPacket = 32;

struct AttentionFpWordPacket {
  prefill_catapult_fp_t data[kAttentionFpWordsPerPacket];
};

struct AttentionPackedWordPacket {
  packed_w4_t data[kAttentionPackedWordsPerPacket];
};

KernelStatus qwen_prefill_attention_stream_top_catapult(
    int seq_len,
    prefill_catapult_fp_t rms_eps,
    ac_channel<AttentionFpWordPacket>& input_sequence_chan,
    ac_channel<AttentionFpWordPacket>& input_layernorm_weight_chan,
    ac_channel<AttentionPackedWordPacket>& q_packed_weight_chan,
    ac_channel<AttentionPackedWordPacket>& k_packed_weight_chan,
    ac_channel<AttentionPackedWordPacket>& v_packed_weight_chan,
    ac_channel<AttentionPackedWordPacket>& o_packed_weight_chan,
    ac_channel<AttentionFpWordPacket>& q_bias_chan,
    ac_channel<AttentionFpWordPacket>& k_bias_chan,
    ac_channel<AttentionFpWordPacket>& v_bias_chan,
    ac_channel<AttentionFpWordPacket>& q_scale_chan,
    ac_channel<AttentionFpWordPacket>& k_scale_chan,
    ac_channel<AttentionFpWordPacket>& v_scale_chan,
    ac_channel<AttentionFpWordPacket>& o_scale_chan,
    ac_channel<AttentionFpWordPacket>& k_cache_out_chan,
    ac_channel<AttentionFpWordPacket>& v_cache_out_chan,
    ac_channel<AttentionFpWordPacket>& output_sequence_chan);

}  // namespace llm_accel
