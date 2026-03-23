#pragma once

#include "../include/ac_channel.h"

#include "qwen_prefill_attention_kernel.h"
#include "qwen_prefill_stream_types.h"

namespace llm_accel {

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
    ac_channel<PrefillStreamFpWordPacket>& output_sequence_chan);

}  // namespace llm_accel
