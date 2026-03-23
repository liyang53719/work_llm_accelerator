#pragma once

#include "../include/ac_channel.h"

#include "qwen_prefill_mlp_kernel.h"
#include "qwen_prefill_stream_types.h"

namespace llm_accel {

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
    ac_channel<PrefillStreamFpWordPacket>& output_sequence_chan);

}  // namespace llm_accel
