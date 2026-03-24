#include "qwen_prefill_glue_top_v1_catapult.h"

namespace llm_accel {

#pragma hls_design top
void qwen_prefill_glue_top_v1_catapult(
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
    ac_channel<PrefillStreamFpWordPacket>& post_attention_layernorm_weight_chan,
    ac_channel<PrefillStreamPackedWordPacket>& gate_packed_weight_tile_chan,
    ac_channel<PrefillStreamPackedWordPacket>& up_packed_weight_tile_chan,
    ac_channel<PrefillStreamPackedWordPacket>& down_packed_weight_tile_chan,
    ac_channel<PrefillStreamFpWordPacket>& gate_scale_tile_chan,
    ac_channel<PrefillStreamFpWordPacket>& up_scale_tile_chan,
    ac_channel<PrefillStreamFpWordPacket>& down_scale_chan,
    ac_channel<PrefillStreamFpWordPacket>& output_sequence_chan) {
    const int attention_seq_len = seq_len;
    const int mlp_seq_len = seq_len;
    const prefill_catapult_fp_t attention_rms_eps = rms_eps;
    const prefill_catapult_fp_t mlp_rms_eps = rms_eps;
    static ac_channel<PrefillStreamFpWordPacket> attention_residual_bridge_chan;

  qwen_prefill_attention_stream_top_catapult(
            attention_seq_len,
            attention_rms_eps,
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
      attention_residual_bridge_chan);

  qwen_prefill_mlp_stream_core_catapult(
      mlp_seq_len,
      mlp_rms_eps,
      attention_residual_bridge_chan,
      post_attention_layernorm_weight_chan,
      gate_packed_weight_tile_chan,
      up_packed_weight_tile_chan,
      down_packed_weight_tile_chan,
      gate_scale_tile_chan,
      up_scale_tile_chan,
      down_scale_chan,
      output_sequence_chan);
}

}  // namespace llm_accel