#pragma once

#include "qwen_prefill_top_core.h"
#include "qwen_catapult_fp.h"

namespace llm_accel {

constexpr std::size_t kPrefillWeightSramElements = kWeightBufferBytes / sizeof(packed_w4_t);
constexpr std::size_t kPrefillKvSramElements = kKvWorkingSetBytes / sizeof(scalar_t);
constexpr std::size_t kPrefillPartialSumSramElements = kPartialSumBytes / sizeof(acc_t);
constexpr std::size_t kPrefillSoftmaxSramElements = kSoftmaxScratchBytes / sizeof(scalar_t);
constexpr std::size_t kPrefillControlSramElements = kControlScratchBytes / sizeof(scalar_t);

KernelStatus qwen_prefill_top_catapult(
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
    std::uint64_t scratch_base_addr,
    const packed_w4_t weight_ddr[kPrefillWeightWindowBytes],
    const prefill_catapult_fp_t scale_ddr[kPrefillScaleWindowElements],
    prefill_catapult_fp_t kv_cache_ddr[kPrefillKvCacheWindowElements],
    prefill_catapult_fp_t activation_ddr[kPrefillActivationWindowElements],
    packed_w4_t weight_sram[kPrefillWeightSramElements],
    prefill_catapult_fp_t kv_sram[kPrefillKvSramElements],
    acc_t partial_sum_sram[kPrefillPartialSumSramElements],
    prefill_catapult_fp_t softmax_sram[kPrefillSoftmaxSramElements],
    prefill_catapult_fp_t control_sram[kPrefillControlSramElements]);

KernelStatus qwen_prefill_top_catapult_fine(
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
    std::uint64_t scratch_base_addr,
    const packed_w4_t weight_ddr[kPrefillWeightWindowBytes],
    const prefill_catapult_fp_t scale_ddr[kPrefillScaleWindowElements],
    prefill_catapult_fp_t kv_cache_ddr[kPrefillKvCacheWindowElements],
    prefill_catapult_fp_t activation_ddr[kPrefillActivationWindowElements],
    packed_w4_t weight_sram[kPrefillWeightSramElements],
    prefill_catapult_fp_t kv_sram[kPrefillKvSramElements],
    acc_t partial_sum_sram[kPrefillPartialSumSramElements],
    prefill_catapult_fp_t softmax_sram[kPrefillSoftmaxSramElements],
    prefill_catapult_fp_t control_sram[kPrefillControlSramElements]);

}  // namespace llm_accel