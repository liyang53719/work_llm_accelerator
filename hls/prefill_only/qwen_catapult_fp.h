#pragma once

#include "../common/llm_accel_types.h"

#ifdef __SYNTHESIS__
#include <ac_int.h>
#include <ac_std_float.h>
#include <ccs_dw_fp_lib.h>
#endif

namespace llm_accel {

#ifdef __SYNTHESIS__
using prefill_catapult_fp_t = ac_std_float<32, 8>;
using prefill_catapult_fp_raw_t = ac_int<32, true>;
#else
using prefill_catapult_fp_t = scalar_t;
#endif

#ifdef __SYNTHESIS__
constexpr int kPrefillCatapultFpExpWidth = 8;
constexpr int kPrefillCatapultFpSigWidth = 32 - kPrefillCatapultFpExpWidth - 1;
constexpr int kPrefillCatapultFpIeeeCompliance = 0;

inline prefill_catapult_fp_t prefill_catapult_fp_const(float value) {
    return prefill_catapult_fp_t(value);
}

inline prefill_catapult_fp_t prefill_catapult_fp_const_int(int value) {
    return prefill_catapult_fp_t(value);
}

inline prefill_catapult_fp_raw_t prefill_catapult_fp_raw_bits(const prefill_catapult_fp_t& value) {
    return value.data_ac_int();
}

inline prefill_catapult_fp_t prefill_catapult_fp_from_raw_bits(const prefill_catapult_fp_raw_t& bits) {
    prefill_catapult_fp_t value;
    value.set_data(bits);
    return value;
}

inline ac_int<3, false> prefill_catapult_fp_rounding_mode() {
    return ac_int<3, false>(0);
}

inline prefill_catapult_fp_t prefill_catapult_fp_add(
        const prefill_catapult_fp_t& lhs,
        const prefill_catapult_fp_t& rhs) {
    prefill_catapult_fp_raw_t result_bits;
    ccs_dw_fp_add<kPrefillCatapultFpSigWidth, kPrefillCatapultFpExpWidth, kPrefillCatapultFpIeeeCompliance>(
            prefill_catapult_fp_raw_bits(lhs),
            prefill_catapult_fp_raw_bits(rhs),
            prefill_catapult_fp_rounding_mode(),
            result_bits);
    return prefill_catapult_fp_from_raw_bits(result_bits);
}

inline prefill_catapult_fp_t prefill_catapult_fp_sub(
        const prefill_catapult_fp_t& lhs,
        const prefill_catapult_fp_t& rhs) {
    prefill_catapult_fp_raw_t result_bits;
    ccs_dw_fp_sub<kPrefillCatapultFpSigWidth, kPrefillCatapultFpExpWidth, kPrefillCatapultFpIeeeCompliance>(
            prefill_catapult_fp_raw_bits(lhs),
            prefill_catapult_fp_raw_bits(rhs),
            prefill_catapult_fp_rounding_mode(),
            result_bits);
    return prefill_catapult_fp_from_raw_bits(result_bits);
}

inline prefill_catapult_fp_t prefill_catapult_fp_mul(
        const prefill_catapult_fp_t& lhs,
        const prefill_catapult_fp_t& rhs) {
    prefill_catapult_fp_raw_t result_bits;
    ccs_dw_fp_mult<kPrefillCatapultFpSigWidth, kPrefillCatapultFpExpWidth, kPrefillCatapultFpIeeeCompliance>(
            prefill_catapult_fp_raw_bits(lhs),
            prefill_catapult_fp_raw_bits(rhs),
            prefill_catapult_fp_rounding_mode(),
            result_bits);
    return prefill_catapult_fp_from_raw_bits(result_bits);
}

inline prefill_catapult_fp_t prefill_catapult_fp_div(
        const prefill_catapult_fp_t& lhs,
        const prefill_catapult_fp_t& rhs) {
    prefill_catapult_fp_raw_t result_bits;
    ccs_dw_fp_div<kPrefillCatapultFpSigWidth, kPrefillCatapultFpExpWidth, kPrefillCatapultFpIeeeCompliance>(
            prefill_catapult_fp_raw_bits(lhs),
            prefill_catapult_fp_raw_bits(rhs),
            prefill_catapult_fp_rounding_mode(),
            result_bits);
    return prefill_catapult_fp_from_raw_bits(result_bits);
}

inline prefill_catapult_fp_t prefill_catapult_fp_mac(
        const prefill_catapult_fp_t& lhs,
        const prefill_catapult_fp_t& rhs,
        const prefill_catapult_fp_t& acc) {
    prefill_catapult_fp_raw_t result_bits;
    ccs_dw_fp_mac<kPrefillCatapultFpSigWidth, kPrefillCatapultFpExpWidth, kPrefillCatapultFpIeeeCompliance>(
            prefill_catapult_fp_raw_bits(lhs),
            prefill_catapult_fp_raw_bits(rhs),
            prefill_catapult_fp_raw_bits(acc),
            prefill_catapult_fp_rounding_mode(),
            result_bits);
    return prefill_catapult_fp_from_raw_bits(result_bits);
}

inline prefill_catapult_fp_t prefill_catapult_fp_sqrt(const prefill_catapult_fp_t& value) {
    prefill_catapult_fp_raw_t result_bits;
    ccs_dw_fp_sqrt<kPrefillCatapultFpSigWidth, kPrefillCatapultFpExpWidth, kPrefillCatapultFpIeeeCompliance>(
            prefill_catapult_fp_raw_bits(value),
            prefill_catapult_fp_rounding_mode(),
            result_bits);
    return prefill_catapult_fp_from_raw_bits(result_bits);
}

inline void prefill_catapult_fp_cmp(
        const prefill_catapult_fp_t& lhs,
        const prefill_catapult_fp_t& rhs,
        ac_int<1, false>& aeqb,
        ac_int<1, false>& altb,
        ac_int<1, false>& agtb,
        ac_int<1, false>& unordered) {
    prefill_catapult_fp_raw_t z0_bits;
    prefill_catapult_fp_raw_t z1_bits;
    ccs_dw_fp_cmp<kPrefillCatapultFpSigWidth, kPrefillCatapultFpExpWidth, kPrefillCatapultFpIeeeCompliance>(
            prefill_catapult_fp_raw_bits(lhs),
            prefill_catapult_fp_raw_bits(rhs),
            0,
            aeqb,
            altb,
            agtb,
            unordered,
            z0_bits,
            z1_bits);
}

inline bool prefill_catapult_fp_eq(const prefill_catapult_fp_t& lhs, const prefill_catapult_fp_t& rhs) {
    ac_int<1, false> aeqb;
    ac_int<1, false> altb;
    ac_int<1, false> agtb;
    ac_int<1, false> unordered;
    prefill_catapult_fp_cmp(lhs, rhs, aeqb, altb, agtb, unordered);
    return aeqb != 0;
}

inline bool prefill_catapult_fp_lt(const prefill_catapult_fp_t& lhs, const prefill_catapult_fp_t& rhs) {
    ac_int<1, false> aeqb;
    ac_int<1, false> altb;
    ac_int<1, false> agtb;
    ac_int<1, false> unordered;
    prefill_catapult_fp_cmp(lhs, rhs, aeqb, altb, agtb, unordered);
    return altb != 0;
}

inline bool prefill_catapult_fp_gt(const prefill_catapult_fp_t& lhs, const prefill_catapult_fp_t& rhs) {
    ac_int<1, false> aeqb;
    ac_int<1, false> altb;
    ac_int<1, false> agtb;
    ac_int<1, false> unordered;
    prefill_catapult_fp_cmp(lhs, rhs, aeqb, altb, agtb, unordered);
    return agtb != 0;
}

inline bool prefill_catapult_fp_le(const prefill_catapult_fp_t& lhs, const prefill_catapult_fp_t& rhs) {
    return prefill_catapult_fp_lt(lhs, rhs) || prefill_catapult_fp_eq(lhs, rhs);
}
#endif

constexpr int kPrefillCatapultKvWidth = kNumKeyValueHeads * kHeadDim;
constexpr int kPrefillCatapultSeqCapacity = kDefaultPrefillSeqTile;
constexpr int kPrefillCatapultQueryCapacity = kDefaultPrefillAttentionQueryTile;
constexpr int kPrefillCatapultKeyCapacity = kDefaultPrefillAttentionKeyTile;
constexpr int kPrefillCatapultHiddenProjCapacity = kDefaultPrefillAttentionHiddenProjTile;
constexpr int kPrefillCatapultKvProjCapacity = kDefaultPrefillAttentionKvProjTile;
constexpr int kPrefillCatapultProjectionTileCapacity =
    kPrefillCatapultHiddenProjCapacity > kPrefillCatapultKvProjCapacity ? kPrefillCatapultHiddenProjCapacity
                                                                        : kPrefillCatapultKvProjCapacity;

}  // namespace llm_accel
