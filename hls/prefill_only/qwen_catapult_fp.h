#pragma once

#include "../common/llm_accel_types.h"

#ifdef __SYNTHESIS__
#include "../include/ac_std_float.h"
#endif

namespace llm_accel {

#ifdef __SYNTHESIS__
using prefill_catapult_fp_t = ac_std_float<32, 8>;
#else
using prefill_catapult_fp_t = scalar_t;
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