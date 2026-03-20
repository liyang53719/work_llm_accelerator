#pragma once

#include "llm_accel_types.h"

namespace llm_accel {

constexpr int kHiddenSize = 1536;
constexpr int kIntermediateSize = 8960;
constexpr int kNumHiddenLayers = 28;
constexpr int kNumAttentionHeads = 12;
constexpr int kNumKeyValueHeads = 2;
constexpr int kHeadDim = 128;
constexpr int kVocabSize = 151936;
constexpr int kMaxSequenceLength = 32768;

constexpr int kDecodeTileM = 1;
constexpr int kDefaultPrefillTileM = 64;
constexpr int kTileN = 128;
constexpr int kAttentionMacRows = 32;
constexpr int kAttentionMacCols = 64;
constexpr int kDefaultPrefillSeqTile = 128;
constexpr int kDefaultPrefillAttentionQueryTile = kAttentionMacRows;
constexpr int kDefaultPrefillAttentionKeyTile = kAttentionMacCols;
constexpr int kDefaultPrefillAttentionHiddenProjTile = kAttentionMacCols;
constexpr int kDefaultPrefillAttentionKvProjTile = kAttentionMacCols;
constexpr int kDefaultPrefillAttentionHeadDimTile = kAttentionMacCols;
constexpr int kDefaultPrefillAttentionQueryHeadsParallel = 2;
constexpr int kDefaultPrefillAttentionKvHeadsParallel = 1;
constexpr int kDefaultPrefillMLPHiddenTile = 256;
constexpr int kDefaultPrefillMLPFFTile = 640;
constexpr float kRmsNormEps = 1.0e-6f;
constexpr float kRopeTheta = 1000000.0f;

inline PrefillTileConfig default_prefill_tile_config() {
  return {
	  {
		  kDefaultPrefillSeqTile,
		  kDefaultPrefillAttentionQueryTile,
		  kDefaultPrefillAttentionKeyTile,
		  kDefaultPrefillAttentionHiddenProjTile,
		  kDefaultPrefillAttentionKvProjTile,
		  kDefaultPrefillAttentionHeadDimTile,
		  kDefaultPrefillAttentionQueryHeadsParallel,
		  kDefaultPrefillAttentionKvHeadsParallel,
	  },
	  {
		  kDefaultPrefillSeqTile,
		  kDefaultPrefillMLPHiddenTile,
		  kDefaultPrefillMLPFFTile,
	  },
  };
}

}  // namespace llm_accel
