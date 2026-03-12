#pragma once

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
constexpr float kRmsNormEps = 1.0e-6f;
constexpr float kRopeTheta = 1000000.0f;

}  // namespace llm_accel