#pragma once

#include "qwen_catapult_fp.h"

namespace llm_accel {

constexpr int kPrefillStreamFpWordsPerPacket = 8;
constexpr int kPrefillStreamPackedWordsPerPacket = 32;

struct PrefillStreamFpWordPacket {
  prefill_catapult_fp_t data[kPrefillStreamFpWordsPerPacket];
};

struct PrefillStreamPackedWordPacket {
  packed_w4_t data[kPrefillStreamPackedWordsPerPacket];
};

}  // namespace llm_accel
