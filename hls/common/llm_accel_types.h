#pragma once

#include <cstdint>

namespace llm_accel {

using packed_w4_t = std::uint8_t;
using act_t = std::int8_t;
using acc_t = std::int32_t;
using scalar_t = float;

struct KernelStatus {
  bool ok;
  int error_code;
};

}  // namespace llm_accel