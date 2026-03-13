#ifndef TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_CSTDINT_H
#define TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_CSTDINT_H

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

namespace std {
using ::int8_t;
using ::int16_t;
using ::int32_t;
using ::int64_t;
using ::uint8_t;
using ::uint16_t;
using ::uint32_t;
using ::uint64_t;
}

#endif