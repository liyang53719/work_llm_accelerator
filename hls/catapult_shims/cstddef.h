#ifndef TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_CSTDDEF_H
#define TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_CSTDDEF_H

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

namespace std {
using ::size_t;
using ::ptrdiff_t;
}

#endif