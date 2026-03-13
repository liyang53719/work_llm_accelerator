#ifndef TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_CSTRING_H
#define TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_CSTRING_H

typedef __SIZE_TYPE__ size_t;

extern "C" {
void *memcpy(void *dest, const void *src, size_t count);
void *memmove(void *dest, const void *src, size_t count);
void *memset(void *dest, int ch, size_t count);
int memcmp(const void *lhs, const void *rhs, size_t count);
}

namespace std {

using ::memcmp;
using ::memcpy;
using ::memmove;
using ::memset;

}  // namespace std

#endif