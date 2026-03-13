#ifndef TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_CMATH_H
#define TVM_WORK_LLM_ACCELERATOR_CATAPULT_SHIMS_CMATH_H

#ifndef FP_NAN
#define FP_NAN 0
#endif

#ifndef FP_INFINITE
#define FP_INFINITE 1
#endif

#ifndef FP_ZERO
#define FP_ZERO 2
#endif

#ifndef FP_SUBNORMAL
#define FP_SUBNORMAL 3
#endif

#ifndef FP_NORMAL
#define FP_NORMAL 4
#endif

extern "C" {
double sqrt(double value);
float sqrtf(float value);
double ceil(double value);
float ceilf(float value);
double floor(double value);
float floorf(float value);
float truncf(float value);
float roundf(float value);
float fmaf(float x, float y, float z);
}

#ifdef fpclassify
#undef fpclassify
#endif
#ifdef signbit
#undef signbit
#endif
#ifdef isnan
#undef isnan
#endif
#ifdef isfinite
#undef isfinite
#endif
#ifdef isnormal
#undef isnormal
#endif
#ifdef isinf
#undef isinf
#endif
#ifdef ceil
#undef ceil
#endif
#ifdef floor
#undef floor
#endif

namespace std {

inline unsigned int bit_cast_float(float value) {
  union float_bits_t {
    float f;
    unsigned int u;
  } bits;
  bits.f = value;
  return bits.u;
}

inline unsigned long long bit_cast_double(double value) {
  union double_bits_t {
    double d;
    unsigned long long u;
  } bits;
  bits.d = value;
  return bits.u;
}

inline int fpclassify(float value) {
  unsigned int bits = bit_cast_float(value);
  unsigned int exponent = (bits >> 23) & 0xffU;
  unsigned int mantissa = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    return mantissa ? FP_NAN : FP_INFINITE;
  }
  if (exponent == 0U) {
    return mantissa ? FP_SUBNORMAL : FP_ZERO;
  }
  return FP_NORMAL;
}

inline int fpclassify(double value) {
  unsigned long long bits = bit_cast_double(value);
  unsigned long long exponent = (bits >> 52) & 0x7ffULL;
  unsigned long long mantissa = bits & 0xfffffffffffffULL;
  if (exponent == 0x7ffULL) {
    return mantissa ? FP_NAN : FP_INFINITE;
  }
  if (exponent == 0ULL) {
    return mantissa ? FP_SUBNORMAL : FP_ZERO;
  }
  return FP_NORMAL;
}

inline bool signbit(float value) {
  return (bit_cast_float(value) >> 31) != 0;
}

inline bool signbit(double value) {
  return (bit_cast_double(value) >> 63) != 0;
}

inline bool isnan(float value) { return value != value; }
inline bool isnan(double value) { return value != value; }

inline bool isfinite(float value) {
  int cls = fpclassify(value);
  return cls == FP_NORMAL || cls == FP_SUBNORMAL || cls == FP_ZERO;
}

inline bool isfinite(double value) {
  int cls = fpclassify(value);
  return cls == FP_NORMAL || cls == FP_SUBNORMAL || cls == FP_ZERO;
}

inline bool isnormal(float value) { return fpclassify(value) == FP_NORMAL; }
inline bool isnormal(double value) { return fpclassify(value) == FP_NORMAL; }
inline bool isinf(float value) { return fpclassify(value) == FP_INFINITE; }
inline bool isinf(double value) { return fpclassify(value) == FP_INFINITE; }

inline int __builtin_isfinite(float value) { return isfinite(value) ? 1 : 0; }
inline int __builtin_isfinite(double value) { return isfinite(value) ? 1 : 0; }
inline int __builtin_isnormal(float value) { return isnormal(value) ? 1 : 0; }
inline int __builtin_isnormal(double value) { return isnormal(value) ? 1 : 0; }
inline int __builtin_signbit(float value) { return signbit(value) ? 1 : 0; }
inline int __builtin_signbit(double value) { return signbit(value) ? 1 : 0; }
inline int __builtin_isnan(float value) { return isnan(value) ? 1 : 0; }
inline int __builtin_isnan(double value) { return isnan(value) ? 1 : 0; }
inline int __builtin_isinf_sign(float value) { return isinf(value) ? (signbit(value) ? -1 : 1) : 0; }
inline int __builtin_isinf_sign(double value) { return isinf(value) ? (signbit(value) ? -1 : 1) : 0; }
inline float ceil(float value) { return ::ceilf(value); }
inline double ceil(double value) { return ::ceil(value); }
inline float floor(float value) { return ::floorf(value); }
inline double floor(double value) { return ::floor(value); }
inline float sqrt(float value) { return ::sqrtf(value); }
inline double sqrt(double value) { return ::sqrt(value); }

}  // namespace std

#endif