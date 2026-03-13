#ifndef TVM_WORK_LLM_ACCELERATOR_HLS_LIMITS_H_
#define TVM_WORK_LLM_ACCELERATOR_HLS_LIMITS_H_

#ifndef _GCC_LIMITS_H_
#define _GCC_LIMITS_H_
#endif

#ifndef CHAR_BIT
#define CHAR_BIT __CHAR_BIT__
#endif

#ifndef MB_LEN_MAX
#define MB_LEN_MAX 1
#endif

#ifndef SCHAR_MAX
#define SCHAR_MAX __SCHAR_MAX__
#endif
#ifndef SCHAR_MIN
#define SCHAR_MIN (-SCHAR_MAX - 1)
#endif

#ifndef UCHAR_MAX
#if __SCHAR_MAX__ == __INT_MAX__
#define UCHAR_MAX (SCHAR_MAX * 2U + 1U)
#else
#define UCHAR_MAX (SCHAR_MAX * 2 + 1)
#endif
#endif

#ifdef __CHAR_UNSIGNED__
#ifndef CHAR_MIN
#if __SCHAR_MAX__ == __INT_MAX__
#define CHAR_MIN 0U
#else
#define CHAR_MIN 0
#endif
#endif
#ifndef CHAR_MAX
#define CHAR_MAX UCHAR_MAX
#endif
#else
#ifndef CHAR_MIN
#define CHAR_MIN SCHAR_MIN
#endif
#ifndef CHAR_MAX
#define CHAR_MAX SCHAR_MAX
#endif
#endif

#ifndef SHRT_MAX
#define SHRT_MAX __SHRT_MAX__
#endif
#ifndef SHRT_MIN
#define SHRT_MIN (-SHRT_MAX - 1)
#endif

#ifndef USHRT_MAX
#if __SHRT_MAX__ == __INT_MAX__
#define USHRT_MAX (SHRT_MAX * 2U + 1U)
#else
#define USHRT_MAX (SHRT_MAX * 2 + 1)
#endif
#endif

#ifndef INT_MAX
#define INT_MAX __INT_MAX__
#endif
#ifndef INT_MIN
#define INT_MIN (-INT_MAX - 1)
#endif

#ifndef UINT_MAX
#define UINT_MAX (INT_MAX * 2U + 1U)
#endif

#ifndef LONG_MAX
#define LONG_MAX __LONG_MAX__
#endif
#ifndef LONG_MIN
#define LONG_MIN (-LONG_MAX - 1L)
#endif

#ifndef ULONG_MAX
#define ULONG_MAX (LONG_MAX * 2UL + 1UL)
#endif

#ifndef LLONG_MAX
#define LLONG_MAX __LONG_LONG_MAX__
#endif
#ifndef LLONG_MIN
#define LLONG_MIN (-LLONG_MAX - 1LL)
#endif

#ifndef ULLONG_MAX
#define ULLONG_MAX (LLONG_MAX * 2ULL + 1ULL)
#endif

namespace std {

template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<bool> {
	static const int digits = 1;
};

template <>
struct numeric_limits<char> {
	static const int digits = CHAR_BIT - ((char)-1 < 0 ? 1 : 0);
};

template <>
struct numeric_limits<signed char> {
	static const int digits = CHAR_BIT - 1;
};

template <>
struct numeric_limits<unsigned char> {
	static const int digits = CHAR_BIT;
};

template <>
struct numeric_limits<short> {
	static const int digits = sizeof(short) * CHAR_BIT - 1;
};

template <>
struct numeric_limits<unsigned short> {
	static const int digits = sizeof(unsigned short) * CHAR_BIT;
};

template <>
struct numeric_limits<int> {
	static const int digits = sizeof(int) * CHAR_BIT - 1;
};

template <>
struct numeric_limits<unsigned int> {
	static const int digits = sizeof(unsigned int) * CHAR_BIT;
};

template <>
struct numeric_limits<long> {
	static const int digits = sizeof(long) * CHAR_BIT - 1;
};

template <>
struct numeric_limits<unsigned long> {
	static const int digits = sizeof(unsigned long) * CHAR_BIT;
};

template <>
struct numeric_limits<long long> {
	static const int digits = sizeof(long long) * CHAR_BIT - 1;
};

template <>
struct numeric_limits<unsigned long long> {
	static const int digits = sizeof(unsigned long long) * CHAR_BIT;
};

template <>
struct numeric_limits<float> {
	static const int digits = __FLT_MANT_DIG__;
};

template <>
struct numeric_limits<double> {
	static const int digits = __DBL_MANT_DIG__;
};

template <>
struct numeric_limits<long double> {
	static const int digits = __LDBL_MANT_DIG__;
};

}  // namespace std

#endif