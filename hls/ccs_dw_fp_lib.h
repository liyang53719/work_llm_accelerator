
//==================================================================
// File: ccs_dw_lib.h
// Description: provides a C++ interface to Designware(r) FP blocks.
//
// BETA version
//==================================================================

#pragma once

#include "ac_std_float.h"
#include "catapult_shims/iostream.h"

namespace std {

template <ac_q_mode QR>
struct test_qr_structs {
  #if __cplusplus > 199711L
  enum { match = false };
  #endif
};

template <>
struct test_qr_structs<AC_RND_CONV> {
  #if __cplusplus > 199711L
  enum { match = true };
  #else
  enum { Rounding_mode_not_supported };
  #endif
};

template <>
struct test_qr_structs<AC_TRN_ZERO> {
  #if __cplusplus > 199711L
  enum { match = true };
  #else
  enum { Rounding_mode_not_supported };
  #endif
};

}  // namespace std

//-------------------------------------------------------------------------------------------
// ccs_dw_map_nan() maps nan->inf if ieee_compliance = 0, or nan->nan if ieee_compliance = 1.
template<int ieee_compliance, int W, int E>
ac_std_float<W,E> ccs_dw_map_nan(const ac_std_float<W,E> &a)
{
  ac_std_float<W,E> r = a;
  if (!ieee_compliance) {
    if (a.isnan()) {
      ac_int<W,false> d = a.data_ac_int();
      d.set_slc(0, ac_int<W-E-1,false>(0)); // NaN => Inf
      r.set_data(d);
    }
  }
  return r;
}

template<int ieee_compliance, int W, int E>
void ccs_dw_map_recode_nan(ac_std_float<W,E> &z)
{
  // make changes to factor real behavior of the RTL
  //   IEEE allows different ways to encode NaN
  //      DW has different encoding than generic implementation in ac_std_float.h
  if (z.isnan()) {
    ac_int<W,false> d = ieee_compliance;  // ieee_compliance==0 => Inf, ieee_compliance==1 => NaN
    d.set_slc(W-E-1, ac_int<E,false>(-1));
    z.set_data(d);
  }
}
//-------------------------------------------------------------------------------------------
// ccs_dw_is_zero() returns true if the input is to be considered a zero value.
// If NoSubNormals = 0, it will return zero for zero inputs only.
// If NoSubNormals = 1, it will return zero for zero AND subnormal inputs.
// Note that NoSubnormals = !ieee_compliance.
template<bool NoSubNormals, int W, int E>
bool ccs_dw_is_zero(const ac_std_float<W,E> &a)
{
  if (!NoSubNormals)
  { return !a; }
  // Treat subnormals (exp field==0) as zero
  return !a.data_ac_int().template slc<E>(W-E-1);
}

//-------------------------------------------------------------------------------------------
// ccs_dw_map_nan_subn() takes care of NaN *and* subnormal mapping.
// It maps nan->inf if ieee_compliance = 0, or nan->nan if ieee_compliance = 1.
// It also maps subnormal inputs to zero if ieee_compliance = 0.
template<int ieee_compliance, int W, int E>
ac_std_float<W,E> ccs_dw_map_nan_subn(const ac_std_float<W,E> &a)
{
  enum { NoSubnormals = !ieee_compliance };

  ac_std_float<W,E> r = ccs_dw_map_nan<ieee_compliance>(a); // Take care of NaN mapping.
  if (ccs_dw_is_zero<NoSubnormals>(a)) {
    r = r.zero(); // Subnormal input => Zero, if ieee_compliance = 1.
  }
  return r;
}

//-------------------------------------------------------------------------------------------
// DW_fp_addsub
// Simulation Model (Also used for fp_add and fp_sub functions)
template<int ieee_compliance, int W, int E>
void ccs_dw_fp_addsub_sim_model(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, ac_int<3,false> rnd, ac_int<1, false> op, ac_std_float<W,E> &z)
{
  const bool NoSubNormals = !ieee_compliance;
  // ccs_dw_map_nan maps nan to inf if ieee_compliance = 0, to nan if ieee_compliance = 1.
  ac_std_float<W,E> a_t = ccs_dw_map_nan<ieee_compliance>(a);
  ac_std_float<W,E> b_t = ccs_dw_map_nan<ieee_compliance>(b);

  // Negating b and adding it to a is the same thing as subtracting b from a.
  b_t = (op == 0) ? b_t : -b_t;

  if (!rnd) {
    z = a_t.template add_generic<AC_RND_CONV,NoSubNormals>(b_t);   // call generic implementation for simulation
  } else if (rnd == 1) {
    z = a_t.template add_generic<AC_TRN_ZERO,NoSubNormals>(b_t);   // call generic implementation for simulation
  } else {
    AC_ASSERT(false, "Rounding type not supported");
  }

  ccs_dw_map_recode_nan<ieee_compliance>(z);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_addsub
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_addsub(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  const ac_int<1,false> op,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // Call simulation model.
  ccs_dw_fp_addsub_sim_model<ieee_compliance>(a_fl, b_fl, rnd, op, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_addsub
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_addsub(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  const ac_int<1,false> op,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // Call simulation model.
  ccs_dw_fp_addsub_sim_model<ieee_compliance>(a_fl, b_fl, rnd, op, z_fl);
  z = z_fl.data_ac_int();
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_addsub(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, const ac_int<1,false> &op)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = int(QR == AC_TRN_ZERO);
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_addsub<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, op, z);
  #else
  ccs_fp_addsub<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, op, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

//-------------------------------------------------------------------------------------------
// DW_fp_add
// DW_lp_piped_fp_add
// Simulation Model (used for both)
template<int ieee_compliance, int W, int E>
void ccs_dw_fp_add_sim_model(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, ac_int<3,false> rnd, ac_std_float<W,E> &z)
{
  // Call addsub simulation model with op = 0 for addition.
  const ac_int<1, false> op_add = 0;
  ccs_dw_fp_addsub_sim_model<ieee_compliance>(a, b, rnd, op_add, z);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_add
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_add(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // reuse sim model from fp_add
  ccs_dw_fp_add_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_add
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_add(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // reuse sim model from fp_add
  ccs_dw_fp_add_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_lp_piped_fp_add
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_lp_piped_fp_add(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // reuse sim model from fp_add
  ccs_dw_fp_add_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_lp_piped_fp_add
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_lp_piped_fp_add(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // reuse sim model from fp_add
  ccs_dw_fp_add_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_add(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_add<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #else
  ccs_fp_add<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> lp_piped_fp_add(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_lp_piped_fp_add<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #else
  ccs_lp_piped_fp_add<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

//-------------------------------------------------------------------------------------------
// DW_fp_sub
// Simulation Model
template<int ieee_compliance, int W, int E>
void ccs_dw_fp_sub_sim_model(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, ac_int<3,false> rnd, ac_std_float<W,E> &z)
{
  // Call addsub simulation model with op = 1 for subtraction.
  const ac_int<1, false> op_sub = 1;
  ccs_dw_fp_addsub_sim_model<ieee_compliance>(a, b, rnd, op_sub, z);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_sub
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_sub(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // reuse sim model from fp_sub
  ccs_dw_fp_sub_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_sub
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_sub(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // reuse sim model from fp_sub
  ccs_dw_fp_sub_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_sub(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_sub<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #else
  ccs_fp_sub<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

//-------------------------------------------------------------------------------------------
// DW_fp_div
// DW_lp_piped_fp_div
// Simulation Model (used for both)
template<int ieee_compliance, int W, int E>
void ccs_dw_fp_div_sim_model(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, ac_int<3,false> rnd, ac_std_float<W,E> &z)
{
  const bool NoSubNormals = !ieee_compliance;
  ac_std_float<W,E> a_t = ccs_dw_map_nan<ieee_compliance>(a);
  ac_std_float<W,E> b_t = ccs_dw_map_nan<ieee_compliance>(b);
  if (!rnd) {
    z = a_t.template div_generic<AC_RND_CONV,NoSubNormals>(b_t);   // call generic implementation for simulation
  } else if (rnd == 1) {
    z = a_t.template div_generic<AC_TRN_ZERO,NoSubNormals>(b_t);   // call generic implementation for simulation
  } else {
    AC_ASSERT(false, "Rounding type not supported");
  }

  ccs_dw_map_recode_nan<ieee_compliance>(z);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_div
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_div(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  ccs_dw_fp_div_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_div
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_div(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  ccs_dw_fp_div_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_lp_piped_fp_div
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_lp_piped_fp_div(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  ccs_dw_fp_div_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_lp_piped_fp_div
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_lp_piped_fp_div(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  ccs_dw_fp_div_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_div(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_div<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #else
  ccs_fp_div<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> lp_piped_fp_div(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_lp_piped_fp_div<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #else
  ccs_lp_piped_fp_div<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

//-------------------------------------------------------------------------------------------
// DW_fp_mac
template<int ieee_compliance, int W, int E>
void ccs_dw_fp_mac_sim_model(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, const ac_std_float<W,E> &c, ac_int<3,false> rnd, ac_std_float<W,E> &z)
{
  const bool NoSubNormals = !ieee_compliance;
  ac_std_float<W,E> a_t = ccs_dw_map_nan<ieee_compliance>(a);
  ac_std_float<W,E> b_t = ccs_dw_map_nan<ieee_compliance>(b);
  ac_std_float<W,E> c_t = ccs_dw_map_nan<ieee_compliance>(c);
  if (!rnd) {
    z = a_t.template fma_generic<AC_RND_CONV,NoSubNormals>(b_t,c_t);   // call generic implementation for simulation
  } else if (rnd == 1) {
    z = a_t.template fma_generic<AC_TRN_ZERO,NoSubNormals>(b_t,c_t);   // call generic implementation for simulation
  } else {
    AC_ASSERT(false, "Rounding type not supported");
  }

  ccs_dw_map_recode_nan<ieee_compliance>(z);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_mac
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_mac(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<sig_width+exp_width+1,true> &c,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, c_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  c_fl.set_data(c);
  // reuse sim model from fp_mac
  ccs_dw_fp_mac_sim_model<ieee_compliance>(a_fl, b_fl, c_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_mac
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_mac(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<sig_width+exp_width+1,true> &c,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, c_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  c_fl.set_data(c);
  // reuse sim model from fp_mac
  ccs_dw_fp_mac_sim_model<ieee_compliance>(a_fl, b_fl, c_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_mac(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, const ac_std_float<W,E> &c)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_mac<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), c.data_ac_int(), rnd, z);
  #else
  ccs_fp_mac<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), c.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

//-------------------------------------------------------------------------------------------
// DW_fp_mult
// DW_lp_piped_fp_mult
// Simulation Model (used for both)
template<int ieee_compliance, int W, int E>
void ccs_dw_fp_mult_sim_model(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, ac_int<3,false> rnd, ac_std_float<W,E> &z)
{
  const bool NoSubNormals = !ieee_compliance;
  ac_std_float<W,E> a_t = ccs_dw_map_nan<ieee_compliance>(a);
  ac_std_float<W,E> b_t = ccs_dw_map_nan<ieee_compliance>(b);
  if (!rnd) {
    z = a_t.template mult_generic<AC_RND_CONV,NoSubNormals>(b_t);   // call generic implementation for simulation
  } else if (rnd == 1) {
    z = a_t.template mult_generic<AC_TRN_ZERO,NoSubNormals>(b_t);   // call generic implementation for simulation
  } else {
    AC_ASSERT(false, "Rounding type not supported");
  }

  ccs_dw_map_recode_nan<ieee_compliance>(z);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_mult
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_mult(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // reuse sim model from fp_mult
  ccs_dw_fp_mult_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_mult
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_mult(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // reuse sim model from fp_mult
  ccs_dw_fp_mult_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_lp_piped_fp_mult
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_lp_piped_fp_mult(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // reuse sim model from fp_mult
  ccs_dw_fp_mult_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_lp_piped_fp_mult
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_lp_piped_fp_mult(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // reuse sim model from fp_mult
  ccs_dw_fp_mult_sim_model<ieee_compliance>(a_fl, b_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_mult(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_mult<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #else
  ccs_fp_mult<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> lp_piped_fp_mult(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_lp_piped_fp_mult<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #else
  ccs_lp_piped_fp_mult<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

//-------------------------------------------------------------------------------------------
// DW_fp_recip
// DW_lp_piped_fp_recip
//
// Simulation Model (used for both)
template<int ieee_compliance, int W, int E>
void ccs_dw_fp_recip_sim_model(const ac_std_float<W,E> &a, ac_int<3,false> rnd, ac_std_float<W,E> &z)
{
  ac_std_float<W,E> one = ac_std_float<W,E>::one();
  const bool NoSubNormals = !ieee_compliance;
  ac_std_float<W,E> a_t = ccs_dw_map_nan<ieee_compliance>(a);
  if (!rnd) {
    z = one.template div_generic<AC_RND_CONV,NoSubNormals>(a_t);   // call generic implementation for simulation
  } else if (rnd == 1) {
    z = one.template div_generic<AC_TRN_ZERO,NoSubNormals>(a_t);   // call generic implementation for simulation
  } else {
    AC_ASSERT(false, "Rounding type not supported");
  }

  ccs_dw_map_recode_nan<ieee_compliance>(z);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_recip
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_recip(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, z_fl;
  a_fl.set_data(a);
  // reuse sim model from fp_recip
  ccs_dw_fp_recip_sim_model<ieee_compliance>(a_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_recip
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_recip(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, z_fl;
  a_fl.set_data(a);
  // reuse sim model from fp_recip
  ccs_dw_fp_recip_sim_model<ieee_compliance>(a_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_lp_piped_fp_recip
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_lp_piped_fp_recip(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, z_fl;
  a_fl.set_data(a);
  // reuse sim model from fp_recip
  ccs_dw_fp_recip_sim_model<ieee_compliance>(a_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_lp_piped_fp_recip
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_lp_piped_fp_recip(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, z_fl;
  a_fl.set_data(a);
  // reuse sim model from fp_recip
  ccs_dw_fp_recip_sim_model<ieee_compliance>(a_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// non-pipelined version
// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_recip(const ac_std_float<W,E> &x)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_recip<sig_width, exp_width, ieee_compliance>(x.data_ac_int(), rnd, z);
  #else
  ccs_fp_recip<sig_width, exp_width, ieee_compliance>(x.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> lp_piped_fp_recip(const ac_std_float<W,E> &x)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_lp_piped_fp_recip<sig_width, exp_width, ieee_compliance>(x.data_ac_int(), rnd, z);
  #else
  ccs_lp_piped_fp_recip<sig_width, exp_width, ieee_compliance>(x.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

//-------------------------------------------------------------------------------------------
// DW_fp_sqrt
// Simulation Model (used for both)

template<int ieee_compliance, int W, int E>
void ccs_dw_fp_sqrt_sim_model(const ac_std_float<W,E> &a, ac_int<3,false> rnd, ac_std_float<W,E> &z)
{
  const bool NoSubNormals = !ieee_compliance;
  ac_std_float<W,E> a_t = ccs_dw_map_nan<ieee_compliance>(a);
  if (!rnd) {
    z = a_t.template sqrt_generic<AC_RND_CONV,NoSubNormals>();   // call generic implementation for simulation
  } else if (rnd == 1) {
    z = a_t.template sqrt_generic<AC_TRN_ZERO,NoSubNormals>();   // call generic implementation for simulation
  } else {
    AC_ASSERT(false, "Rounding type not supported");
  }

  ccs_dw_map_recode_nan<ieee_compliance>(z);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_sqrt
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_sqrt(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, z_fl;
  a_fl.set_data(a);
  // reuse sim model from fp_sqrt
  ccs_dw_fp_sqrt_sim_model<ieee_compliance>(a_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_sqrt
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_sqrt(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, z_fl;
  a_fl.set_data(a);
  // reuse sim model from fp_sqrt
  ccs_dw_fp_sqrt_sim_model<ieee_compliance>(a_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// non-pipelined version
// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_sqrt(const ac_std_float<W,E> &x)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_sqrt<sig_width, exp_width, ieee_compliance>(x.data_ac_int(), rnd, z);
  #else
  ccs_fp_sqrt<sig_width, exp_width, ieee_compliance>(x.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

//---------------------------------------------------------------------------------------------------
// ccs_dw_fxpt_invsqrt() performs the inverse square root operation on inputs in the domain
// (0.25, 1) and produces an output in the range of (1, 2). The normalized floating point
// input from ccs_dw_fp_invsqrt_sim_model() is passed to this function.
//
// This function produces the same output as a designware DW_inv_sqrt instance with prec_control = 0.

template<int W_io>
bool ccs_dw_fxpt_invsqrt (
  const ac_fixed<W_io, 0, false> InvSQRT_inp,
  ac_fixed<W_io, 1, false> &InvSQRT_out
)
{
  typedef ac_fixed<W_io, 0, false> T_in;
  typedef ac_fixed<W_io, 1, false> T_out;
  
  typedef typename ac::rt_2T<T_out, T_out>::mult T_out_sqr;
  // Find type needed to store the variable t later.
  typedef typename ac::rt_2T<T_out_sqr, T_in>::mult T_t; 
  
  ac_fixed<W_io, 1, false> b = 0.0;
  bool exact = false;
  
  for (int k = 0; k < W_io; k++) {
    ac_fixed<W_io, 1, false> b_t = b;
    b_t[W_io - k - 1] = 1; // Same as adding 2^-k, this sets the kth bit to 1.
    T_t t = b_t*b_t*InvSQRT_inp;
    // If 1 - b_t*b_t*a >= 0, the approximation is too small and we need to set the kth bit to 1.
    if (t <= 1) {
      b = b_t;
      if (t == 1) {
        exact = true;
        break; // The exact output has been found, so we don't need to stay in this loop any longer.
      }
    }
  }
  
  InvSQRT_out = b;
  
  return !exact; // Used later for rounding.
}

//-------------------------------------------------------------------------------------------
// DW_fp_invsqrt
// Simulation Model

template<int ieee_compliance, int W, int E>
void ccs_dw_fp_invsqrt_sim_model(const ac_std_float<W,E> &a, ac_int<3,false> rnd, ac_std_float<W,E> &z)
{
  AC_ASSERT(rnd == 0 || rnd == 1, "Rounding mode not supported.");

  const bool SIGNA = a.signbit(); // runtime constant, has sign bit of input.
  const ac_int<W, false> a_data = a.data_ac_int();
  const ac_int<E, false> EA = a_data.template slc<E>(W - E - 1); // extract exp bits.
  ac_int<E, false> exp_max = -1;
  const bool MAX_EXP_A = (EA == exp_max); // Is exponent all 1s?
  const ac_int<W - E - 1, false> SIGA = a_data.template slc<W - E - 1>(0);
  const bool ZerSig_A = (SIGA == 0); // Are all significand bits zero?
  typedef ac_int<1, false> one_bit_type;

  const bool Zero_A = ieee_compliance ? a == a.zero() : EA == 0; // Do we consider the input to be zero?
  // Do we consider the input to be subnormal?
  const bool Denorm_A = ieee_compliance ? (EA == 0 && !ZerSig_A) : false;
  const bool NaN_A = ieee_compliance ? a.isnan() : false; // Do we consider input to be NaN?
  const bool Inf_A = ieee_compliance ? a.isinf() : MAX_EXP_A; // Do we consider the input  to be inf?
  // NaN_Reg is set to NaN if IEEE compliant, set to +inf otherwise.
  ac_int<W, false> NaN_Reg = ieee_compliance;
  NaN_Reg.set_slc(W - E - 1, ac_int<E, false>(-1));
  const ac_int<W, false> Inf_Reg = a.inf().data_ac_int(); // Set to +inf.
  ac_int<W - E, false> MA = SIGA; // Copy paste significand bits to MA.
  MA[W - E - 1] = Denorm_A ? 0 : 1; // If the input is considered subnormal, set MSB to 0, else 1.

  ac_int<W, false> z_reg = 0;
  if (NaN_A || (SIGNA && !Zero_A)) {
    // If IEEE compliant, set output to NaN if input is nan or negative non-zero.
    // If not IEEE compliant, output is set to +inf if above input condition is satisfied.
    z_reg = NaN_Reg;
  } else if (Zero_A) {
    z_reg = Inf_Reg; // Tentatively set output to +inf, for zero input.
    z_reg[W - 1] = SIGNA; // If input is -0, set output to -inf.
  } else if (Inf_A) {
    z_reg = 0; // If input is +inf, set output to 0.
  } else {
    // Subtract exponent bias to get actual exponent value.
    const ac_int<E, true> actual_EA = EA - a.exp_bias;
    ac_int<E + 2, true> LZ_INA = 0;
    if (Denorm_A) {
      int leading_1 = MA.leading_sign().to_int();
      // LZ_INA must always be odd for denormalized inputs, in order to ensure that normalization and
      // denormalization w.r.t. ccs_dw_fxpt_invsqrt() is done correctly.
      LZ_INA = leading_1 - int(leading_1%2 == 0);
      MA <<= LZ_INA; // Start normalizing mantissa to make it equivalent to that of a normalized input.
    } else {
      LZ_INA = -1;
    }
    // If actual_EA is odd, a further left shift by 1 is required to normalize correctly.
    ac_int<W - E + 1, false> extended_MA =  ac_int<W - E + 1, false>(MA) << int(actual_EA[0]);
    // quarter_input is true if the input to ccs_dw_fxpt_invsqrt() is 0.25, which only happens
    // if the floating point input is a perfect square.
    bool quarter_input = extended_MA.template slc<W - E - 1>(0) == 0 && extended_MA[W - E] == 0;
    ac_int<E + 2, true> EM = -(actual_EA - LZ_INA - one_bit_type(quarter_input) + one_bit_type(Denorm_A));
    ac_int<E + 2, true> EZ = EM >> 1; // EZ will later be used for de-normalization.

    // All bits of extended_MA are copied to InvSQRT_inp.
    ac_fixed<W - E + 1, 0, false> InvSQRT_inp = 0.0;
    InvSQRT_inp.set_slc(0, extended_MA.template slc<W - E + 1>(0));
    ac_fixed<W - E + 1, 1, false> InvSQRT_out;
    bool sticky = ccs_dw_fxpt_invsqrt(InvSQRT_inp, InvSQRT_out);

    // If input is a perfect square, we disregard the output of ccs_dw_fxpt_invsqrt() and do our own
    // output handling.
    ac_fixed<W - E, 1, false> Mantissa = quarter_input ? 1.0 : InvSQRT_out;
    bool Round = !quarter_input && InvSQRT_out[0];
    bool LS = !quarter_input && InvSQRT_out[1];
    bool STK = !quarter_input && sticky;

    bool RND_eval = (rnd == 0) && Round && (LS || STK); // Is rounding Mantissa necessary?
    ac_fixed<W - E + 1, 2, false> temp_mantissa;
    if (RND_eval) {
      ac_fixed<W - E, 1, false> quant_val = 0.0;
      quant_val[0] = 1;
      temp_mantissa = Mantissa + quant_val;
    } else {
      temp_mantissa = Mantissa;
    }

    Mantissa = temp_mantissa;
    // Add exponent bias to the mantissa, as well as unity if the MSB of temp_mantissa is set to 1.
    EZ += a.exp_bias + int(temp_mantissa[W - E]);
    if (EZ >= exp_max) {
      z_reg = Inf_Reg; // Output is too big to represent with given floating point format, set it to +inf.
    } else {
      z_reg[W - 1] = 0; // Output is always positive.
      // Add exponent and mantissa bits to output at the appropriate indices.
      z_reg.set_slc(W - E - 1, EZ.template slc<E>(0));
      z_reg.set_slc(0, Mantissa.template slc<W - E - 1>(0));
    }
  }

  z.set_data(z_reg);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_invsqrt
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_invsqrt(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width + exp_width + 1, E = exp_width };
  ac_std_float<W,E> a_fl, z_fl;
  a_fl.set_data(a);
  // Call simulation model.
  ccs_dw_fp_invsqrt_sim_model<ieee_compliance>(a_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_invsqrt
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_invsqrt(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width + exp_width + 1, E = exp_width };
  ac_std_float<W,E> a_fl, z_fl;
  a_fl.set_data(a);
  // Call simulation model.
  ccs_dw_fp_invsqrt_sim_model<ieee_compliance>(a_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_invsqrt(const ac_std_float<W,E> &x)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = (QR == AC_TRN_ZERO);
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_invsqrt<sig_width, exp_width, ieee_compliance>(x.data_ac_int(), rnd, z);
  #else
  ccs_fp_invsqrt<sig_width, exp_width, ieee_compliance>(x.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

//-------------------------------------------------------------------------------------------
// DW_fp_cmp
// Simulation Model

template<int ieee_compliance, int W, int E>
void ccs_dw_fp_cmp_sim_model(
  const ac_std_float<W,E> &a,
  const ac_std_float<W,E> &b,
  const ac_int<1,false> &zctr,
  ac_int<1,false> &aeqb,
  ac_int<1,false> &altb,
  ac_int<1,false> &agtb,
  ac_int<1,false> &unordered,
  ac_std_float<W,E> &z0,
  ac_std_float<W,E> &z1
)
{
  // Map input NaNs and subnormals according to the ieee_compliance parameter.
  ac_std_float<W,E> a_t = ccs_dw_map_nan_subn<ieee_compliance>(a);
  ac_std_float<W,E> b_t = ccs_dw_map_nan_subn<ieee_compliance>(b);

  bool unordered_t = (a_t.isnan() || b_t.isnan()); // Set to true if a and/or b are NaN after mapping.
  bool aeqb_t = (a_t == b_t); // Set to true if inputs are equal after mapping.
  bool altb_t = (a_t < b_t); // Set to true if a < b after mapping.
  bool agtb_t = (a_t > b_t); // Set to true if a > b after mapping.

  aeqb = int(aeqb_t);
  altb = int(altb_t);
  agtb = int(agtb_t);
  unordered = int(unordered_t);

  if (zctr == 0) {
    // Corner cases:
    // If a == b, z0 = a and z1 = b.
    // If a and/or b is NaN, z0 = a and z1 = b.
    // In all other cases, z0 = Min(a, b) and z1 = Max(a, b).
    z0 = (altb_t || unordered_t || aeqb) ? a : b;
    z1 = (agtb_t&& !unordered_t && !aeqb) ? a : b;
  } else {
    // Corner cases:
    // If a == b, z0 = b and z1 = a.
    // If a and/or b is NaN, z0 = a and z1 = b.
    // In all other cases, z0 = Max(a, b) and z1 = Min(a, b).
    z0 = ((agtb_t || unordered_t) && !aeqb) ? a : b;
    z1 = ((altb_t&& !unordered_t) || aeqb) ? a : b;
  }
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_cmp
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_cmp (
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<1,false> &zctr,
  ac_int<1,false> &aeqb,
  ac_int<1,false> &altb,
  ac_int<1,false> &agtb,
  ac_int<1,false> &unordered,
  ac_int<sig_width+exp_width+1,true> &z0,
  ac_int<sig_width+exp_width+1,true> &z1
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z0_fl, z1_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // Call simulation model.
  ccs_dw_fp_cmp_sim_model<ieee_compliance>(a_fl, b_fl, zctr, aeqb, altb, agtb, unordered, z0_fl, z1_fl);
  z0 = z0_fl.data_ac_int();
  z1 = z1_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_cmp
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_cmp (
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<1,false> &zctr,
  ac_int<1,false> &aeqb,
  ac_int<1,false> &altb,
  ac_int<1,false> &agtb,
  ac_int<1,false> &unordered,
  ac_int<sig_width+exp_width+1,true> &z0,
  ac_int<sig_width+exp_width+1,true> &z1
)
{
  enum { W = sig_width+exp_width+1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, z0_fl, z1_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  // Call simulation model.
  ccs_dw_fp_cmp_sim_model<ieee_compliance>(a_fl, b_fl, zctr, aeqb, altb, agtb, unordered, z0_fl, z1_fl);
  z0 = z0_fl.data_ac_int();
  z1 = z1_fl.data_ac_int();
}

// Top-level API - handles conversion to ac_int, concatenation of relational outputs for map_to_operator.
template<int ieee_compliance, int W, int E>
void fp_cmp(
  const ac_std_float<W,E> &a,
  const ac_std_float<W,E> &b,
  const ac_int<1,false> &zctr,
  ac_int<4, false> &arelb, // aeqb, altb, agtb and unordered outputs are concatenated into 4-bit arelb.
  ac_std_float<W,E> &z0,
  ac_std_float<W,E> &z1
)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  ac_int<1, false> aeqb, altb, agtb, unordered;
  ac_int<W, true> z0_int, z1_int;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_cmp<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), zctr, aeqb, altb, agtb, unordered, z0_int, z1_int);
  #else
  ccs_fp_cmp<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), zctr, aeqb, altb, agtb, unordered, z0_int, z1_int);
  #endif
  // Concatenate all 4 relational outputs between a and b (i.e. aeqb, altb, agtb and unordered) into
  // 4-bit arelb output.
  arelb[0] = aeqb;
  arelb[1] = altb;
  arelb[2] = agtb;
  arelb[3] = unordered;
  z0.set_data(z0_int);
  z1.set_data(z1_int);
}

//-------------------------------------------------------------------------------------------
// DW_fp_sum3
// Simulation Model (used for both pipelined and non-pipelined version)

template<int ieee_compliance, int W, int E>
void ccs_dw_fp_sum3_sim_model(
  const ac_std_float<W,E> &a,
  const ac_std_float<W,E> &b,
  const ac_std_float<W,E> &c,
  const ac_int<3,false> rnd,
  ac_std_float<W,E> &z
)
{
  AC_ASSERT(rnd == 0 || rnd == 1, "Rounding mode not supported.");

  ac_int<W,false> nan_or_inf = ieee_compliance; // ieee_compliance==0 => Inf, ieee_compliance==1 => NaN
  nan_or_inf.set_slc(W - E - 1, ac_int<E, false>(-1));
  const ac_int<W, false> pinf = a.inf().data_ac_int(); // +inf
  const ac_int<W, false> ninf = (-(a.inf())).data_ac_int(); // -inf
  const ac_int<W, false> a_int = a.data_ac_int();
  const ac_int<W, false> b_int = b.data_ac_int();
  const ac_int<W, false> c_int = c.data_ac_int();
  const bool a_sign = a.signbit();
  const bool b_sign = b.signbit();
  const bool c_sign = c.signbit();
  // Are input values considered infinity?
  const bool a_is_inf = a.isinf() || (a.isnan() && !ieee_compliance);
  const bool b_is_inf = b.isinf() || (b.isnan() && !ieee_compliance);
  const bool c_is_inf = c.isinf() || (c.isnan() && !ieee_compliance);
  // Are input values considered NaNs?
  const bool a_is_nan = a.isnan() && ieee_compliance;
  const bool b_is_nan = b.isnan() && ieee_compliance;
  const bool c_is_nan = c.isnan() && ieee_compliance;
  // Extract input exponents.
  ac_int<E, false> a_exp = a_int.template slc<E>(W - E - 1);
  ac_int<E, false> b_exp = b_int.template slc<E>(W - E - 1);
  ac_int<E, false> c_exp = c_int.template slc<E>(W - E - 1);
  // Extract input significands.
  ac_int<W - E - 1, false> a_sig = a_int.template slc<W - E - 1>(0);
  ac_int<W - E - 1, false> b_sig = b_int.template slc<W - E - 1>(0);
  ac_int<W - E - 1, false> c_sig = c_int.template slc<W - E - 1>(0);

  enum { NoSubNormals = !ieee_compliance };

  // If input values are to be considered zeros, set the input significand to zero.

  bool a_is_zero = ccs_dw_is_zero<NoSubNormals>(a);
  if (a_is_zero) {
    a_sig = 0;
  }

  bool b_is_zero = ccs_dw_is_zero<NoSubNormals>(b);
  if (b_is_zero) {
    b_sig = 0;
  }

  bool c_is_zero = ccs_dw_is_zero<NoSubNormals>(c);
  if (c_is_zero) {
    c_sig = 0;
  }

  // If input values are to be considered subnormals, set the input exponent to 1.

  bool a_is_subn = !a_is_zero && a_exp == 0;
  if (a_is_subn) {
    a_exp = 1;
  }

  bool b_is_subn = !b_is_zero && b_exp == 0;
  if (b_is_subn) {
    b_exp = 1;
  }

  bool c_is_subn = !c_is_zero && c_exp == 0;
  if (c_is_subn) {
    c_exp = 1;
  }

  // z_int contains bitvector for final floating point output.
  ac_int<W, false> z_int;

  if (a_is_nan || b_is_nan || c_is_nan) {
    z_int = nan_or_inf; // If any input is NaN, output is NaN as well.
  }
  // If one or more input pairs are infinite and have opposite signs, the output will be set to NaN.
  // If not, the output will be set to +/-inf, depending on what the signs of the infinities are.
  else if (a_is_inf) {
    if ((b_is_inf && a_sign != b_sign) || (c_is_inf && a_sign != c_sign)) {
      z_int = nan_or_inf;
    } else {
      z_int = a_sign ? ninf : pinf;
    }
  } else if (b_is_inf) {
    if ((a_is_inf && a_sign != b_sign) || (c_is_inf && b_sign != c_sign)) {
      z_int = nan_or_inf;
    } else {
      z_int = b_sign ? ninf : pinf;
    }
  } else if (c_is_inf) {
    if ((a_is_inf && a_sign != c_sign) || (b_is_inf && b_sign != c_sign)) {
      z_int = nan_or_inf;
    } else {
      z_int = c_sign ? ninf : pinf;
    }
  } else if (a_is_zero && b_is_zero && c_is_zero) {
    // If all the inputs are zeros, the output is set to zero as well.
    z_int = 0;
    z_int[W - 1] = ieee_compliance && a_sign && b_sign && c_sign;
  }
  // If the addition is of the form x - x + y, the output will be the same as y.
  // If y is zero, the output will be positive zero.
  else if (a == -b) {
    z_int = c.data_ac_int();
    z_int[W - 1] = !c_is_zero && z_int[W - 1];
  } else if (b == -c) {
    z_int = a.data_ac_int();
    z_int[W - 1] = !a_is_zero && z_int[W - 1];
  } else if (a == -c) {
    z_int = b.data_ac_int();
    z_int[W - 1] = !b_is_zero && z_int[W - 1];
  } else {
    // The *_sig_plus_MSB variables have an MSB added to the significand. The MSB is set to zero if the
    // input associated is zero or subnormal, and it's set to one otherwise.
    ac_int<W - E, false> a_sig_plus_MSB = a_sig;
    a_sig_plus_MSB[W - E - 1] = !(a_is_subn || a_is_zero);
    ac_int<W - E, false> b_sig_plus_MSB = b_sig;
    b_sig_plus_MSB[W - E - 1] = !(b_is_subn || b_is_zero);
    ac_int<W - E, false> c_sig_plus_MSB = c_sig;
    c_sig_plus_MSB[W - E - 1] = !(c_is_subn || c_is_zero);
    // Find the maximum exponent value. Subtract the input exponent values with this maximum value,
    // which will indicate how much you have to right-shift each of the input significands by.
    ac_int<E, false> max_eval = AC_MAX(AC_MAX(a_exp, b_exp), c_exp);
    // RSA stands for "Right Shift Amount", in other words, the amount to right-shift by.
    const ac_int<E, false> a_sig_RSA = max_eval - a_exp;
    const ac_int<E, false> b_sig_RSA = max_eval - b_exp;
    const ac_int<E, false> c_sig_RSA = max_eval - c_exp;

    // Pad *_sig_plus_MSB variables with sig_bits + 4 LSBs, all of which are set to zero, and given the
    // *_lsb_pad appendix. These variables are the right-shifted according to the values above, in order
    // to align all the significands correctly for addition later.

    // *_RS_flag variables indicate the loss of 1-bits while right-shifting, which later helps in
    // figuring out whether the output should be rounded or not.

    ac_int<2*W - 2*E + 3, false> a_sig_lsb_pad = 0;
    a_sig_lsb_pad.set_slc(W - E + 3, a_sig_plus_MSB);
    int a_sig_slc_LSB = AC_MIN(AC_MAX(a_sig_RSA.to_int() - W + E + 1, 0), W - E + 3);
    bool a_RS_flag = a_sig_lsb_pad.template slc<W - E>(a_sig_slc_LSB) != 0;
    a_sig_lsb_pad >>= a_sig_RSA;

    ac_int<2*W - 2*E + 3, false> b_sig_lsb_pad = 0;
    b_sig_lsb_pad.set_slc(W - E + 3, b_sig_plus_MSB);
    int b_sig_slc_LSB = AC_MIN(AC_MAX(b_sig_RSA.to_int() - W + E + 1, 0), W - E + 3);
    bool b_RS_flag = b_sig_lsb_pad.template slc<W - E>(b_sig_slc_LSB) != 0;
    b_sig_lsb_pad >>= b_sig_RSA;

    ac_int<2*W - 2*E + 3, false> c_sig_lsb_pad = 0;
    c_sig_lsb_pad.set_slc(W - E + 3, c_sig_plus_MSB);
    int c_sig_slc_LSB = AC_MIN(AC_MAX(c_sig_RSA.to_int() - W + E + 1, 0), W - E + 3);
    bool c_RS_flag = c_sig_lsb_pad.template slc<W - E>(c_sig_slc_LSB) != 0;
    c_sig_lsb_pad >>= c_sig_RSA;

    // Concatenate significand with exp for comparison purposes.

    ac_int<W, false> a_exp_sig_concat;
    a_exp_sig_concat.set_slc(0, a_sig_plus_MSB);
    a_exp_sig_concat.set_slc(W - E, a_exp);

    ac_int<W, false> b_exp_sig_concat;
    b_exp_sig_concat.set_slc(0, b_sig_plus_MSB);
    b_exp_sig_concat.set_slc(W - E, b_exp);

    ac_int<W, false> c_exp_sig_concat;
    c_exp_sig_concat.set_slc(0, c_sig_plus_MSB);
    c_exp_sig_concat.set_slc(W - E, c_exp);

    // The flags below help us decide whether to set the lsb of the *_sig_lsb_pad variables to 1.
    bool set_lsb_flag_1 = false, set_lsb_flag_2 = false, set_lsb_flag_3 = false;

    if (a_RS_flag && b_RS_flag) {
      set_lsb_flag_1 = a_exp_sig_concat < b_exp_sig_concat;
      set_lsb_flag_2 = !set_lsb_flag_1;
      set_lsb_flag_3 = false;
    }

    if (b_RS_flag && c_RS_flag) {
      set_lsb_flag_2 = b_exp_sig_concat < c_exp_sig_concat;
      set_lsb_flag_3 = !set_lsb_flag_2;
      set_lsb_flag_1 = false;
    }

    if (a_RS_flag && c_RS_flag) {
      set_lsb_flag_1 = a_exp_sig_concat < c_exp_sig_concat;
      set_lsb_flag_3 = !set_lsb_flag_1;
      set_lsb_flag_2 = false;
    }

    a_sig_lsb_pad[0] = !set_lsb_flag_1 && a_RS_flag;
    b_sig_lsb_pad[0] = !set_lsb_flag_2 && b_RS_flag;
    c_sig_lsb_pad[0] = !set_lsb_flag_3 && c_RS_flag;

    // Add three more MSBs to *_lsb_pad variables.
    ac_int<2*W - 2*E + 6, false> a_sig_bs_pad = a_sig_lsb_pad;
    ac_int<2*W - 2*E + 6, false> b_sig_bs_pad = b_sig_lsb_pad;
    ac_int<2*W - 2*E + 6, false> c_sig_bs_pad = c_sig_lsb_pad;

    a_sig_bs_pad = a_sign ? ac_int<2*W - 2*E + 6, false>(~a_sig_bs_pad + 1) : a_sig_bs_pad;
    b_sig_bs_pad = b_sign ? ac_int<2*W - 2*E + 6, false>(~b_sig_bs_pad + 1) : b_sig_bs_pad;
    c_sig_bs_pad = c_sign ? ac_int<2*W - 2*E + 6, false>(~c_sig_bs_pad + 1) : c_sig_bs_pad;

    // Add all the appropriately aligned and (if input is negative) negated significands together.
    ac_int<2*W - 2*E + 6, false> sig_add_res = a_sig_bs_pad + b_sig_bs_pad + c_sig_bs_pad;

    bool sig_ares_sign = sig_add_res[2*W - 2*E + 5];

    ac_int<2*W - 2*E + 5, false> abs_sig_ares;

    if (sig_ares_sign) {
      // If the addition result is negative, the 2s complement is stored in abs_sig_ares.
      abs_sig_ares = ~sig_add_res + 1;
    } else {
      abs_sig_ares = sig_add_res;
    }

    ac_int<E + 2, false> z_exp = max_eval;

    // Left-shift abs_sig_ares such that the leading one coincides with the 3rd MSB.
    int ares_LSA = abs_sig_ares.leading_sign().to_int() - 2;

    bool ares_RS_flag = false;

    if (ares_LSA > 0) {
      // Note that z_exp - 1 can never be less than 0 because z_exp can never be less than 1.
      ares_LSA = AC_MIN(ares_LSA, z_exp.to_int() - 1);
    } else if (ares_LSA < 0) {
      ares_RS_flag = (ares_LSA == -2 && abs_sig_ares[1]) || abs_sig_ares[0];
    }

    abs_sig_ares <<= ares_LSA;
    // Adjust z_exp according to how much we left-shifted by.
    z_exp -= ares_LSA;

    bool rnd_param3 = abs_sig_ares[W - E + 3];
    bool rnd_param4 = abs_sig_ares[W - E + 2];
    bool rnd_param5 = abs_sig_ares.template slc<W - E + 2>(0) != 0;
    rnd_param5 = rnd_param5 || a_RS_flag || b_RS_flag || c_RS_flag || ares_RS_flag;

    // rnd_out takes into account multiple factors to decide whether the output should be rounded or not.
    bool rnd_out = (rnd == 0 && rnd_param4 && (rnd_param3 || rnd_param5));

    if (rnd_out) {
      ac_int<W - E + 4, false> rnd_add = 0;
      rnd_add[W - E + 3] = 1;
      abs_sig_ares += rnd_add;

      // There's a chance that the second abs_sig_ares MSB is set to 1 after rounding is performed. If
      // that happens, we must right-shift the result by 1 to ensure correct alignment for slicing later.
      z_exp += int(abs_sig_ares[2*W - 2*E + 3]);
      abs_sig_ares >>= int(abs_sig_ares[2*W - 2*E + 3]);
    }

    ac_int<E, false> exp_max = -1; // Stores max. possible exp value, i.e. all 1s exp.

    if (abs_sig_ares == 0) {
      z_int = 0; // Zero addition result => zero output.
    }
    // If the bit at index [2*W - 2*E + 2] is zero, that means that the output will be subnormal.
    else if (abs_sig_ares[2*W - 2*E + 2] == 0) {
      z_int = 0;
      z_int[W - 1] = sig_ares_sign;
      // Output is left at 0 if ieee_compliance = 0. If ieee_compliance = 1, addition result bits are
      // sliced into the output.
      if (ieee_compliance) {
        z_int.set_slc(0, abs_sig_ares.template slc<W - E - 1>(W - E + 3));
      }
    } else {
      if (z_exp >= exp_max) {
        // If z_exp equals or exceeds exp_max, the output is set to the maximum possible magnitude.
        // For rnd = 0, the max. possible magnitude is inf.
        // For rnd = 1, the max. possible magnitude is (1.11111...)*2^(max_exp_that_isnt_all_1s-1-bias)
        if (rnd == 0) {
          abs_sig_ares = 0;
          z_exp = exp_max;
        } else {
          abs_sig_ares.template set_val<AC_VAL_MAX>();
          z_exp = exp_max - 1;
        }
      } else if (z_exp == 0) {
        z_exp = 1;
      }
      z_int[W - 1] = sig_ares_sign; // Sign of output = sign of sig_add_res.
      z_int.set_slc(W - E - 1, z_exp.template slc<E>(0));
      z_int.set_slc(0, abs_sig_ares.template slc<W - E - 1>(W - E + 3));
    }
  }

  z.set_data(z_int);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_sum3
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_sum3(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<sig_width+exp_width+1,true> &c,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width + exp_width + 1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, c_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  c_fl.set_data(c);
  // Call simulation model.
  ccs_dw_fp_sum3_sim_model<ieee_compliance>(a_fl, b_fl, c_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_sum3
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_sum3(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<sig_width+exp_width+1,true> &c,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width + exp_width + 1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, c_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  c_fl.set_data(c);
  // Call simulation model.
  ccs_dw_fp_sum3_sim_model<ieee_compliance>(a_fl, b_fl, c_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_lp_piped_fp_sum3
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_lp_piped_fp_sum3(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<sig_width+exp_width+1,true> &c,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width + exp_width + 1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, c_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  c_fl.set_data(c);
  // reuse sim model from fp_sum3
  ccs_dw_fp_sum3_sim_model<ieee_compliance>(a_fl, b_fl, c_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_lp_piped_fp_sum3
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_lp_piped_fp_sum3(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<sig_width+exp_width+1,true> &c,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width + exp_width + 1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, c_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  c_fl.set_data(c);
  // reuse sim model from fp_sum3
  ccs_dw_fp_sum3_sim_model<ieee_compliance>(a_fl, b_fl, c_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// non-pipelined version
// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_sum3(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, const ac_std_float<W,E> &c)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #ifndef __SYNTHESIS__
  static bool print_once = true;
  if (print_once && sig_width >= 15) {
    // The DW_fp_sum4 VHDL simulation model uses the conv_integer function, and on platforms with 32-bit
    // integers, any significand width equal to or exceeding 15 will throw an error during VHDL
    // simulation because the bitvector passed to conv_integer will be too large.
    std::cout << "WARNING: Consider reducing the significand width to a value below 15 to ensure simulation with the VHDL DW_fp_sum3 module, if that is being used." << std::endl;
    print_once = false;
  }
  #endif
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_sum3<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), c.data_ac_int(), rnd, z);
  #else
  ccs_fp_sum3<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), c.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> lp_piped_fp_sum3(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, const ac_std_float<W,E> &c)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_lp_piped_fp_sum3<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), c.data_ac_int(), rnd, z);
  #else
  ccs_lp_piped_fp_sum3<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), c.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}

//-------------------------------------------------------------------------------------------
// DW_fp_sum4
// Simulation model.

template<int ieee_compliance, int W, int E>
void ccs_dw_fp_sum4_sim_model(
  const ac_std_float<W,E> &a,
  const ac_std_float<W,E> &b,
  const ac_std_float<W,E> &c,
  const ac_std_float<W,E> &d,
  const ac_int<3,false> rnd,
  ac_std_float<W,E> &z
)
{
  AC_ASSERT(rnd == 0 || rnd == 1, "Rounding mode not supported.");

  ac_int<W,false> nan_or_inf = ieee_compliance; // ieee_compliance==0 => Inf, ieee_compliance==1 => NaN
  nan_or_inf.set_slc(W - E - 1, ac_int<E, false>(-1));
  const ac_int<W, false> pinf = a.inf().data_ac_int(); // +inf
  const ac_int<W, false> ninf = (-(a.inf())).data_ac_int(); // -inf
  const ac_int<W, false> a_int = a.data_ac_int();
  const ac_int<W, false> b_int = b.data_ac_int();
  const ac_int<W, false> c_int = c.data_ac_int();
  const ac_int<W, false> d_int = d.data_ac_int();
  const bool a_sign = a.signbit();
  const bool b_sign = b.signbit();
  const bool c_sign = c.signbit();
  const bool d_sign = d.signbit();
  // Are input values considered infinity?
  const bool a_is_inf = a.isinf() || (a.isnan() && !ieee_compliance);
  const bool b_is_inf = b.isinf() || (b.isnan() && !ieee_compliance);
  const bool c_is_inf = c.isinf() || (c.isnan() && !ieee_compliance);
  const bool d_is_inf = d.isinf() || (d.isnan() && !ieee_compliance);
  // Are input values considered NaNs?
  const bool a_is_nan = a.isnan() && ieee_compliance;
  const bool b_is_nan = b.isnan() && ieee_compliance;
  const bool c_is_nan = c.isnan() && ieee_compliance;
  const bool d_is_nan = d.isnan() && ieee_compliance;
  // Extract input exponents.
  ac_int<E, false> a_exp = a_int.template slc<E>(W - E - 1);
  ac_int<E, false> b_exp = b_int.template slc<E>(W - E - 1);
  ac_int<E, false> c_exp = c_int.template slc<E>(W - E - 1);
  ac_int<E, false> d_exp = d_int.template slc<E>(W - E - 1);
  // Extract input significands.
  ac_int<W - E - 1, false> a_sig = a_int.template slc<W - E - 1>(0);
  ac_int<W - E - 1, false> b_sig = b_int.template slc<W - E - 1>(0);
  ac_int<W - E - 1, false> c_sig = c_int.template slc<W - E - 1>(0);
  ac_int<W - E - 1, false> d_sig = d_int.template slc<W - E - 1>(0);

  enum { NoSubNormals = !ieee_compliance };

  // If input values are to be considered zeros, set the input significand to zero.

  bool a_is_zero = ccs_dw_is_zero<NoSubNormals>(a);
  if (a_is_zero) {
    a_sig = 0;
  }

  bool b_is_zero = ccs_dw_is_zero<NoSubNormals>(b);
  if (b_is_zero) {
    b_sig = 0;
  }

  bool c_is_zero = ccs_dw_is_zero<NoSubNormals>(c);
  if (c_is_zero) {
    c_sig = 0;
  }

  bool d_is_zero = ccs_dw_is_zero<NoSubNormals>(d);
  if (d_is_zero) {
    d_sig = 0;
  }

  // If input values are to be considered subnormals, set the input exponent to 1.

  bool a_is_subn = !a_is_zero && a_exp == 0;
  if (a_is_subn) {
    a_exp = 1;
  }

  bool b_is_subn = !b_is_zero && b_exp == 0;
  if (b_is_subn) {
    b_exp = 1;
  }

  bool c_is_subn = !c_is_zero && c_exp == 0;
  if (c_is_subn) {
    c_exp = 1;
  }

  bool d_is_subn = !d_is_zero && d_exp == 0;
  if (d_is_subn) {
    d_exp = 1;
  }

  // z_int contains bitvector for final floating point output.
  ac_int<W, false> z_int = 0;

  if (a_is_nan || b_is_nan || c_is_nan || d_is_nan) {
    z_int = nan_or_inf; // If any input is NaN, output is NaN as well.
  }
  // If one or more input pairs are infinite and have opposite signs, the output will be set to NaN.
  // If not, the output will be set to +/-inf, depending on what the signs of the infinities are.
  else if (a_is_inf) {
    if ((b_is_inf && a_sign != b_sign)||(c_is_inf && a_sign != c_sign)||(d_is_inf && a_sign != d_sign)) {
      z_int = nan_or_inf;
    } else {
      z_int = a_sign ? ninf : pinf;
    }
  } else if (b_is_inf) {
    if ((a_is_inf && b_sign != a_sign)||(c_is_inf && b_sign != c_sign)||(d_is_inf && b_sign != d_sign)) {
      z_int = nan_or_inf;
    } else {
      z_int = b_sign ? ninf : pinf;
    }
  } else if (c_is_inf) {
    if ((a_is_inf && c_sign != a_sign)||(b_is_inf && c_sign != b_sign)||(d_is_inf && c_sign != d_sign)) {
      z_int = nan_or_inf;
    } else {
      z_int = c_sign ? ninf : pinf;
    }
  } else if (d_is_inf) {
    if ((a_is_inf && d_sign != a_sign)||(b_is_inf && d_sign != b_sign)||(c_is_inf && d_sign != c_sign)) {
      z_int = nan_or_inf;
    } else {
      z_int = d_sign ? ninf : pinf;
    }
  } else if (a_is_zero && b_is_zero && c_is_zero && d_is_zero) {
    // If all the inputs are zeros, the output is set to zero as well.
    z_int = 0;
    z_int[W - 1] = ieee_compliance && a_sign && b_sign && c_sign && d_sign;
  } else {
    // The *_sig_plus_MSB variables have an MSB added to the significand. The MSB is set to zero if the
    // input associated is zero or subnormal, and it's set to one otherwise.
    ac_int<W - E, false> a_sig_plus_MSB = a_sig;
    a_sig_plus_MSB[W - E - 1] = !(a_is_subn || a_is_zero);
    ac_int<W - E, false> b_sig_plus_MSB = b_sig;
    b_sig_plus_MSB[W - E - 1] = !(b_is_subn || b_is_zero);
    ac_int<W - E, false> c_sig_plus_MSB = c_sig;
    c_sig_plus_MSB[W - E - 1] = !(c_is_subn || c_is_zero);
    ac_int<W - E, false> d_sig_plus_MSB = d_sig;
    d_sig_plus_MSB[W - E - 1] = !(d_is_subn || d_is_zero);

    if (a_exp == b_exp && a_sig_plus_MSB == b_sig_plus_MSB && a_sign != b_sign) {
      a_exp = 0;
      a_sig_plus_MSB = 0;
      b_exp = 0;
      b_sig_plus_MSB = 0;
    }
    if (a_exp == c_exp && a_sig_plus_MSB == c_sig_plus_MSB && a_sign != c_sign) {
      a_exp = 0;
      a_sig_plus_MSB = 0;
      c_exp = 0;
      c_sig_plus_MSB = 0;
    }
    if (a_exp == d_exp && a_sig_plus_MSB == d_sig_plus_MSB && a_sign != d_sign) {
      a_exp = 0;
      a_sig_plus_MSB = 0;
      d_exp = 0;
      d_sig_plus_MSB = 0;
    }
    if (b_exp == c_exp && b_sig_plus_MSB == c_sig_plus_MSB && b_sign != c_sign) {
      b_exp = 0;
      b_sig_plus_MSB = 0;
      c_exp = 0;
      c_sig_plus_MSB = 0;
    }
    if (b_exp == d_exp && b_sig_plus_MSB == d_sig_plus_MSB && b_sign != d_sign) {
      b_exp = 0;
      b_sig_plus_MSB = 0;
      d_exp = 0;
      d_sig_plus_MSB = 0;
    }
    if (c_exp == d_exp && c_sig_plus_MSB == d_sig_plus_MSB && c_sign != d_sign) {
      c_exp = 0;
      c_sig_plus_MSB = 0;
      d_exp = 0;
      d_sig_plus_MSB = 0;
    }

    // Find the maximum exponent value. Subtract the input exponent values with this maximum value,
    // which will indicate how much you have to right-shift each of the input significands by.
    ac_int<E, false> max_eval = AC_MAX(AC_MAX(a_exp, b_exp), AC_MAX(c_exp, d_exp));
    // RSA stands for "Right Shift Amount", in other words, the amount to right-shift by.
    const ac_int<E, false> a_sig_RSA = max_eval - a_exp;
    const ac_int<E, false> b_sig_RSA = max_eval - b_exp;
    const ac_int<E, false> c_sig_RSA = max_eval - c_exp;
    const ac_int<E, false> d_sig_RSA = max_eval - d_exp;

    // Pad *_sig_plus_MSB variables with sig_bits + 4 LSBs, all of which are set to zero, and given the
    // *_lsb_pad appendix. These variables are the right-shifted according to the values above, in order
    // to align all the significands correctly for addition later.

    // *_RS_flag variables indicate the loss of 1-bits while right-shifting, which later helps in
    // figuring out whether the output should be rounded or not.

    ac_int<2*W - 2*E + 3, false> a_sig_lsb_pad = 0;
    a_sig_lsb_pad.set_slc(W - E + 3, a_sig_plus_MSB);
    const int a_sig_slc_LSB = AC_MIN(AC_MAX(a_sig_RSA.to_int() - W + E + 1, 0), W - E + 3);
    const bool a_RS_flag = a_sig_lsb_pad.template slc<W - E>(a_sig_slc_LSB) != 0;
    a_sig_lsb_pad >>= a_sig_RSA;

    ac_int<2*W - 2*E + 3, false> b_sig_lsb_pad = 0;
    b_sig_lsb_pad.set_slc(W - E + 3, b_sig_plus_MSB);
    const int b_sig_slc_LSB = AC_MIN(AC_MAX(b_sig_RSA.to_int() - W + E + 1, 0), W - E + 3);
    const bool b_RS_flag = b_sig_lsb_pad.template slc<W - E>(b_sig_slc_LSB) != 0;

    b_sig_lsb_pad >>= b_sig_RSA;

    ac_int<2*W - 2*E + 3, false> c_sig_lsb_pad = 0;
    c_sig_lsb_pad.set_slc(W - E + 3, c_sig_plus_MSB);
    const int c_sig_slc_LSB = AC_MIN(AC_MAX(c_sig_RSA.to_int() - W + E + 1, 0), W - E + 3);
    const bool c_RS_flag = c_sig_lsb_pad.template slc<W - E>(c_sig_slc_LSB) != 0;
    c_sig_lsb_pad >>= c_sig_RSA;

    ac_int<2*W - 2*E + 3, false> d_sig_lsb_pad = 0;
    d_sig_lsb_pad.set_slc(W - E + 3, d_sig_plus_MSB);
    const int d_sig_slc_LSB = AC_MIN(AC_MAX(d_sig_RSA.to_int() - W + E + 1, 0), W - E + 3);
    const bool d_RS_flag = d_sig_lsb_pad.template slc<W - E>(d_sig_slc_LSB) != 0;
    d_sig_lsb_pad >>= d_sig_RSA;

    // Concatenate significand with exp for comparison purposes.

    ac_int<W, false> a_exp_sig_concat = 0;
    a_exp_sig_concat.set_slc(0, a_sig_plus_MSB);
    a_exp_sig_concat.set_slc(W - E, a_exp);

    ac_int<W, false> b_exp_sig_concat = 0;
    b_exp_sig_concat.set_slc(0, b_sig_plus_MSB);
    b_exp_sig_concat.set_slc(W - E, b_exp);

    ac_int<W, false> c_exp_sig_concat = 0;
    c_exp_sig_concat.set_slc(0, c_sig_plus_MSB);
    c_exp_sig_concat.set_slc(W - E, c_exp);

    ac_int<W, false> d_exp_sig_concat = 0;
    d_exp_sig_concat.set_slc(0, d_sig_plus_MSB);
    d_exp_sig_concat.set_slc(W - E, d_exp);

    bool flag1 = false, flag2 = false, flag3 = false, flag4 = false, flag5 = false;

    const bool flag6 = a_sig_lsb_pad.template slc<2*W - 2*E + 2>(1) == 0;
    const bool flag7 = b_sig_lsb_pad.template slc<2*W - 2*E + 2>(1) == 0;
    const bool flag8 = c_sig_lsb_pad.template slc<2*W - 2*E + 2>(1) == 0;
    const bool flag9 = d_sig_lsb_pad.template slc<2*W - 2*E + 2>(1) == 0;

    if (a_RS_flag && flag6 && b_RS_flag && flag7 && c_RS_flag && flag8) {
      flag5 = true;
      if (a_exp_sig_concat > b_exp_sig_concat && a_exp_sig_concat > c_exp_sig_concat) {
        flag2 = flag3 = flag4 = true;
      } else if (c_exp_sig_concat > a_exp_sig_concat && c_exp_sig_concat > b_exp_sig_concat) {
        flag1 = flag2 = flag4 = true;
      } else if (b_exp_sig_concat > a_exp_sig_concat && b_exp_sig_concat > c_exp_sig_concat) {
        flag1 = flag3 = flag4 = true;
      }
    }
    if (a_RS_flag && flag6 && b_RS_flag && flag7 && d_RS_flag && flag9) {
      flag5 = true;
      if (a_exp_sig_concat > b_exp_sig_concat && a_exp_sig_concat > d_exp_sig_concat) {
        flag2 = flag3 = flag4 = true;
      } else if (b_exp_sig_concat > a_exp_sig_concat && b_exp_sig_concat > d_exp_sig_concat) {
        flag1 = flag3 = flag4 = true;
      } else if (d_exp_sig_concat > a_exp_sig_concat && d_exp_sig_concat > b_exp_sig_concat) {
        flag1 = flag2 = flag3 = true;
      }
    }
    if (a_RS_flag && flag6 && c_RS_flag && flag8 && d_RS_flag && flag9) {
      flag5 = true;
      if (a_exp_sig_concat > c_exp_sig_concat && a_exp_sig_concat > d_exp_sig_concat) {
        flag2 = flag3 = flag4 = true;
      } else if (c_exp_sig_concat > a_exp_sig_concat && c_exp_sig_concat > d_exp_sig_concat) {
        flag1 = flag2 = flag4 = true;
      } else if (d_exp_sig_concat > a_exp_sig_concat && d_exp_sig_concat > c_exp_sig_concat) {
        flag1 = flag2 = flag3 = true;
      }
    }
    if (b_RS_flag && flag7 && c_RS_flag && flag8 && d_RS_flag && flag9) {
      flag5 = true;
      if (b_exp_sig_concat > c_exp_sig_concat && b_exp_sig_concat > d_exp_sig_concat) {
        flag1 = flag3 = flag4 = true;
      } else if (c_exp_sig_concat > b_exp_sig_concat && c_exp_sig_concat > d_exp_sig_concat) {
        flag1 = flag2 = flag4 = true;
      } else if (d_exp_sig_concat > b_exp_sig_concat && d_exp_sig_concat > c_exp_sig_concat) {
        flag1 = flag2 = flag3 = true;
      }
    }

    if (!flag5) {
      flag1 = a_RS_flag && flag6 && (
                (b_RS_flag && flag7 && a_exp_sig_concat < b_exp_sig_concat) ||
                (c_RS_flag && flag8 && a_exp_sig_concat < c_exp_sig_concat) ||
                (d_RS_flag && flag9 && a_exp_sig_concat < d_exp_sig_concat)
              );
      flag2 = b_RS_flag && flag7 && (
                (a_RS_flag && flag6 && b_exp_sig_concat < a_exp_sig_concat) ||
                (c_RS_flag && flag8 && b_exp_sig_concat < c_exp_sig_concat) ||
                (d_RS_flag && flag9 && b_exp_sig_concat < d_exp_sig_concat)
              );
      flag3 = c_RS_flag && flag8 && (
                (a_RS_flag && flag6 && c_exp_sig_concat < a_exp_sig_concat) ||
                (b_RS_flag && flag7 && c_exp_sig_concat < b_exp_sig_concat) ||
                (d_RS_flag && flag9 && c_exp_sig_concat < d_exp_sig_concat)
              );
      flag4 = d_RS_flag && flag9 && (
                (a_RS_flag && flag6 && d_exp_sig_concat < a_exp_sig_concat) ||
                (b_RS_flag && flag7 && d_exp_sig_concat < b_exp_sig_concat) ||
                (c_RS_flag && flag8 && d_exp_sig_concat < c_exp_sig_concat)
              );
    }

    // Add three more MSBs to *_lsb_pad variables.
    ac_int<2*W - 2*E + 6, false> a_sig_bs_pad = a_sig_lsb_pad;
    ac_int<2*W - 2*E + 6, false> b_sig_bs_pad = b_sig_lsb_pad;
    ac_int<2*W - 2*E + 6, false> c_sig_bs_pad = c_sig_lsb_pad;
    ac_int<2*W - 2*E + 6, false> d_sig_bs_pad = d_sig_lsb_pad;

    a_sig_bs_pad[0] = !flag1 && a_RS_flag;
    b_sig_bs_pad[0] = !flag2 && b_RS_flag;
    c_sig_bs_pad[0] = !flag3 && c_RS_flag;
    d_sig_bs_pad[0] = !flag4 && d_RS_flag;

    a_sig_bs_pad = a_sign ? ac_int<2*W - 2*E + 6, false>(~a_sig_bs_pad + 1) : a_sig_bs_pad;
    b_sig_bs_pad = b_sign ? ac_int<2*W - 2*E + 6, false>(~b_sig_bs_pad + 1) : b_sig_bs_pad;
    c_sig_bs_pad = c_sign ? ac_int<2*W - 2*E + 6, false>(~c_sig_bs_pad + 1) : c_sig_bs_pad;
    d_sig_bs_pad = d_sign ? ac_int<2*W - 2*E + 6, false>(~d_sig_bs_pad + 1) : d_sig_bs_pad;

    // Add all the appropriately aligned and (if input is negative) negated significands together.
    ac_int<2*W - 2*E + 6, false> sig_add_res = a_sig_bs_pad + b_sig_bs_pad + c_sig_bs_pad + d_sig_bs_pad;

    bool sig_ares_sign = sig_add_res[2*W - 2*E + 5];

    ac_int<2*W - 2*E + 5, false> abs_sig_ares = 0;

    if (sig_ares_sign) {
      // If the addition result is negative, the 2s complement is stored in abs_sig_ares.
      abs_sig_ares = ~sig_add_res + 1;
    } else {
      abs_sig_ares = sig_add_res;
    }

    ac_int<E + 2, false> z_exp = max_eval;

    if ((a_RS_flag || b_RS_flag || c_RS_flag || d_RS_flag) && abs_sig_ares.template slc<W - E + 5>(W - E) == 0) {
      if (a_RS_flag) {
        z_int = a.data_ac_int();
      }
      if (b_RS_flag) {
        z_int = b.data_ac_int();
      }
      if (c_RS_flag) {
        z_int = c.data_ac_int();
      }
      if (d_RS_flag) {
        z_int = d.data_ac_int();
      }
    } else {
      // Left-shift abs_sig_ares such that the leading one coincides with the 3rd MSB.
      int ares_LSA = abs_sig_ares.leading_sign().to_int() - 2;

      bool ares_RS_flag = false;

      if (ares_LSA > 0) {
        ares_LSA = AC_MIN(ares_LSA, AC_MAX(z_exp.to_int() - 1, 0));
      } else if (ares_LSA < 0) {
        ares_RS_flag = (ares_LSA == -2 && abs_sig_ares[1]) || abs_sig_ares[0];
      }

      abs_sig_ares <<= ares_LSA;
      
      // Adjust z_exp according to how much we left-shifted by.
      z_exp -= ares_LSA;

      bool rnd_param3 = abs_sig_ares[W - E + 3];
      bool rnd_param4 = abs_sig_ares[W - E + 2];
      bool rnd_param5 = abs_sig_ares.template slc<W - E + 2>(0) != 0;
      rnd_param5 = rnd_param5 || a_RS_flag || b_RS_flag || c_RS_flag || d_RS_flag || ares_RS_flag;

      // rnd_out takes into account multiple factors to decide whether the output should be rounded or not.
      bool rnd_out = (rnd == 0 && rnd_param4 && (rnd_param3 || rnd_param5));

      ac_int<W - E + 4, false> rnd_add = 0;
      rnd_add[W - E + 3] = 1;

      if (rnd_out) {
        abs_sig_ares = abs_sig_ares + rnd_add;
      }

      z_exp += int(abs_sig_ares[2*W - 2*E + 3]);
      abs_sig_ares >>= int(abs_sig_ares[2*W - 2*E + 3]);

      if (abs_sig_ares == 0) {
        z_int = 0;
      } else if (abs_sig_ares.template slc<3>(2*W - 2*E + 2) == 0) {
        z_int = 0;
        z_int[W - 1] = sig_ares_sign;
        if (ieee_compliance) {
          z_int.set_slc(0, abs_sig_ares.template slc<W - E - 1>(W - E + 3));
        }
      } else {
        const ac_int<E, false> exp_max = -1; // Max, i.e. all 1s exponent.
        if (z_exp >= exp_max) {
          if (rnd == 0) {
            // Set output to +inf if z_exp >= exp_max and rnd == 0.
            z_exp = exp_max;
            abs_sig_ares = 0;
          } else {
            // Set output to the max normalized value if z_exp >= exp_max and rnd == 1.
            z_exp = exp_max - 1;
            abs_sig_ares = -1; // abs_sig_ares set to all 1s.
          }
        } else if (z_exp == 0) {
          z_exp = 1; // If z_exp is equal to zero, set it to 1.
        }
        
        z_int.set_slc(0, abs_sig_ares.template slc<W - E - 1>(W - E + 3));
        z_int.set_slc(W - E - 1, z_exp.template slc<E>(0));
        z_int[W - 1] = sig_ares_sign;
      }
    }
  }

  z.set_data(z_int);
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_dw_fp_sum4
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_dw_fp_sum4(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<sig_width+exp_width+1,true> &c,
  const ac_int<sig_width+exp_width+1,true> &d,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width + exp_width + 1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, c_fl, d_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  c_fl.set_data(c);
  d_fl.set_data(d);
  // Call simulation model.
  ccs_dw_fp_sum4_sim_model<ieee_compliance>(a_fl, b_fl, c_fl, d_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Mapped to the operator using ac_int
#pragma map_to_operator ccs_fp_sum4
template<int sig_width, int exp_width, int ieee_compliance>
void ccs_fp_sum4(
  const ac_int<sig_width+exp_width+1,true> &a,
  const ac_int<sig_width+exp_width+1,true> &b,
  const ac_int<sig_width+exp_width+1,true> &c,
  const ac_int<sig_width+exp_width+1,true> &d,
  const ac_int<3,false> &rnd,
  ac_int<sig_width+exp_width+1,true> &z
)
{
  enum { W = sig_width + exp_width + 1, E = exp_width };
  ac_std_float<W,E> a_fl, b_fl, c_fl, d_fl, z_fl;
  a_fl.set_data(a);
  b_fl.set_data(b);
  c_fl.set_data(c);
  d_fl.set_data(d);
  // Call simulation model.
  ccs_dw_fp_sum4_sim_model<ieee_compliance>(a_fl, b_fl, c_fl, d_fl, rnd, z_fl);
  z = z_fl.data_ac_int();
}

// Top-level API - handles conversion to ac_int for map_to_operator
template<ac_q_mode QR, int ieee_compliance, int W, int E>
ac_std_float<W,E> fp_sum4(const ac_std_float<W,E> &a, const ac_std_float<W,E> &b, const ac_std_float<W,E> &c, ac_std_float<W,E> &d)
{
  enum {
    sig_width = W-E-1,
    exp_width = E
  };
  #ifndef __SYNTHESIS__
  static bool print_once = true;
  if (print_once && sig_width >= 15) {
    // The DW_fp_sum4 VHDL simulation model uses the conv_integer function, and on platforms with 32-bit
    // integers, any significand width equal to or exceeding 15 will throw an error during VHDL
    // simulation because the bitvector passed to conv_integer will be too large.
    std::cout << "WARNING: Consider reducing the significand width to a value below 15 to ensure simulation with the VHDL DW_fp_sum3 module, if that is being used." << std::endl;
    print_once = false;
  }
  #endif
  #if __cplusplus > 199711L
  static_assert(std::test_qr_structs<QR>::match, "Rounding mode not supported");
  #else
  std::test_qr_structs<QR>::Rounding_mode_not_supported;
  #endif
  ac_int<3,false> rnd = QR == AC_TRN_ZERO;
  ac_int<W,true> z;
  #ifdef USING_CCS_LEGACY_DW_LIB
  ccs_dw_fp_sum4<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), c.data_ac_int(), d.data_ac_int(), rnd, z);
  #else
  ccs_fp_sum4<sig_width, exp_width, ieee_compliance>(a.data_ac_int(), b.data_ac_int(), c.data_ac_int(), d.data_ac_int(), rnd, z);
  #endif
  ac_std_float<W,E> z_fl;
  z_fl.set_data(z);
  return z_fl;
}
