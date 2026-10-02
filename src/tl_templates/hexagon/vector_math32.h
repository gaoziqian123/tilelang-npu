#pragma once
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

// Standalone register leaves; contract: docs/hexagon/vector_math32_v1.md.
// Deliberately independent of exp.h/vector_leaf.h and their scalar repairs.
namespace tl { namespace hvx_math32_v1 {
constexpr bool has_single_rounding_fma_f32 = false;
// No sf*sf+sf HVX instruction is exposed by the v79 SDK. Never substitute
// qfloat MAC or two IEEE operations for an explicitly fused operation.
template <bool Available = has_single_rounding_fma_f32>
inline HVX_Vector fma_f32(HVX_Vector, HVX_Vector, HVX_Vector) {
  static_assert(Available && has_single_rounding_fma_f32,
                "HVX v79 single-rounding FP32 FMA mode is unsupported");
  return Q6_V_vzero();
}

// Approximate exp, all binary32 bit patterns accepted. Not strict/correctly
// rounded. All exceptional handling and gradual output underflow are integer
// HVX operations; no lane extraction, libm, scalar repair or data-dependent loop.
inline __attribute__((always_inline)) HVX_Vector exp_f32(HVX_Vector input) {
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector one = Q6_V_vsplat_R(1);
  const HVX_Vector inf = Q6_V_vsplat_R(0x7f800000);
  HVX_Vector mag = Q6_V_vand_VV(input, Q6_V_vsplat_R(0x7fffffff));
  HVX_VectorPred nan = Q6_Q_vcmp_gt_VuwVuw(mag, inf);
  HVX_VectorPred negative = Q6_Q_vcmp_gt_VwVw(zero, input);
  // Sanitize before floating arithmetic (including signaling NaNs).
  HVX_VectorPred large = Q6_Q_vcmp_gt_VuwVuw(mag, Q6_V_vsplat_R(0x42d00000));
  HVX_Vector x = Q6_V_vmux_QVV(large, zero, input);
  x = Q6_Vsf_vmax_VsfVsf(Q6_V_vsplat_R(0xc2d00000),
      Q6_Vsf_vmin_VsfVsf(x, Q6_V_vsplat_R(0x42b17217)));
  // Round to nearest integer using the binary32 magic bias. |k| <= 150.
  const HVX_Vector bias = Q6_V_vsplat_R(0x4b400000);
  HVX_Vector t = Q6_Vsf_vadd_VsfVsf(
      Q6_Vsf_vmpy_VsfVsf(x, Q6_V_vsplat_R(0x3fb8aa3b)), bias);
  HVX_Vector k = Q6_Vw_vsub_VwVw(t, bias);
  HVX_Vector f = Q6_Vsf_vsub_VsfVsf(t, bias);
  // Split ln(2): f*hi is exact in this range. No FMA assumption.
  HVX_Vector r = Q6_Vsf_vsub_VsfVsf(x,
      Q6_Vsf_vmpy_VsfVsf(f, Q6_V_vsplat_R(0x3f317200)));
  r = Q6_Vsf_vsub_VsfVsf(r,
      Q6_Vsf_vmpy_VsfVsf(f, Q6_V_vsplat_R(0x35bfbe8e)));
  HVX_Vector y = Q6_V_vsplat_R(0x37d00d01); // 1/8!
  const int coeff[8] = {0x39500d01, 0x3ab60b61, 0x3c088889, 0x3d2aaaab,
                        0x3e2aaaab, 0x3f000000, 0x3f800000, 0x3f800000};
#pragma unroll
  for (int i = 0; i < 8; ++i)
    y = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(y, r), Q6_V_vsplat_R(coeff[i]));

  // Scale via exponent bits, avoiding target floating FTZ on subnormals.
  HVX_Vector e = Q6_Vw_vadd_VwVw(Q6_Vuw_vlsr_VuwR(y, 23), k);
  HVX_Vector normal = Q6_Vw_vadd_VwVw(y, Q6_Vw_vasl_VwR(k, 23));
  HVX_Vector sig = Q6_V_vor_VV(Q6_V_vand_VV(y, Q6_V_vsplat_R(0x7fffff)),
                               Q6_V_vsplat_R(0x800000));
  HVX_Vector shift = Q6_Vw_vmax_VwVw(one, Q6_Vw_vmin_VwVw(
      Q6_V_vsplat_R(24), Q6_Vw_vsub_VwVw(one, e)));
  HVX_Vector sub = Q6_Vw_vasr_VwVw(sig, shift); // positive significand
  HVX_Vector half = Q6_Vw_vasl_VwVw(one, Q6_Vw_vsub_VwVw(shift, one));
  HVX_Vector mask = Q6_Vw_vsub_VwVw(Q6_Vw_vasl_VwVw(one, shift), one);
  HVX_Vector rem = Q6_V_vand_VV(sig, mask);
  HVX_Vector inc = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(rem, half), one, zero);
  inc = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(rem, half), Q6_V_vand_VV(sub, one), inc);
  sub = Q6_Vw_vadd_VwVw(sub, inc); // ties to even; may round up to FLT_MIN
  sub = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(-23), e), zero, sub);
  y = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(e, zero), normal, sub);
  // Finite overflow threshold is the first binary32 x >= log(FLT_MAX).
  HVX_Vector overflow = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(
      mag, Q6_V_vsplat_R(0x42b17217)), inf, y);
  HVX_Vector underflow = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(
      mag, Q6_V_vsplat_R(0x42cfffff)), zero, y);
  y = Q6_V_vmux_QVV(negative, underflow, overflow);
  // Preserve sign/payload, quiet sNaNs; no floating exception promise.
  return Q6_V_vmux_QVV(nan, Q6_V_vor_VV(input, Q6_V_vsplat_R(0x00400000)), y);
}
}} // namespace tl::hvx_math32_v1
