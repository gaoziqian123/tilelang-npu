#pragma once

#include "common.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

namespace tl {
// Typed carriers prevent accidentally applying integer arithmetic to FP bits.
template <typename T> struct hvx_vec { HVX_Vector raw; };
static_assert(sizeof(HVX_Vector) == 128, "Hexagon templates require 128-byte HVX");
template <typename T> TL_DEVICE hvx_vec<T> hvx_load(const T *p) {
  hex_require(is_aligned<128>(p));
  return {*reinterpret_cast<const HVX_Vector *>(p)};
}
template <typename T> TL_DEVICE void hvx_store(T *p, hvx_vec<T> v) {
  hex_require(is_aligned<128>(p));
  *reinterpret_cast<HVX_Vector *>(p) = v.raw;
}
TL_DEVICE hvx_vec<float> hvx_add(hvx_vec<float> a, hvx_vec<float> b) {
  return {Q6_Vsf_vadd_VsfVsf(a.raw, b.raw)};
}
TL_DEVICE hvx_vec<float> hvx_mul(hvx_vec<float> a, hvx_vec<float> b) {
  return {Q6_Vsf_vmpy_VsfVsf(a.raw, b.raw)};
}
TL_DEVICE hvx_vec<float> hvx_sub(hvx_vec<float> a, hvx_vec<float> b) {
  return {Q6_Vsf_vsub_VsfVsf(a.raw, b.raw)};
}
TL_DEVICE hvx_vec<half_t> hvx_sub(hvx_vec<half_t> a, hvx_vec<half_t> b) {
  return {Q6_Vhf_vsub_VhfVhf(a.raw, b.raw)};
}
// hrt_vsf_div_approx recipe: two Newton steps. Finite positive normal
// denominator only, no zero/Inf/NaN/subnormal guarantee. Keep the approximate
// name so codegen cannot silently use it for unrestricted TIR division.
TL_DEVICE hvx_vec<float> hvx_div_approx(hvx_vec<float> a, hvx_vec<float> b) {
  HVX_Vector r = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311D3), b.raw);
  HVX_Vector two = Q6_V_vsplat_R(0x40000000);
  for (int i = 0; i < 2; ++i)
    r = Q6_Vsf_vmpy_VsfVsf(r, Q6_Vsf_vsub_VsfVsf(two, Q6_Vsf_vmpy_VsfVsf(b.raw, r)));
  return {Q6_Vsf_vmpy_VsfVsf(a.raw, r)};
}
// General division uses scalar IEEE arithmetic on LOCAL stack spills, never
// scalar VTCM accesses. TODO: a proven full-domain vector division recipe.
template <typename T> TL_DEVICE hvx_vec<T> hvx_div(hvx_vec<T> a, hvx_vec<T> b) {
  alignas(128) T x[128 / sizeof(T)], y[128 / sizeof(T)];
  hvx_store(x, a); hvx_store(y, b);
  for (unsigned i = 0; i < 128 / sizeof(T); ++i) x[i] = x[i] / y[i];
  return hvx_load(x);
}
// Lane access is to a register spill on the local stack, not to VTCM.
template <typename T> TL_DEVICE T hvx_lane(hvx_vec<T> v, int lane) {
  hex_require(lane >= 0 && lane < int(128 / sizeof(T)));
  alignas(128) T tmp[128 / sizeof(T)];
  hvx_store(tmp, v);
  return tmp[lane];
}
template <typename T> TL_DEVICE hvx_vec<T> hvx_set_lane(hvx_vec<T> v, int lane, T value) {
  hex_require(lane >= 0 && lane < int(128 / sizeof(T)));
  alignas(128) T tmp[128 / sizeof(T)];
  hvx_store(tmp, v); tmp[lane] = value;
  return hvx_load(tmp);
}
// Two separately aligned chunks; no assumption of a nonexistent 256B HVX
// register. Codegen can also emit two hvx_vec<float> operations directly.
struct hvx_float64 { hvx_vec<float> lo, hi; };
TL_DEVICE hvx_float64 hvx_load64(const float *p) { return {hvx_load(p), hvx_load(p + 32)}; }
TL_DEVICE void hvx_store64(float *p, hvx_float64 v) { hvx_store(p, v.lo); hvx_store(p + 32, v.hi); }
TL_DEVICE hvx_float64 hvx_add(hvx_float64 a, hvx_float64 b) { return {hvx_add(a.lo, b.lo), hvx_add(a.hi, b.hi)}; }
TL_DEVICE hvx_float64 hvx_mul(hvx_float64 a, hvx_float64 b) { return {hvx_mul(a.lo, b.lo), hvx_mul(a.hi, b.hi)}; }
TL_DEVICE hvx_float64 hvx_sub(hvx_float64 a, hvx_float64 b) { return {hvx_sub(a.lo, b.lo), hvx_sub(a.hi, b.hi)}; }
TL_DEVICE hvx_float64 hvx_div(hvx_float64 a, hvx_float64 b) { return {hvx_div(a.lo, b.lo), hvx_div(a.hi, b.hi)}; }
TL_DEVICE hvx_vec<float> hvx_max(hvx_vec<float> a, hvx_vec<float> b) {
  return {Q6_Vsf_vmax_VsfVsf(a.raw, b.raw)};
}
TL_DEVICE hvx_vec<half_t> hvx_add(hvx_vec<half_t> a, hvx_vec<half_t> b) {
  return {Q6_Vhf_vadd_VhfVhf(a.raw, b.raw)};
}
TL_DEVICE hvx_vec<half_t> hvx_mul(hvx_vec<half_t> a, hvx_vec<half_t> b) {
  return {Q6_Vhf_vmpy_VhfVhf(a.raw, b.raw)};
}
TL_DEVICE hvx_vec<half_t> hvx_max(hvx_vec<half_t> a, hvx_vec<half_t> b) {
  return {Q6_Vhf_vfmax_VhfVhf(a.raw, b.raw)};
}
// hexagon_rt.h hrt_exp_fp16 recipe. Bounded approximate exp (base-2
// argument clamped to [-15,15]), NOT an IEEE exp or a recurrent-state exp.
TL_DEVICE hvx_vec<half_t> hvx_exp(hvx_vec<half_t> v) {
  HVX_Vector t = Q6_Vhf_vmpy_VhfVhf(v.raw, Q6_Vh_vsplat_R(0x3DC5));
  t = Q6_Vhf_vfmax_VhfVhf(t, Q6_Vh_vsplat_R(0xCB80));
  t = Q6_Vhf_vfmin_VhfVhf(t, Q6_Vh_vsplat_R(0x4B80));
  HVX_Vector ni = Q6_Vh_vcvt_Vhf(t);
  HVX_Vector nf = Q6_Vhf_vcvt_Vh(ni);
  HVX_VectorPred q = Q6_Q_vcmp_gt_VhVh(Q6_Vhf_vsub_VhfVhf(nf, t), Q6_V_vzero());
  ni = Q6_V_vmux_QVV(q, Q6_Vh_vsub_VhVh(ni, Q6_Vh_vsplat_R(1)), ni);
  HVX_Vector f = Q6_Vhf_vsub_VhfVhf(t, Q6_Vhf_vcvt_Vh(ni));
  HVX_Vector p = Q6_Vhf_vmpyacc_VhfVhfVhf(Q6_Vh_vsplat_R(0x2B1B), Q6_Vh_vsplat_R(0x20ED), f);
  p = Q6_Vhf_vmpyacc_VhfVhfVhf(Q6_Vh_vsplat_R(0x33B0), p, f);
  p = Q6_Vhf_vmpyacc_VhfVhfVhf(Q6_Vh_vsplat_R(0x398C), p, f);
  p = Q6_Vhf_vmpyacc_VhfVhfVhf(Q6_Vh_vsplat_R(0x3C00), p, f);
  return {Q6_Vhf_vmpy_VhfVhf(p, Q6_Vh_vasl_VhR(Q6_Vh_vadd_VhVh(ni, Q6_Vh_vsplat_R(15)), 10))};
}
// Rotate-fold recipes from hrt_reduce_{sum,max}_f32_32. The spill is local
// DDR/stack, not scalar VTCM traffic.
TL_DEVICE float hvx_reduce_sum(hvx_vec<float> v) {
  for (int rot = 64; rot >= 4; rot >>= 1)
    v.raw = Q6_Vsf_vadd_VsfVsf(v.raw, Q6_V_vror_VR(v.raw, rot));
  alignas(128) float out[32];
  hvx_store(out, v);
  return out[0];
}
TL_DEVICE float hvx_reduce_max(hvx_vec<float> v) {
  for (int rot = 64; rot >= 4; rot >>= 1)
    v.raw = Q6_Vsf_vmax_VsfVsf(v.raw, Q6_V_vror_VR(v.raw, rot));
  alignas(128) float out[32];
  hvx_store(out, v);
  return out[0];
}
} // namespace tl
