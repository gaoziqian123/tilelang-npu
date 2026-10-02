#pragma once
#include "hvx.h"
#include <math.h>
#ifdef TL_HEX_EXP_PATH_DIAG
extern "C" void tl_hex_exp_path(int);
#define TL_EXP_PATH(n) tl_hex_exp_path(n)
#else
#define TL_EXP_PATH(n) ((void)0)
#endif
namespace tl {
// Explicit mixed-precision API: fp16 input/output, fp16 base-2 polynomial.
// On [-10,0], error contract is 1% relative + two fp16 subnormal ULPs.
// Outside that interval use scalar expf rounding to fp16, not clipped math.
// -infinity is handled vectorially as exact zero (causal masks).
__attribute__((noinline,cold)) static HVX_Vector exp16_repair(HVX_Vector x, HVX_Vector y) {
  alignas(128) half_t in[64], out[64];
  __builtin_memcpy(in,&x,128); __builtin_memcpy(out,&y,128);
  for(int i=0;i<64;++i) {
    float a=(float)in[i];
    if(!(a>=-10.f && a<=0.f)) out[i]=(half_t)expf(a);
  }
  __builtin_memcpy(&y,out,128); return y;
}
TL_DEVICE HVX_Vector exp_f16_softmax(HVX_Vector x) {
  HVX_Vector z=Q6_V_vzero();
  HVX_VectorPred ni=Q6_Q_vcmp_eq_VhVh(x,Q6_Vh_vsplat_R(0xfc00));
  HVX_Vector a=Q6_V_vmux_QVV(ni,z,x);
  HVX_Vector y=hvx_exp(hvx_vec<half_t>{a}).raw;
  y=Q6_V_vmux_QVV(ni,z,y);
  // Bitwise test also rejects positive numbers and NaNs.
  HVX_Vector flags=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuhVuh(
      Q6_V_vand_VV(a,Q6_Vh_vsplat_R(0x7fff)),Q6_Vh_vsplat_R(0x4900)),
      Q6_Vh_vsplat_R(-1),z);
  flags=Q6_V_vor_VV(flags,Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(a,z),Q6_Vh_vsplat_R(-1),z));
  for(int shift=64;shift>=2;shift/=2) flags=Q6_V_vor_VV(flags,Q6_V_vror_VR(flags,shift));
  if(Q6_R_vextract_VR(flags,0)) return exp16_repair(x,y);
  return y;
}
// Cold IEEE repair is intentionally out-of-line: its scalar double-libm ABI
// and 256B lane arrays must not extend hot vector register lifetimes.
__attribute__((noinline,cold)) static HVX_Vector exp32_repair(HVX_Vector input, HVX_Vector y) {
  alignas(128) float in[32],out[32];
  *(HVX_Vector*)in=input; *(HVX_Vector*)out=y;
  for(int i=0;i<32;i++) if(!(in[i]>=-87.f && in[i]<=87.f)) {
    unsigned bits; __builtin_memcpy(&bits,in+i,4);
    unsigned result;
    if((bits&0x7fffffffu)>0x7f800000u) result=bits|0x00400000u;
    else if(bits==0xff800000u || in[i]<-104.f) result=0;
    else if(in[i]>88.722839f) result=0x7f800000u;
    else if(in[i]<-87.33654f) {
      double m=exp((double)in[i])*0x1p149;
      unsigned a=(unsigned)m; double frac=m-a;
      result=a+(frac>0.5 || (frac==0.5 && (a&1)));
    } else {
      float f=(float)exp((double)in[i]); __builtin_memcpy(&result,&f,4);
    }
    __builtin_memcpy(out+i,&result,4);
  }
  return *(HVX_Vector*)out;
}
// Leaf vector exp, range reduction/polynomial from official HTP hvx-exp.h.
// Exceptional/tiny/overflow lanes use libm: preserves -inf -> exact zero,
// NaNs, subnormal tails and the scalar API outside the polynomial domain.
#ifndef TL_HEX_INLINE_VECTOR_EXP
#define TL_HEX_INLINE_VECTOR_EXP 0
#endif
#ifndef TL_HEX_EXP_QF_HORNER
#define TL_HEX_EXP_QF_HORNER 0
#endif
#ifndef TL_HEX_EXP_CENTERED4
#define TL_HEX_EXP_CENTERED4 0
#endif
#ifndef TL_HEX_EXP_OUTLINE_GENERAL
#define TL_HEX_EXP_OUTLINE_GENERAL 0
#endif
#if TL_HEX_EXP_OUTLINE_GENERAL
__attribute__((noinline)) static HVX_Vector exp32_general(HVX_Vector input);
#endif
template <bool NearDomain = true>
#if TL_HEX_INLINE_VECTOR_EXP
__attribute__((always_inline)) inline
#else
__attribute__((noinline)) static
#endif
HVX_Vector native_exp32_impl(HVX_Vector input) {
  HVX_Vector z=Q6_V_vzero(), one=Q6_V_vsplat_R(0x3f800000);
  // General near-zero negative domain: avoid exponent reconstruction. This
  // includes exact -inf masks, but rejects NaNs, +inf, and finite tails.
  HVX_VectorPred ni=Q6_Q_vcmp_eq_VwVw(input,Q6_V_vsplat_R(0xff800000));
  HVX_Vector small=Q6_V_vmux_QVV(ni,z,input);
  HVX_Vector bad=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(
      Q6_V_vand_VV(small,Q6_V_vsplat_R(0x7fffffff)),
      Q6_V_vsplat_R(0x3f000000)),Q6_V_vsplat_R(-1),z);
  bad=Q6_V_vor_VV(bad,Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(small,z),Q6_V_vsplat_R(-1),z));
  for(int shift=64;shift>=4;shift/=2)
    bad=Q6_V_vor_VV(bad,Q6_V_vror_VR(bad,shift));
  if(NearDomain && Q6_R_vextract_VR(bad,0)==0) {
    TL_EXP_PATH(0);
    // Degree eight Taylor on [-0.5,0]; analytic absolute remainder < 5.4e-9.
    HVX_Vector y=Q6_V_vsplat_R(0x37d00d01); // 1/40320
    const int c[8]={0x39500d01,0x3ab60b61,0x3c088889,0x3d2aaaab,
                    0x3e2aaaab,0x3f000000,0x3f800000,0x3f800000};
#if TL_HEX_EXP_CENTERED4
    // exp(x) = exp(-1/4) * exp(t), t=x+1/4 in [-1/4,1/4].
    // Degree-four Taylor: relative truncation <= exp(1/2)/(120*4^5)
    // < 1.342e-5. Including binary32 coefficient rounding and <=12 rounded
    // operations contributes < 2e-6 relative (|P|>=0.778), below 2e-5.
    // Coefficients are nearest binary32 exp(-1/4)/j!, j=0..4.
    // Runtime guard and full-domain fallback remain identical.
    HVX_Vector t=Q6_Vsf_vadd_VsfVsf(small,Q6_V_vsplat_R(0x3e800000));
    HVX_Vector t2=Q6_Vsf_vmpy_VsfVsf(t,t);
    HVX_Vector t4=Q6_Vsf_vmpy_VsfVsf(t2,t2);
    HVX_Vector c0=Q6_V_vsplat_R(0x3f475f7d);
    HVX_Vector p0=Q6_Vsf_vadd_VsfVsf(c0,Q6_Vsf_vmpy_VsfVsf(c0,t));
    HVX_Vector p1=Q6_Vsf_vadd_VsfVsf(Q6_V_vsplat_R(0x3ec75f7d),
        Q6_Vsf_vmpy_VsfVsf(Q6_V_vsplat_R(0x3e04ea53),t));
    y=Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vadd_VsfVsf(p0,Q6_Vsf_vmpy_VsfVsf(p1,t2)),
        Q6_Vsf_vmpy_VsfVsf(Q6_V_vsplat_R(0x3d04ea53),t4));
#elif TL_HEX_EXP_QF_HORNER == 2
    // Estrin evaluation of the identical degree-eight polynomial. Four
    // independent linear pairs shorten the dependent FP32 multiply/add chain.
    // The runtime [-0.5,0] guard and exceptional handling are unchanged.
    HVX_Vector x2=Q6_Vsf_vmpy_VsfVsf(small,small);
    HVX_Vector x4=Q6_Vsf_vmpy_VsfVsf(x2,x2);
    HVX_Vector x8=Q6_Vsf_vmpy_VsfVsf(x4,x4);
    HVX_Vector p0=Q6_Vsf_vadd_VsfVsf(one,small);
    HVX_Vector p1=Q6_Vsf_vadd_VsfVsf(Q6_V_vsplat_R(c[5]),Q6_Vsf_vmpy_VsfVsf(Q6_V_vsplat_R(c[4]),small));
    HVX_Vector p2=Q6_Vsf_vadd_VsfVsf(Q6_V_vsplat_R(c[3]),Q6_Vsf_vmpy_VsfVsf(Q6_V_vsplat_R(c[2]),small));
    HVX_Vector p3=Q6_Vsf_vadd_VsfVsf(Q6_V_vsplat_R(c[1]),Q6_Vsf_vmpy_VsfVsf(Q6_V_vsplat_R(c[0]),small));
    HVX_Vector a=Q6_Vsf_vadd_VsfVsf(p0,Q6_Vsf_vmpy_VsfVsf(p1,x2));
    HVX_Vector b=Q6_Vsf_vadd_VsfVsf(p2,Q6_Vsf_vmpy_VsfVsf(p3,x2));
    y=Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vadd_VsfVsf(a,Q6_Vsf_vmpy_VsfVsf(b,x4)),Q6_Vsf_vmpy_VsfVsf(y,x8));
#elif TL_HEX_EXP_QF_HORNER == 1
    // Same degree-eight polynomial and runtime domain guard. Keep the
    // intermediate in normalized qf32 rather than rounding to sf32 twice
    // per Horner step. This is an opt-in precision/performance experiment;
    // the public domain and exceptional repair contract are unchanged.
    HVX_Vector xq=Q6_Vqf32_vadd_VsfVsf(small,z);
    y=Q6_Vqf32_vadd_VsfVsf(y,z);
    #pragma unroll
    for(int i=0;i<8;i++) {
      y=Q6_Vqf32_vmpy_Vqf32Vqf32(y,xq);
      y=Q6_Vqf32_vadd_Vqf32Vsf(y,Q6_V_vsplat_R(c[i]));
      y=Q6_Vqf32_vadd_Vqf32Vsf(y,z);
    }
    y=Q6_Vsf_equals_Vqf32(y);
#else
    #pragma unroll
    for(int i=0;i<8;i++)
      y=Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(y,small),Q6_V_vsplat_R(c[i]));
#endif
    return Q6_V_vmux_QVV(ni,z,y);
  }
  TL_EXP_PATH(1);
#if TL_HEX_EXP_OUTLINE_GENERAL
  return exp32_general(input);
}
// Keep general range reduction/constants outside the near path's register
// allocation unit. Same operations, coefficients and IEEE repair, no math
// reassociation and no assumptions about the caller's input distribution.
__attribute__((noinline)) static HVX_Vector exp32_general(HVX_Vector input) {
  HVX_Vector z=Q6_V_vzero(), one=Q6_V_vsplat_R(0x3f800000);
#endif
  HVX_Vector x=Q6_Vsf_vmax_VsfVsf(Q6_V_vsplat_R(0xc2ae0000),
      Q6_Vsf_vmin_VsfVsf(input,Q6_V_vsplat_R(0x42ae0000)));
  HVX_Vector f=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x,Q6_V_vsplat_R(0x3fb8aa3b)));
  HVX_Vector exponent=Q6_Vw_vsub_VwVw(Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(f,23),Q6_V_vsplat_R(255)),Q6_V_vsplat_R(127));
  HVX_Vector mask=Q6_Vw_vasr_VwVw(Q6_V_vsplat_R(0x7fffff),exponent);
  HVX_Vector increment=Q6_Vw_vasr_VwVw(Q6_V_vsplat_R(0x800000),exponent);
  HVX_VectorPred neg=Q6_Q_vcmp_gt_VwVw(z,f);
  HVX_Vector flo=Q6_V_vand_VV(Q6_V_vmux_QVV(neg,Q6_Vw_vadd_VwVw(f,increment),f),Q6_V_vnot_V(mask));
  flo=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(f,mask),z),f,flo);
  flo=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(z,exponent),
      Q6_V_vmux_QVV(neg,Q6_V_vsplat_R(0xbf800000),z),flo);
  HVX_Vector kexp=Q6_Vw_vsub_VwVw(Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(flo,23),Q6_V_vsplat_R(255)),Q6_V_vsplat_R(127));
  HVX_Vector k=Q6_Vw_vasr_VwVw(Q6_V_vor_VV(Q6_V_vand_VV(flo,Q6_V_vsplat_R(0x7fffff)),Q6_V_vsplat_R(0x800000)),Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(23),kexp));
  k=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(z,kexp),z,k);
  k=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(z,flo),Q6_Vw_vsub_VwVw(z,k),k);
  HVX_Vector r=Q6_Vqf32_vsub_Vqf32Vqf32(Q6_Vqf32_vadd_VsfVsf(x,z),Q6_Vqf32_vmpy_VsfVsf(flo,Q6_V_vsplat_R(0x3f317218)));
  r=Q6_Vqf32_vadd_Vqf32Vsf(r,z);
  HVX_Vector y=Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(0x39506967),z);
  const int coeff[5]={0x3ab743ce,0x3c088908,0x3d2aa9c1,0x3e2aaaaa,0x3f000000};
  for(int i=0;i<5;i++) {
    y=Q6_Vqf32_vmpy_Vqf32Vqf32(y,r);
    y=Q6_Vqf32_vadd_Vqf32Vsf(y,Q6_V_vsplat_R(coeff[i]));
    y=Q6_Vqf32_vadd_Vqf32Vsf(y,z);
  }
  HVX_Vector r2=Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(r,r),z);
  y=Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(y,r2),r);
  y=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(y,one));
  y=Q6_Vw_vaslacc_VwVwR(y,k,23);
  HVX_VectorPred minus_inf=Q6_Q_vcmp_eq_VwVw(input,Q6_V_vsplat_R(0xff800000));
  y=Q6_V_vmux_QVV(minus_inf,z,y);
  HVX_Vector absolute=Q6_V_vand_VV(input,Q6_V_vsplat_R(0x7fffffff));
  HVX_Vector flags=Q6_V_vmux_QVV(
      Q6_Q_vcmp_gt_VuwVuw(absolute,Q6_V_vsplat_R(0x42ae0000)),
      Q6_V_vsplat_R(-1),z);
  flags=Q6_V_vmux_QVV(minus_inf,z,flags);
  for(int shift=64;shift>=4;shift/=2)
    flags=Q6_V_vor_VV(flags,Q6_V_vror_VR(flags,shift));
  if(Q6_R_vextract_VR(flags,0)==0) return y;
  TL_EXP_PATH(2);
  return exp32_repair(input,y);
}
TL_DEVICE HVX_Vector native_exp32(HVX_Vector input) {
  return native_exp32_impl<true>(input);
}
}
