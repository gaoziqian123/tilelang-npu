#pragma once
#ifdef __hexagon__
#include "hvx.h"
#else
#define TL_DEVICE inline
#endif
#include <math.h>
#include <string.h>
namespace tl {
// Do not let Hexagon scalar max instruction selection replace ordered select
// with NaN-suppressing sfmax. NaNs reset the ordered fold to the RHS.
TL_DEVICE float row_ordered_max(float a,float b) {
  unsigned ua,ub; __builtin_memcpy(&ua,&a,4); __builtin_memcpy(&ub,&b,4);
  if((ub&0x7fffffffu)>0x7f800000u) return b;
  if((ua&0x7fffffffu)>0x7f800000u) return b;
  return a>b?a:b;
}
// Ordered exceptional-value fallback preserves the scalar compare/select
// semantics (including NaNs), while finite rows use fp32 vector reduction.
template<bool Max> TL_DEVICE float row_reduce_f32(const float* p, int n, float initial) {
  bool exceptional = !isfinite(initial);
  // Neutral -infinity for max is valid; other exceptional seeds retain order.
  if (Max && initial == -INFINITY) exceptional = false;
#ifdef __hexagon__
  HVX_Vector flags=Q6_V_vzero(); int scan=0;
  HVX_Vector acc=Q6_V_vsplat_R(Max ? 0xff800000 : 0);
  for(;scan+32<=n;scan+=32) {
    HVX_Vector x; __builtin_memcpy(&x,p+scan,128);
    // Classification and reduction consume the same load. Do not materialize
    // a second row pass (or keep every unrolled row vector live until the
    // exception branch). Exceptional results below are discarded and the
    // original ordered scalar fold is replayed with the unchanged seed.
    acc=Max ? Q6_Vsf_vmax_VsfVsf(acc,x) : Q6_Vsf_vadd_VsfVsf(acc,x);
    x=Q6_V_vand_VV(x,Q6_V_vsplat_R(0x7fffffff));
    flags=Q6_V_vor_VV(flags,Q6_V_vmux_QVV(
        Q6_Q_vcmp_gt_VuwVuw(x,Q6_V_vsplat_R(Max?0x7f800000:0x7f7fffff)),
        Q6_V_vsplat_R(-1),Q6_V_vzero()));
  }
  for(int shift=64;shift>=4;shift/=2)
    flags=Q6_V_vor_VV(flags,Q6_V_vror_VR(flags,shift));
  exceptional |= Q6_R_vextract_VR(flags,0)!=0;
  for(;scan<n;scan++) if(Max?isnan(p[scan]):!isfinite(p[scan])) exceptional=true;
#else
  for (int i=0;i<n;i++) if (Max ? isnan(p[i]) : !isfinite(p[i])) exceptional=true;
#endif
  if (exceptional) {
    for(int i=0;i<n;i++) initial=Max ? row_ordered_max(initial,p[i]) : initial+p[i];
    return initial;
  }
#ifdef __hexagon__
  int i=n-(n%32);
  for(int shift=64;shift>=4;shift/=2) {
    HVX_Vector x=Q6_V_vror_VR(acc,shift);
    acc=Max ? Q6_Vsf_vmax_VsfVsf(acc,x) : Q6_Vsf_vadd_VsfVsf(acc,x);
  }
  alignas(128) float lanes[32]; __builtin_memcpy(lanes,&acc,128);
#else
  float lanes[32]; for(int j=0;j<32;j++) lanes[j]=Max ? -INFINITY : 0.f;
  int i=0;
  for(;i+32<=n;i+=32) for(int j=0;j<32;j++)
    lanes[j]=Max ? (lanes[j]>p[i+j]?lanes[j]:p[i+j]) : lanes[j]+p[i+j];
  for(int shift=16;shift>=1;shift/=2) {
    float next[32];
    for(int j=0;j<32;j++) next[j]=Max ?
        (lanes[j]>lanes[(j+shift)%32]?lanes[j]:lanes[(j+shift)%32]) :
        lanes[j]+lanes[(j+shift)%32];
    __builtin_memcpy(lanes,next,sizeof lanes);
  }
#endif
  initial=Max ? (initial>lanes[0]?initial:lanes[0]) : initial+lanes[0];
  for(;i<n;i++) initial=Max ? (initial>p[i]?initial:p[i]) : initial+p[i];
  return initial;
}
}
