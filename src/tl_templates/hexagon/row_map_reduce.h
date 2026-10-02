#pragma once
#include "row_reduce.h"
namespace tl {
// Incremental form of row_reduce_f32. No map expression lives in this leaf.
// State: 32 lane chains + 32 classification words (stored as float bits).
template<bool Max> TL_DEVICE void row_map_init(float* s) {
#ifdef __hexagon__
  HVX_Vector a=Q6_V_vsplat_R(Max?0xff800000:0), f=Q6_V_vzero();
  __builtin_memcpy(s,&a,128); __builtin_memcpy(s+32,&f,128);
#else
  for(int j=0;j<32;j++) { s[j]=Max?-INFINITY:0.f; s[32+j]=0.f; }
#endif
}
template<bool Max> TL_DEVICE void row_map_update(float* s, const float* values,
                                                int offset, int count, int n) {
#ifdef __hexagon__
  if(count==32 && offset%32==0 && offset+32<=n) {
    HVX_Vector a,x,f; __builtin_memcpy(&a,s,128);
    __builtin_memcpy(&x,values,128); __builtin_memcpy(&f,s+32,128);
    a=Max?Q6_Vsf_vmax_VsfVsf(a,x):Q6_Vsf_vadd_VsfVsf(a,x);
    x=Q6_V_vand_VV(x,Q6_V_vsplat_R(0x7fffffff));
    f=Q6_V_vor_VV(f,Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(x,
        Q6_V_vsplat_R(Max?0x7f800000:0x7f7fffff)),Q6_V_vsplat_R(-1),Q6_V_vzero()));
    __builtin_memcpy(s,&a,128); __builtin_memcpy(s+32,&f,128); return;
  }
#endif
  for(int k=0;k<count;k++) {
    int i=offset+k, lane=i%32; float x=values[k];
    if(Max?isnan(x):!isfinite(x)) { unsigned f=~0u; __builtin_memcpy(s+32+lane,&f,4); }
    if(i<n-n%32) s[lane]=Max?(s[lane]>x?s[lane]:x):s[lane]+x;
  }
}
template<bool Max> TL_DEVICE float row_map_finish(float* s,const float* p,int n,float seed) {
  bool exceptional=!isfinite(seed);
  if(Max && seed==-INFINITY) exceptional=false;
#ifdef __hexagon__
  HVX_Vector flags; __builtin_memcpy(&flags,s+32,128);
  for(int shift=64;shift>=4;shift/=2) flags=Q6_V_vor_VV(flags,Q6_V_vror_VR(flags,shift));
  exceptional |= Q6_R_vextract_VR(flags,0)!=0;
#else
  for(int j=0;j<32;j++) { unsigned f; __builtin_memcpy(&f,s+32+j,4); exceptional|=f!=0; }
#endif
  if(exceptional) {
    for(int i=0;i<n;i++) seed=Max?row_ordered_max(seed,p[i]):seed+p[i];
    return seed;
  }
#ifdef __hexagon__
  HVX_Vector a; __builtin_memcpy(&a,s,128);
  for(int shift=64;shift>=4;shift/=2) {
    HVX_Vector x=Q6_V_vror_VR(a,shift);
    a=Max?Q6_Vsf_vmax_VsfVsf(a,x):Q6_Vsf_vadd_VsfVsf(a,x);
  }
  __builtin_memcpy(s,&a,128);
#else
  for(int shift=16;shift>=1;shift/=2) {
    float next[32];
    for(int j=0;j<32;j++) next[j]=Max?(s[j]>s[(j+shift)%32]?s[j]:s[(j+shift)%32]):s[j]+s[(j+shift)%32];
    __builtin_memcpy(s,next,128);
  }
#endif
  seed=Max?(seed>s[0]?seed:s[0]):seed+s[0];
  for(int i=n-n%32;i<n;i++) seed=Max?(seed>p[i]?seed:p[i]):seed+p[i];
  return seed;
}
}
