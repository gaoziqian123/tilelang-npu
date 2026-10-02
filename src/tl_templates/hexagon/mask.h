#pragma once
#ifdef __hexagon__
#include "hvx.h"
#endif
namespace tl {
// Contiguous in-place select: p[j] = j <= last ? p[j] : fill.
inline void mask_prefix_f32(float* p,int n,int last,float fill) {
#ifdef __hexagon__
  alignas(128) static const int ix[32]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
  HVX_Vector index=*reinterpret_cast<const HVX_Vector*>(ix);
  int bits; __builtin_memcpy(&bits,&fill,4);
  HVX_Vector value=Q6_V_vsplat_R(bits);
  int j=0;
  for(;j+32<=n;j+=32) {
    HVX_Vector x; __builtin_memcpy(&x,p+j,128);
    HVX_VectorPred masked=Q6_Q_vcmp_gt_VwVw(Q6_Vw_vadd_VwVw(index,Q6_V_vsplat_R(j)),Q6_V_vsplat_R(last));
    x=Q6_V_vmux_QVV(masked,value,x);
    __builtin_memcpy(p+j,&x,128);
  }
  for(;j<n;j++) if(j>last) p[j]=fill;
#else
  for(int j=0;j<n;j++) if(j>last) p[j]=fill;
#endif
}
}
