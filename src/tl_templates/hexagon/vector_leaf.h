#pragma once
#include "hvx.h"
#include "cast_layout.h"
#include "hf_math.h"
#include "vector_math32.h"
#include <type_traits>

// Physical register leaves. No loop/owner inference belongs in this file.
namespace tl::hvx_leaf {
TL_DEVICE HVX_Vector exp32_v1(HVX_Vector x) { return hvx_math32_v1::exp_f32(x); }
TL_DEVICE HVX_Vector pred_and(HVX_Vector a,HVX_Vector b) { return Q6_V_vand_VV(a,b); }
TL_DEVICE HVX_Vector pred_or(HVX_Vector a,HVX_Vector b) { return Q6_V_vor_VV(a,b); }
TL_DEVICE HVX_Vector pred_not(HVX_Vector a) { return Q6_V_vnot_V(a); }
// ggml/reference polynomial contract, not correctly-rounded IEEE exp2.
// Preserve QF constants, evaluation order, HF boundaries and exponent flush.
// Finite nonpositive values and -inf accepted; NaNs/positive values rejected.
TL_DEVICE HVX_Vector exp2_16_nonpositive(HVX_Vector input) {
  HVX_Vector z=Q6_V_vzero(), ones=Q6_V_vsplat_R(-1);
  HVX_Vector mag=Q6_V_vand_VV(input,Q6_Vh_vsplat_R(0x7fff));
  HVX_Vector bad=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuhVuh(mag,Q6_Vh_vsplat_R(0x7c00)),ones,z);
  bad=Q6_V_vor_VV(bad,Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(input,z),ones,z));
  for(int s=2;s<=64;s*=2) bad=Q6_V_vor_VV(bad,Q6_V_vror_VR(bad,s));
  hex_require_id(Q6_R_vextract_VR(bad,0)==0, 1001);
  auto ni=Q6_Q_vcmp_eq_VhVh(input,Q6_Vh_vsplat_R(0xfc00));
  HVX_Vector x=Q6_Vhf_vmax_VhfVhf(Q6_Vh_vsplat_R(0xce00),input);
  HVX_Vector k=Q6_Vh_equals_Vhf(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(x,Q6_Vh_vsplat_R(0x3800))));
  HVX_Vector f=Q6_Vqf16_vsub_VhfVhf(x,Q6_Vhf_equals_Vh(k));
  HVX_Vector y=Q6_Vqf16_vmpy_Vqf16Vqf16(Q6_Vh_vsplat_R(0x5082),f);
  y=Q6_Vqf16_vadd_Vqf16Vhf(y,Q6_Vh_vsplat_R(0x157d));
  y=Q6_Vqf16_vmpy_Vqf16Vqf16(y,f);
  y=Q6_Vqf16_vadd_Vqf16Vhf(y,Q6_Vh_vsplat_R(0x20ed));
  y=Q6_Vqf16_vmpy_Vqf16Vqf16(y,f);
  y=Q6_Vqf16_vadd_Vqf16Vhf(y,Q6_Vh_vsplat_R(0x2b1b));
  y=Q6_Vqf16_vmpy_Vqf16Vqf16(y,f);
  y=Q6_Vqf16_vadd_Vqf16Vhf(y,Q6_Vh_vsplat_R(0x33b0));
  y=Q6_Vqf16_vmpy_Vqf16Vqf16(y,f);
  y=Q6_Vqf16_vadd_Vqf16Vhf(y,Q6_Vh_vsplat_R(0x398c));
  y=Q6_Vqf16_vmpy_Vqf16Vqf16(y,f);
  y=Q6_Vqf16_vadd_Vqf16Vhf(y,Q6_Vh_vsplat_R(0x3c00));
  y=Q6_Vhf_equals_Vqf16(y);
  HVX_Vector exponent=Q6_Vh_vadd_VhVh(k,Q6_Vuh_vlsr_VuhR(Q6_Vh_vasl_VhR(y,1),11));
  y=Q6_Vh_vaslacc_VhVhR(y,k,10);
  y=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(z,exponent),z,y);
  return Q6_V_vmux_QVV(ni,z,y);
}
TL_DEVICE HVX_Vector load(const void *p, int valid_bytes, HVX_Vector fill) {
  hex_require(is_aligned<128>(p) && valid_bytes>=0 && valid_bytes<=128);
  HVX_Vector x=*static_cast<const HVX_Vector*>(p);
  if(valid_bytes==128) return x;
  return Q6_V_vmux_QVV(Q6_Q_vsetq_R(valid_bytes),x,fill);
}
TL_DEVICE void store(void *p, HVX_Vector x, int valid_bytes) {
  hex_require(is_aligned<128>(p) && valid_bytes>=0 && valid_bytes<=128);
  if(valid_bytes==128) *static_cast<HVX_Vector*>(p)=x;
  else if(valid_bytes) Q6_vmem_QRIV(Q6_Q_vsetq_R(valid_bytes),static_cast<HVX_Vector*>(p),x);
}
TL_DEVICE HVX_Vector splat32(unsigned bits) { return Q6_V_vsplat_R(bits); }
TL_DEVICE HVX_Vector splat16(unsigned bits) { return Q6_Vh_vsplat_R(bits); }
TL_DEVICE unsigned extract32(HVX_Vector x,int lane) {
  hex_require(lane>=0 && lane<32); return Q6_R_vextract_VR(x,lane*4);
}
TL_DEVICE unsigned extract16(HVX_Vector x,int lane) {
  hex_require(lane>=0 && lane<64);
  return (Q6_R_vextract_VR(x,(lane/2)*4)>>((lane%2)*16))&65535;
}
TL_DEVICE HVX_Vector fma16(HVX_Vector a,HVX_Vector b,HVX_Vector c) {
  return Q6_Vhf_vmpyacc_VhfVhfVhf(c,a,b);
}
// IEEE bit ordering comparisons: all NaNs detected before comparison, signed
// zeros equal, subnormals compared without target FTZ arithmetic. Byte mask.
template<int Bits> TL_DEVICE HVX_Vector compare(HVX_Vector a,HVX_Vector b,int relation,int unordered) {
  using W=uint32_t __attribute__((vector_size(128)));
  using H=uint16_t __attribute__((vector_size(128)));
  using U=typename std::conditional<Bits==16,H,W>::type;
  U x,y; __builtin_memcpy(&x,&a,128); __builtin_memcpy(&y,&b,128);
  constexpr unsigned sign=Bits==16?0x8000u:0x80000000u;
  constexpr unsigned inf=Bits==16?0x7c00u:0x7f800000u;
  U ax=x&(sign-1), ay=y&(sign-1);
  auto nan=(ax>inf)|(ay>inf);
  auto eq=(x==y)|((ax==0)&(ay==0));
  U kx=(x&sign)?~x:(x^sign), ky=(y&sign)?~y:(y^sign);
  auto lt=(kx<ky)&~eq;
  auto gt=(kx>ky)&~eq;
  auto result=eq;
  switch(relation) {
    case 0: break; case 1: result=~eq; break; case 2: result=lt; break;
    case 3: result=lt|eq; break; case 4: result=gt; break; case 5: result=gt|eq; break;
    default: hex_require(false);
  }
  result=unordered?(result|nan):(result&~nan);
  HVX_Vector out; __builtin_memcpy(&out,&result,128); return out;
}
// Same coefficients and HF rounding as exp2_hf, but invalid lanes trap rather
// than invoking libm. This is a bounded approximation, not strict IEEE exp2.
TL_DEVICE HVX_Vector exp2_16_bounded(HVX_Vector x) {
  return exp2_hf_impl<true>(x);
}
TL_DEVICE HVX_Vector rotate(HVX_Vector x, int bytes) {
  hex_require(bytes >= 0 && bytes < 128);
  return Q6_V_vror_VR(x, bytes);
}
// out[j] = [b,a][j+bytes], in bytes.
TL_DEVICE HVX_Vector align(HVX_Vector a, HVX_Vector b, int bytes) {
  hex_require(bytes >= 0 && bytes < 128);
  return Q6_V_valign_VVR(a,b,bytes);
}
TL_DEVICE HVX_Vector interleave16(HVX_Vector x) { return Q6_Vh_vshuff_Vh(x); }
TL_DEVICE HVX_Vector deal16(HVX_Vector x) { return Q6_Vh_vdeal_Vh(x); }
TL_DEVICE HVX_Vector select(HVX_Vector mask, HVX_Vector a, HVX_Vector b) {
  return Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VbVb(mask,Q6_V_vsplat_R(-1)),a,b);
}
TL_DEVICE HVX_Vector abs32(HVX_Vector x) { return Q6_V_vand_VV(x,Q6_V_vsplat_R(0x7fffffff)); }
TL_DEVICE HVX_Vector neg32(HVX_Vector x) { return Q6_V_vxor_VV(x,Q6_V_vsplat_R(0x80000000)); }
TL_DEVICE HVX_Vector abs16(HVX_Vector x) { return Q6_V_vand_VV(x,Q6_Vh_vsplat_R(0x7fff)); }
TL_DEVICE HVX_Vector neg16(HVX_Vector x) { return Q6_V_vxor_VV(x,Q6_Vh_vsplat_R(0x8000)); }
TL_DEVICE HVX_Vector add32(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vadd_VsfVsf(a,b); }
TL_DEVICE HVX_Vector sub32(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vsub_VsfVsf(a,b); }
TL_DEVICE HVX_Vector mul32(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vmpy_VsfVsf(a,b); }
TL_DEVICE HVX_Vector max32(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vmax_VsfVsf(a,b); }
TL_DEVICE HVX_Vector min32(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_vmin_VsfVsf(a,b); }
TL_DEVICE HVX_Vector add16(HVX_Vector a, HVX_Vector b) { return Q6_Vhf_vadd_VhfVhf(a,b); }
TL_DEVICE HVX_Vector sub16(HVX_Vector a, HVX_Vector b) { return Q6_Vhf_vsub_VhfVhf(a,b); }
TL_DEVICE HVX_Vector mul16(HVX_Vector a, HVX_Vector b) { return Q6_Vhf_vmpy_VhfVhf(a,b); }
TL_DEVICE HVX_Vector max16(HVX_Vector a, HVX_Vector b) { return Q6_Vhf_vfmax_VhfVhf(a,b); }
TL_DEVICE HVX_Vector min16(HVX_Vector a, HVX_Vector b) { return Q6_Vhf_vfmin_VhfVhf(a,b); }
// Tree and pair direction are ABI. No seed is implicitly inserted.
TL_DEVICE HVX_Vector sum32_asc(HVX_Vector x) {
  for (int bytes=4; bytes<=64; bytes*=2) x=add32(x,rotate(x,bytes));
  return Q6_V_vsplat_R(Q6_R_vextract_VR(x,0));
}
TL_DEVICE HVX_Vector max32_asc(HVX_Vector x) {
  for (int bytes=4; bytes<=64; bytes*=2) x=max32(x,rotate(x,bytes));
  return Q6_V_vsplat_R(Q6_R_vextract_VR(x,0));
}
TL_DEVICE HVX_Vector widen_linear(HVX_Vector x, int part) {
  hex_require(part==0 || part==1);
  auto bits=detail::float_bits_from_half(&x,part);
  HVX_Vector out; __builtin_memcpy(&out,&bits,128); return out;
}
TL_DEVICE HVX_Vector narrow_evenodd(HVX_Vector a, HVX_Vector b) {
  alignas(128) float av[32], bv[32];
  __builtin_memcpy(av, &a, sizeof(a));
  __builtin_memcpy(bv, &b, sizeof(b));
  auto bits=detail::half_pair_rne(av, bv);
  HVX_Vector out; __builtin_memcpy(&out,&bits,128); return out;
}
TL_DEVICE HVX_Vector narrow_linear(HVX_Vector a, HVX_Vector b) {
  return deal16(narrow_evenodd(a,b));
}
TL_DEVICE HVX_Vector widen_evenodd(HVX_Vector x, int part) {
  return widen_linear(deal16(x),part);
}
TL_DEVICE HVX_Vector broadcast32(HVX_Vector x, int lane) {
  hex_require(lane>=0 && lane<32);
  return Q6_V_vsplat_R(Q6_R_vextract_VR(x,lane*4));
}
TL_DEVICE HVX_Vector broadcast16(HVX_Vector x, int lane) {
  hex_require(lane>=0 && lane<64);
  unsigned word=Q6_R_vextract_VR(x,(lane/2)*4);
  return Q6_Vh_vsplat_R(word >> ((lane%2)*16));
}
// Versioned two-Newton approximation, NOT IEEE division. A vector guard rejects
// denominators outside [2^-60,2^60] before evaluating the approximation.
TL_DEVICE HVX_Vector rcp32_nr2(HVX_Vector x) {
  HVX_Vector z=Q6_V_vzero(), ones=Q6_V_vsplat_R(-1);
  HVX_Vector bad=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vsplat_R(0x21800000),x),ones,z);
  bad=Q6_V_vor_VV(bad,Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(x,Q6_V_vsplat_R(0x5d800000)),ones,z));
  for(int s=4;s<=64;s*=2) bad=Q6_V_vor_VV(bad,rotate(bad,s));
  hex_require_id(Q6_R_vextract_VR(bad,0)==0, 1002);
  HVX_Vector r=Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7ef311d3),x);
  for(int i=0;i<2;++i) r=mul32(r,sub32(Q6_V_vsplat_R(0x40000000),mul32(x,r)));
  return r;
}
TL_DEVICE HVX_Vector div32_nr2(HVX_Vector a, HVX_Vector b) {
  return mul32(a,rcp32_nr2(b));
}
} // namespace tl::hvx_leaf
