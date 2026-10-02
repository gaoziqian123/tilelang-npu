#pragma once
#include "hvx.h"
#include <math.h>

namespace tl {
// QFloat has a non-IEEE encoding. Keep it confined to typed leaf operations;
// never expose its register bits as an IEEE half expression or buffer.
struct qfloat16_register { HVX_Vector bits; };
TL_DEVICE HVX_Vector sub_hf(HVX_Vector a, HVX_Vector b) {
  qfloat16_register d{Q6_Vqf16_vsub_VhfVhf(a,b)};
  return Q6_Vhf_equals_Vqf16(d.bits);
}

// Polynomial adapted from llama.cpp (MIT), hvx-exp.h, revision fcc891.
// Copyright (c) 2023-2026 The ggml authors. See accompanying MIT notice.
// The fast domain deliberately excludes subnormal output; cold libm repair
// preserves all half subnormal results and IEEE exceptional classifications.
__attribute__((noinline,cold)) static HVX_Vector exp2_hf_repair(HVX_Vector x, HVX_Vector y) {
  alignas(128) half_t in[64], out[64];
  __builtin_memcpy(in,&x,128); __builtin_memcpy(out,&y,128);
  for (int i=0;i<64;++i)
    if (!((float)in[i]>=-14.f && (float)in[i]<=0.f))
      out[i]=(half_t)exp2((double)in[i]);
  __builtin_memcpy(&y,out,128); return y;
}
template <bool VectorRequired = false>
TL_DEVICE HVX_Vector exp2_hf_impl(HVX_Vector input) {
  const HVX_Vector z=Q6_V_vzero();
  // Unsigned magnitude and signed bits reject NaNs, infinities and positives.
  HVX_Vector bad=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuhVuh(
      Q6_V_vand_VV(input,Q6_Vh_vsplat_R(0x7fff)),Q6_Vh_vsplat_R(0x4b00)),Q6_Vh_vsplat_R(-1),z);
  bad=Q6_V_vor_VV(bad,Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VhVh(input,z),Q6_Vh_vsplat_R(-1),z));
  HVX_Vector x=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(bad,z),input,z);
  HVX_Vector k=Q6_Vh_equals_Vhf(sub_hf(x,Q6_Vh_vsplat_R(0x3800)));
  qfloat16_register f{Q6_Vqf16_vsub_VhfVhf(x,Q6_Vhf_equals_Vh(k))};
  // Cancellation in x-k leaves an unnormalized QF mantissa at large |x|.
  // Normalize at this exact HF boundary before the QF multiply chain.
  f.bits=Q6_Vqf16_vadd_VhfVhf(Q6_Vhf_equals_Vqf16(f.bits),z);
  // ln(2)^6/6! rounded to IEEE HF = 0x090c. Use the HF-input
  // multiply explicitly; no assumption about a raw QFloat constant encoding.
  qfloat16_register y{Q6_Vqf16_vmpy_VhfVhf(Q6_Vh_vsplat_R(0x090c),Q6_Vhf_equals_Vqf16(f.bits))};
  y.bits=Q6_Vqf16_vadd_Vqf16Vhf(y.bits,Q6_Vh_vsplat_R(0x157d));
  y.bits=Q6_Vqf16_vmpy_Vqf16Vqf16(y.bits,f.bits);
  y.bits=Q6_Vqf16_vadd_Vqf16Vhf(y.bits,Q6_Vh_vsplat_R(0x20ed));
  y.bits=Q6_Vqf16_vmpy_Vqf16Vqf16(y.bits,f.bits);
  y.bits=Q6_Vqf16_vadd_Vqf16Vhf(y.bits,Q6_Vh_vsplat_R(0x2b1b));
  y.bits=Q6_Vqf16_vmpy_Vqf16Vqf16(y.bits,f.bits);
  y.bits=Q6_Vqf16_vadd_Vqf16Vhf(y.bits,Q6_Vh_vsplat_R(0x33b0));
  y.bits=Q6_Vqf16_vmpy_Vqf16Vqf16(y.bits,f.bits);
  y.bits=Q6_Vqf16_vadd_Vqf16Vhf(y.bits,Q6_Vh_vsplat_R(0x398c));
  y.bits=Q6_Vqf16_vmpy_Vqf16Vqf16(y.bits,f.bits);
  y.bits=Q6_Vqf16_vadd_Vqf16Vhf(y.bits,Q6_Vh_vsplat_R(0x3c00));
  HVX_Vector out=Q6_Vh_vaslacc_VhVhR(Q6_Vhf_equals_Vqf16(y.bits),k,10);
  for(int s=64;s>=2;s/=2) bad=Q6_V_vor_VV(bad,Q6_V_vror_VR(bad,s));
  if constexpr (VectorRequired) {
    hex_require(Q6_R_vextract_VR(bad,0)==0);
  } else {
    if(Q6_R_vextract_VR(bad,0)) return exp2_hf_repair(input,out);
  }
  return out;
}
TL_DEVICE HVX_Vector exp2_hf(HVX_Vector input) {
  return exp2_hf_impl<false>(input);
}
} // namespace tl
