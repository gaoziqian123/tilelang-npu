#pragma once

#include "hvx.h"

// Packet strings follow ggml-hexagon/htp/hmx-utils.h. range is a byte range,
// 2048*n-1, not a tile count. Only ACTIVATION has :deep.
#define HMX_LOAD_MPY_F16(act, wt, range) \
  "{\n activation.hf = mxmem(" act ", " range ")\n" \
  " weight.hf = mxmem(" wt ", " range ")\n}\n"
#define HMX_LOAD_MPY_DEEP_F16(act, wt, range) \
  "{\n activation.hf = mxmem(" act ", " range "):deep\n" \
  " weight.hf = mxmem(" wt ", " range ")\n}\n"
#define HMX_STORE_AFTER_F16(out, scale_reg) \
  "mxmem(" out ", " scale_reg "):after.hf = acc\n"
#define HMX_SET_BIAS(scales) "bias = mxmem2(" scales ")\n"
#define HMX_CLRACC_F16() "mxclracc.hf\n"

namespace tl {
// Caller owns HMX on the RPC/open thread and passes VTCM AH/WH operands.
// No lock, accumulator clear or scale mutation is hidden in an MMA call.
template <int Tiles, bool Deep> TL_DEVICE void hmx_mma_f16_impl(const void *act, const void *wt) {
  static_assert(Tiles > 0, "positive K tile count required");
  hex_require(is_aligned<128>(act) && is_aligned<128>(wt));
  auto a = static_cast<const unsigned char *>(act);
  auto w = static_cast<const unsigned char *>(wt);
  for (int i = 0; i < Tiles; i += 32) {
    unsigned range = unsigned(((Tiles - i < 32) ? Tiles - i : 32) * 2048 - 1);
    if constexpr (Deep)
      asm volatile(HMX_LOAD_MPY_DEEP_F16("%0", "%1", "%2") : : "r"(a), "r"(w), "r"(range) : "memory");
    else
      asm volatile(HMX_LOAD_MPY_F16("%0", "%1", "%2") : : "r"(a), "r"(w), "r"(range) : "memory");
    // Do not form pointers beyond the final operand allocation.
    if (i + 32 < Tiles) { a += 65536; w += 65536; }
  }
}
template <int Tiles> TL_DEVICE void hmx_mma_deep_f16(const void *act, const void *wt) {
  hmx_mma_f16_impl<Tiles, true>(act, wt);
}
template <int Tiles> TL_DEVICE void hmx_mma_f16(const void *act, const void *wt) {
  hmx_mma_f16_impl<Tiles, false>(act, wt);
}
TL_DEVICE void hmx_clear_f16() {
  asm volatile(HMX_CLRACC_F16() : : : "memory");
}
// scales is a 256-byte aligned-storage region (128-byte address alignment):
// one WORD-splat scale vector followed by zero padding, as in ggml-hexagon.
// Halfword splat is NOT equivalent: each word holds scale and bias.
TL_DEVICE void hmx_init_scale(void *scales, half_t scale) {
  hex_require(is_aligned<128>(scales));
  uint16_t bits;
  __builtin_memcpy(&bits, &scale, 2);
  auto p = static_cast<HVX_Vector *>(scales);
  p[0] = Q6_V_vsplat_R(bits);
  p[1] = Q6_V_vzero();
  asm volatile(HMX_SET_BIAS("%0") : : "r"(scales) : "memory");
}
TL_DEVICE void hmx_store_after_f16(void *out, const void *scales) {
  hex_require(is_aligned<128>(out) && is_aligned<128>(scales));
  // init_scale binds bias before CLRACC/MMA; register selector is zero,
  // never the scales address. Do not mix legacy readout into a live session.
  asm volatile(HMX_STORE_AFTER_F16("%0", "%1")
               : : "r"(out), "r"(0) : "memory");
}
} // namespace tl
