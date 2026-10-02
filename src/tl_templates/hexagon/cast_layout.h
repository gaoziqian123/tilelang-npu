#pragma once
#include <stdint.h>
#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

namespace tl {
namespace detail {
using cast_words = uint32_t __attribute__((vector_size(128)));

// Exact binary16 -> binary32 expansion, including gradual subnormals. NaNs
// retain sign/payload and are quieted. Integer vector operations avoid native
// conversion flush-to-zero / floating-environment dependencies.
__attribute__((always_inline)) inline cast_words float_bits_from_half(const void *src, int part = 0) {
#ifdef __hexagon__
  HVX_Vector raw;
  __builtin_memcpy(&raw, src, 128);
  __asm__ volatile("" : "+v"(raw));
  HVX_VectorPair unpacked = Q6_Wuw_vunpack_Vuh(raw);
  HVX_Vector words = part ? Q6_V_hi_W(unpacked) : Q6_V_lo_W(unpacked);
  cast_words u;
  __builtin_memcpy(&u, &words, 128);
#else
  using halves = uint16_t __attribute__((vector_size(64)));
  halves h;
  __builtin_memcpy(&h, static_cast<const char*>(src) + part*64, 64);
  cast_words u = __builtin_convertvector(h, cast_words);
#endif
  cast_words sign = (u & 0x8000u) << 16;
  cast_words exponent = (u >> 10) & 31u;
  cast_words mantissa = u & 1023u;
  cast_words normal = ((exponent + 112u) << 23) | (mantissa << 13);
  cast_words m = mantissa, e = cast_words{} + 113u;
  for (int i = 0; i < 10; ++i) {
    auto shift = m < 1024u;
    m = shift ? m << 1 : m;
    e = shift ? e - 1u : e;
  }
  cast_words sub = (e << 23) | ((m & 1023u) << 13);
  sub = (mantissa == 0u) ? cast_words{} : sub;
  cast_words special = (cast_words{} + 0x7f800000u) | (mantissa << 13);
  special = (mantissa != 0u) ? special | 0x400000u : special;
  return sign | ((exponent == 31u) ? special : ((exponent == 0u) ? sub : normal));
}

// IEEE binary32 -> binary16, RNE, independent of the FP environment.  Integer
// lanes preserve subnormals, signed zero, infinities and quiet NaNs (including
// their sign/payload).  Every variable shift is clamped before evaluation.
__attribute__((always_inline)) inline cast_words half_bits_rne(const float *src) {
  cast_words u;
  __builtin_memcpy(&u, src, 128);
  cast_words sign = (u >> 16) & 0x8000u;
  cast_words mag = u & 0x7fffffffu;
  cast_words exp = mag >> 23;
  cast_words normal = ((mag + 0xfffu + ((mag >> 13) & 1u)) >> 13) - 0x1c000u;
  cast_words shift = 126u - exp;
  shift = (exp < 102u) ? cast_words{} + 24u : shift;
  shift = (exp > 112u) ? cast_words{} + 14u : shift;
  cast_words mant = (mag & 0x7fffffu) | 0x800000u;
  cast_words sub = (mant + ((cast_words{} + 1u) << (shift - 1u)) - 1u +
                    ((mant >> shift) & 1u)) >> shift;
  sub = (exp < 102u) ? cast_words{} : sub;
  cast_words out = (exp < 113u) ? sub : normal;
  out = (mag >= 0x477ff000u) ? cast_words{} + 0x7c00u : out;
  out = (mag > 0x7f800000u) ? ((mag >> 13) | 0x200u) & 0x7fffu : out;
  return sign | out;
}
} // namespace detail

namespace detail {
// Physical row/column pair: even half lanes from a, odd half lanes from b.
// v79 SDK19 lowers the standard intrinsic to sf->qf32 plus paired qf32->hf.
// Finite results use RNE independently of USR rounding bits (device boundary
// probe covers all four modes). Its NaNs canonicalize and -0 becomes +0;
// repair payload/sign with integer vector selects, not scalar lane fallbacks.
// This function neither reads nor writes USR. The host path remains the exact
// integer implementation; host tests alone do not validate the HVX path.
__attribute__((always_inline)) inline cast_words half_pair_rne(const float *a,
                                                              const float *b) {
#ifdef __hexagon__
  HVX_Vector va, vb;
  __builtin_memcpy(&va, a, 128);
  __builtin_memcpy(&vb, b, 128);
  // Preserve complete vector values through LLVM's aggregate/SROA passes.
  // Without this register constraint SDK19 can scalarize payload repair.
  __asm__ volatile("" : "+v"(va), "+v"(vb));
  HVX_Vector h = Q6_Vhf_vcvt_VsfVsf(va, vb);
  cast_words ua, ub, out;
  __builtin_memcpy(&ua, &va, 128);
  __builtin_memcpy(&ub, &vb, 128);
  __builtin_memcpy(&out, &h, 128);
  cast_words ma=ua&0x7fffffffu, mb=ub&0x7fffffffu;
  cast_words na=((ua>>13)|0x200u)&0x7fffu;
  cast_words nb=((ub>>13)|0x200u)&0x7fffu;
  cast_words lo=(ma>0x7f800000u)?na:(out&0x7fffu);
  cast_words hi=(mb>0x7f800000u)?nb:((out>>16)&0x7fffu);
  return lo | (hi<<16) | ((ua>>16)&0x8000u) | (ub&0x80000000u);
#else
  return half_bits_rne(a) | (half_bits_rne(b)<<16);
#endif
}
} // namespace detail

// Linear-order numeric conversion, including a single 32-lane half register.
// The last input is duplicated for a 32-lane tail: no speculative source read.
// memcpy preserves the destination's actual alignment and exact byte extent.
template <int Lanes>
__attribute__((always_inline)) inline void cast_linear_f32_f16(void *dst,
                                                              const float *src) {
  static_assert(Lanes > 0 && Lanes % 32 == 0);
  for (int i = 0; i < Lanes; i += 64) {
    auto pair = detail::half_pair_rne(src+i, src+i+(i+32<Lanes ? 32 : 0));
#ifdef __hexagon__
    HVX_Vector raw;
    __builtin_memcpy(&raw, &pair, 128);
    raw = Q6_Vh_vdeal_Vh(raw);
    __builtin_memcpy(static_cast<char *>(dst)+2*i, &raw,
                     i+32<Lanes ? 128 : 64);
#else
    using halves = uint16_t __attribute__((vector_size(64)));
    halves lo = __builtin_convertvector(pair & 0xffffu, halves);
    __builtin_memcpy(static_cast<char *>(dst)+2*i, &lo, 64);
    if (i+32<Lanes) {
      halves hi = __builtin_convertvector(pair >> 16, halves);
      __builtin_memcpy(static_cast<char *>(dst)+2*i+64, &hi, 64);
    }
#endif
  }
}

// One complete physical vector per iteration. AH pairs rows, WH pairs columns.
// Source is ordinary private memory: WH's gather never touches VTCM scalarly.
// Lowering proves bounds, compact strides, disjoint scopes and 128B destination
// alignment. No enclosing-block overread and no half-precision scratch buffer.
template <bool WH>
__attribute__((always_inline)) inline void cast_pack_f32_f16(void *dst, const float *src, int columns,
                                int stride, int row, int col, int rows, int cols) {
  auto d = static_cast<detail::cast_words *>(dst);
  for (int r = 0; r < rows; r += WH ? 32 : 2)
    for (int c = 0; c < cols; c += WH ? 2 : 32) {
      detail::cast_words packed;
      if constexpr (WH) {
        alignas(128) float x[32], y[32];
        for (int i = 0; i < 32; ++i) {
          x[i] = src[(r+i)*stride+c];
          y[i] = src[(r+i)*stride+c+1];
        }
        packed = detail::half_pair_rne(x,y);
      } else {
        packed = detail::half_pair_rne(src+r*stride+c,src+(r+1)*stride+c);
      }
      int tile = ((row+r)/32)*(columns/32)+(col+c)/32;
      int pair = WH ? ((col+c)%32)/2 : ((row+r)%32)/2;
      d[tile*16+pair] = packed;
    }
}
} // namespace tl
