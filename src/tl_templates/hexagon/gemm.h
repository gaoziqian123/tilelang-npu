#pragma once

#include "hmx.h"

namespace tl {
namespace detail {
// A 64-byte row can straddle two aligned HVX blocks. Load only the blocks
// intersecting the row, then align it into the low half. Source allocations
// must make those enclosing 128-byte blocks readable (round allocation base
// down/end up to 128). No scalar loads, no unconditional extra-block read.
TL_DEVICE HVX_Vector gemm_load_row(const half_t *p) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  int off = addr & 127;
  auto base = reinterpret_cast<const HVX_Vector *>(addr & ~uintptr_t(127));
  HVX_Vector lo = base[0];
  HVX_Vector hi = off > 64 ? base[1] : lo;
  return Q6_V_valign_VVR(hi, lo, off);
}
// Predicated vector stores touch exactly 64 bytes; no read/modify/write of
// neighboring rows (which could belong to a different worker).
TL_DEVICE void gemm_store_row(half_t *p, HVX_Vector low) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  int off = addr & 127;
  auto base = reinterpret_cast<HVX_Vector *>(addr & ~uintptr_t(127));
  HVX_Vector v = Q6_V_vlalign_VVR(low, low, off);
  int end = off + 64 < 128 ? off + 64 : 128;
  HVX_VectorPred upper = end == 128 ? Q6_Q_not_Q(Q6_Q_vsetq_R(0)) : Q6_Q_vsetq_R(end);
  HVX_VectorPred mask = Q6_Q_xor_QQ(upper, Q6_Q_vsetq_R(off));
  Q6_vmem_QRIV(mask, base, v);
  if (off > 64) Q6_vmem_QRIV(Q6_Q_vsetq_R(off - 64), base + 1, v);
}
TL_DEVICE HVX_Vector gemm_row_pair(const half_t *a, const half_t *b) {
  HVX_Vector x = gemm_load_row(a), y = gemm_load_row(b);
  return Q6_V_vmux_QVV(Q6_Q_vsetq_R(64), x, Q6_V_vror_VR(y, 64));
}
// Register transpose from hexagon_rt.h hrt_transpose32_hvx. Local predicate
// construction avoids its mutable process-global initialization state.
TL_DEVICE void gemm_transpose32(HVX_Vector *t) {
  for (int i = 0; i < 4; ++i) {
    int j = i + 1, st = 1 << i;
    for (int v = 0; v < 16; ++v)
      for (int r = 0; r < j; ++r) t[v] = Q6_Vh_vdeal_Vh(t[v]);
    for (int a = 0; a < 16; ++a) if (!(a & st)) {
      int b = a | st;
      HVX_Vector e = Q6_Vh_vshuffe_VhVh(t[b], t[a]);
      HVX_Vector o = Q6_Vh_vshuffo_VhVh(t[b], t[a]);
      t[a] = e; t[b] = o;
    }
    for (int v = 0; v < 16; ++v)
      for (int r = 0; r < j; ++r) t[v] = Q6_Vh_vshuff_Vh(t[v]);
  }
  alignas(128) int16_t odd[64], dif[64]; // local stack, NOT VTCM
  for (int l = 0; l < 64; ++l) {
    odd[l] = l & 1;
    dif[l] = ((l >> 5) & 1) != (l & 1);
  }
  auto qo = Q6_Q_vcmp_gt_VhVh(*reinterpret_cast<HVX_Vector *>(odd), Q6_V_vzero());
  auto qd = Q6_Q_vcmp_gt_VhVh(*reinterpret_cast<HVX_Vector *>(dif), Q6_V_vzero());
  for (int v = 0; v < 16; ++v) {
    HVX_Vector r = Q6_V_vror_VR(t[v], 64);
    HVX_Vector sw = Q6_V_vmux_QVV(qo, Q6_Vh_vshuffe_VhVh(r, r), Q6_Vh_vshuffo_VhVh(r, r));
    t[v] = Q6_V_vmux_QVV(qd, sw, t[v]);
  }
}
} // namespace detail

// AH[M,N] -> WH[N,M] logical transpose preserves intra-tile bytes: WH
// already interleaves K pairs. Only the outer tile grid must be transposed.
// Input/output tile order is chosen by IR; no math or barrier hidden.
TL_DEVICE void gemm_transpose_packed_tile(void *dst, const void *src) {
  auto s=static_cast<const HVX_Vector *>(src);
  auto d=static_cast<HVX_Vector *>(dst);
  for(int r=0;r<16;r++) d[r]=s[r];
}

// Tiles advance along source rows: [Tiles*32,32], stride in half elements.
// Nonoverlapping source/destination; row_stride >= 32. Source enclosing-block
// readability contract is documented in detail::gemm_load_row above.
// Tiles is NOT a flattened (row-block, K-block) panel count: K never advances
// here. A [32,2560] block needs 80 calls with Tiles=1 and src += 32 per K
// tile, not one call with Tiles=80 (which would consume 2560 source rows).
// Packed output is contiguous across Tiles; callers with K-major inner tile
// order must also supply the per-(row,K) destination offset themselves.
template <int Tiles>
TL_DEVICE void gemm_pack_ah_strided(half_t *dst_vtcm, const half_t *src, int row_stride) {
  static_assert(Tiles > 0, "positive tile count required");
  hex_require(is_aligned<128>(dst_vtcm) && row_stride >= 32);
  auto d = reinterpret_cast<HVX_Vector *>(dst_vtcm);
  for (int t = 0; t < Tiles; ++t)
    for (int rp = 0; rp < 16; ++rp) {
      const half_t *a = src + size_t(t * 32 + 2 * rp) * row_stride;
      Q6_dcfetch_A(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(a) + 512));
      Q6_dcfetch_A(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(a + row_stride) + 512));
      d[t * 16 + rp] = Q6_Vh_vshuff_Vh(detail::gemm_row_pair(a, a + row_stride));
    }
}
// B[N,K]: Tiles advance along N, each consumes 32 rows and the first 32 K
template <int Pairs>
TL_DEVICE void gemm_pack_ah_pair_strided(half_t *dst, const half_t *src, int stride) {
  hex_require(is_aligned<128>(dst) && is_aligned<128>(src) && stride % 64 == 0);
  auto d = reinterpret_cast<HVX_Vector *>(dst);
  for (int t=0; t<Pairs; ++t)
    for (int r=0; r<16; ++r) {
      const half_t *a = src + size_t(t*32+2*r)*stride;
      Q6_dcfetch_A(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(a) + 512));
      Q6_dcfetch_A(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(a+stride) + 512));
      HVX_Vector x = *reinterpret_cast<const HVX_Vector *>(a);
      HVX_Vector y = *reinterpret_cast<const HVX_Vector *>(a+stride);
      HVX_VectorPair p = Q6_W_vshuff_VVR(y,x,-2);
      d[t*32+r] = Q6_V_lo_W(p);
      d[t*32+16+r] = Q6_V_hi_W(p);
    }
}
// columns at src. Transpose to W[k,n] before zip16. Macro owns K-panel offsets.
// Likewise a B[1024,2560] panel needs 32*80 individual tile calls in (n,k)
// order, NOT Tiles=2560. The bounded per-tile scratch below is reused, not
// allocated once per tile; none of these helpers recursively expand Tiles.
// TODO: measure DSP transpose cost; production wh_convert runs on the host.
template <int Tiles>
TL_DEVICE void gemm_pack_wh_nt_strided(half_t *dst_vtcm, const half_t *src, int row_stride) {
  static_assert(Tiles > 0, "positive tile count required");
  hex_require(is_aligned<128>(dst_vtcm) && row_stride >= 32);
  auto d = reinterpret_cast<HVX_Vector *>(dst_vtcm);
  for (int t = 0; t < Tiles; ++t) {
    // Official HTP word scatter: each loaded word holds two successive K
    // values. Offset k_pair*128+n*4 writes their interleaved WH row pair.
    // Only 64 source bytes belong to this tile. No scalar VTCM accesses.
    alignas(128) static const int offsets[32] = {
        0,128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,
        0,128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920};
    HVX_Vector base = *reinterpret_cast<const HVX_Vector *>(offsets);
    for (int r = 0; r < 32; ++r) {
      const half_t *p = src + size_t(t * 32 + r) * row_stride;
      Q6_dcfetch_A(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(p) + 256));
      HVX_Vector v = detail::gemm_load_row(p);
      HVX_Vector off = Q6_Vw_vadd_VwVw(base, Q6_V_vsplat_R(r * 4));
      Q6_vscatter_QRMVwV(Q6_Q_vsetq_R(64), reinterpret_cast<uintptr_t>(d + t * 16), 2047, off, v);
    }
  }
}
TL_DEVICE void gemm_scatter_release(void *dst) {
  asm volatile("vmem(%0 + #0):scatter_release" : : "r"(dst) : "memory");
}
// Paired K tiles, official HTP scatter offsets extended to region 4095.
template <int Pairs>
TL_DEVICE void gemm_pack_wh_nt_pair_strided(half_t *dst, const half_t *src, int stride) {
  hex_require(is_aligned<128>(dst) && is_aligned<128>(src) && stride % 64 == 0);
  alignas(128) static const int offsets[32] = {
    0,128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,
    2048,2176,2304,2432,2560,2688,2816,2944,3072,3200,3328,3456,3584,3712,3840,3968};
  HVX_Vector base = *reinterpret_cast<const HVX_Vector *>(offsets);
  for (int t=0; t<Pairs; ++t) {
    for (int r=0; r<32; ++r)
      Q6_dcfetch_A(const_cast<half_t *>(src + size_t(t*32+r)*stride));
    for (int r=0; r<32; ++r) {
      const half_t *p = src + size_t(t*32+r)*stride;
      Q6_dcfetch_A(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(p) + 256));
      HVX_Vector v = *reinterpret_cast<const HVX_Vector *>(p);
      HVX_Vector off = Q6_Vw_vadd_VwVw(base, Q6_V_vsplat_R(r*4));
      Q6_vscatter_RMVwV(reinterpret_cast<uintptr_t>(dst+t*2048),4095,off,v);
    }
  }
}
// Two adjacent N tiles: vdeal of their interleaved row pairs produces two
template <int K>
TL_DEVICE void gemm_pack_ah_strip(half_t *dst, const half_t *src, int stride) {
  static_assert(K > 0 && K % 64 == 0);
  hex_require(is_aligned<128>(dst) && is_aligned<128>(src) && stride % 64 == 0);
  auto d = reinterpret_cast<HVX_Vector *>(dst);
  for (int r=0; r<16; ++r)
    for (int k=0; k<K; k+=64) {
      const half_t *a = src + size_t(2*r)*stride + k;
      Q6_dcfetch_A(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(a) + 512));
      Q6_dcfetch_A(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(a+stride) + 512));
      HVX_Vector x = *reinterpret_cast<const HVX_Vector *>(a);
      HVX_Vector y = *reinterpret_cast<const HVX_Vector *>(a+stride);
      HVX_VectorPair p = Q6_W_vshuff_VVR(y,x,-2);
      d[k/2+r] = Q6_V_lo_W(p);
      d[k/2+16+r] = Q6_V_hi_W(p);
    }
}
template <int K>
TL_DEVICE void gemm_pack_wh_nt_strip(half_t *dst, const half_t *src, int stride) {
  static_assert(K % 64 == 0);
  hex_require(is_aligned<128>(dst) && is_aligned<128>(src) && stride % 64 == 0);
  alignas(128) static const int offsets[32] = {
    0,128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920,
    2048,2176,2304,2432,2560,2688,2816,2944,3072,3200,3328,3456,3584,3712,3840,3968};
  HVX_Vector base = *reinterpret_cast<const HVX_Vector *>(offsets);
  for (int r=0; r<32; ++r) {
    HVX_Vector off = Q6_Vw_vadd_VwVw(base, Q6_V_vsplat_R(r*4));
    for (int k=0; k<K; k+=64) {
      const half_t *p = src + size_t(r)*stride + k;
      Q6_dcfetch_A(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(p) + 2048));
      HVX_Vector v = *reinterpret_cast<const HVX_Vector *>(p);
      Q6_vscatter_RMVwV(reinterpret_cast<uintptr_t>(dst+k*32),4095,off,v);
    }
  }
}
// complete 128-byte RM rows. Lowering proves 64-column destination alignment.
// Read one logical AH row directly into ordinary private storage. Source
// addresses are complete aligned HVX registers; memcpy does not promise
// alignment for the private destination. No scalar VTCM access.
TL_DEVICE void gemm_unpack_ah_row(void *dst_void, const void *src_void, int columns, int row) {
  auto dst = static_cast<half_t *>(dst_void);
  auto src = static_cast<const HVX_Vector *>(src_void);
  hex_require(columns > 0 && columns % 32 == 0 && row >= 0 && is_aligned<128>(src));
  for (int c=0;c<columns;c+=64) {
    int tile=(row/32)*(columns/32)+c/32;
    int rp=(row%32)/2;
    if (c+32 == columns) {
      // A final single tile still owns a complete 128B row-pair register.
      // Read it once, then extract only this row's 64B into private memory.
      // Never read tile+1 or perform a half-vector VTCM load.
      HVX_Vector value=Q6_Vh_vdeal_Vh(src[tile*16+rp]);
      if (row&1) value=Q6_V_vror_VR(value,64);
      __builtin_memcpy(dst+c,&value,64);
      continue;
    }
    HVX_VectorPair pair=Q6_W_vdeal_VVR(src[(tile+1)*16+rp],src[tile*16+rp],-2);
    HVX_Vector value=(row&1)?Q6_V_hi_W(pair):Q6_V_lo_W(pair);
    __builtin_memcpy(dst+c,&value,128);
  }
}
template <int Pairs>
TL_DEVICE void gemm_unpack_ah_pair_strided(half_t *dst, const half_t *src, int stride) {
  hex_require(is_aligned<128>(dst) && is_aligned<128>(src) && stride % 64 == 0);
  auto s = reinterpret_cast<const HVX_Vector *>(src);
  for (int t=0; t<Pairs; ++t)
    for (int r=0; r<16; ++r) {
      HVX_VectorPair p = Q6_W_vdeal_VVR(s[t*32+16+r], s[t*32+r], -2);
      *reinterpret_cast<HVX_Vector *>(dst + (t*32+2*r)*stride) = Q6_V_lo_W(p);
      *reinterpret_cast<HVX_Vector *>(dst + (t*32+2*r+1)*stride) = Q6_V_hi_W(p);
    }
}
TL_DEVICE void gemm_unpack_ah_rows(void* dst, const void* src, int columns, int row, int rows) {
  for(int r=0;r<rows;r++) gemm_unpack_ah_row(static_cast<half_t*>(dst)+r*columns,src,columns,row+r);
}
TL_DEVICE void gemm_pack_ah_rows(void* dst_void, const void* src_void, int columns, int row, int rows) {
  hex_require(columns%32==0 && row%2==0 && rows%2==0 && is_aligned<128>(dst_void));
  auto dst=static_cast<HVX_Vector*>(dst_void);
  auto src=static_cast<const half_t*>(src_void);
  for(int r=0;r<rows;r+=2) for(int c=0;c<columns;c+=64) {
    if(c+32==columns) {
      alignas(128) half_t pair[64];
      __builtin_memcpy(pair,src+r*columns+c,64);
      __builtin_memcpy(pair+32,src+(r+1)*columns+c,64);
      int tile=((row+r)/32)*(columns/32)+c/32;
      dst[tile*16+((row+r)%32)/2]=Q6_Vh_vshuff_Vh(*(HVX_Vector*)pair);
      continue;
    }
    HVX_Vector a,b;
    __builtin_memcpy(&a,src+r*columns+c,128);
    __builtin_memcpy(&b,src+(r+1)*columns+c,128);
    HVX_VectorPair pair=Q6_W_vshuff_VVR(b,a,-2);
    int tile=((row+r)/32)*(columns/32)+c/32;
    int rp=((row+r)%32)/2;
    dst[tile*16+rp]=Q6_V_lo_W(pair);
    dst[(tile+1)*16+rp]=Q6_V_hi_W(pair);
  }
}
template <int Tiles>
TL_DEVICE void gemm_unpack_ah_strided(half_t *dst, const half_t *src_ah_vtcm, int row_stride) {
  static_assert(Tiles > 0, "positive tile count required");
  hex_require(is_aligned<128>(src_ah_vtcm) && row_stride >= 32);
  auto s = reinterpret_cast<const HVX_Vector *>(src_ah_vtcm);
  for (int t = 0; t < Tiles; ++t)
    for (int rp = 0; rp < 16; ++rp) {
      HVX_Vector v = Q6_Vh_vdeal_Vh(s[t * 16 + rp]);
      half_t *p = dst + size_t(t * 32 + 2 * rp) * row_stride;
      detail::gemm_store_row(p, v);
      detail::gemm_store_row(p + row_stride, Q6_V_vror_VR(v, 64));
    }
}
// Macro-emitter building blocks, not a matrix multiplication entry point.
// A hardware tile is 32x32 fp16 = 2048 bytes. RM here means a COMPACT
// 32x32 tile, not a view into an arbitrary parent tensor. Macro lowering owns
// parent strides, transpose, padding, and tile ordering (AH: m,k; WH: n,k).
// In particular WH input is logical W[k,n], NOT an untransposed W[n,k].
// All addresses are 128B aligned, all 2048 bytes readable/writable. Source
// and destination may coincide; otherwise they must not overlap.
// hexagon_rt.h: 32x32 AH and WH both use the row-pair zip16 permutation.
template <int Tiles = 1>
TL_DEVICE void gemm_pack_ah_tiles(void *dst, const void *rm) {
  static_assert(Tiles > 0, "positive tile count required");
  hex_require(is_aligned<128>(dst) && is_aligned<128>(rm));
  auto d = static_cast<HVX_Vector *>(dst);
  auto s = static_cast<const HVX_Vector *>(rm);
  for (int t = 0; t < Tiles; ++t)
    for (int rp = 0; rp < 16; ++rp)
      d[t * 16 + rp] = Q6_Vh_vshuff_Vh(s[t * 16 + rp]);
}
template <int Tiles = 1>
TL_DEVICE void gemm_pack_wh_tiles(void *dst, const void *rm_kn) {
  gemm_pack_ah_tiles<Tiles>(dst, rm_kn);
}
// acc readout is AH row-pair layout; vdeal restores two compact RM rows.
// No scalar VTCM loads/stores, including the 64-byte row halves.
template <int Tiles = 1>
TL_DEVICE void gemm_unpack_ah_tiles(void *rm, const void *ah) {
  static_assert(Tiles > 0, "positive tile count required");
  hex_require(is_aligned<128>(rm) && is_aligned<128>(ah));
  auto d = static_cast<HVX_Vector *>(rm);
  auto s = static_cast<const HVX_Vector *>(ah);
  for (int t = 0; t < Tiles; ++t)
    for (int rp = 0; rp < 16; ++rp)
      d[t * 16 + rp] = Q6_Vh_vdeal_Vh(s[t * 16 + rp]);
}

} // namespace tl
