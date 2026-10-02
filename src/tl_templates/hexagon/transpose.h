#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

namespace tl {
// Pure byte permutation. The allocation contract (including peripheral read
// padding) is checked before any access. No scalar DSP memory fallback.
template <int Bytes>
__attribute__((always_inline)) inline bool transpose_2d(void *dst, const void *src, size_t db, size_t sb,
                           int ds, int ss, int dr, int dc, int sr, int sc,
                           int rows, int cols) {
  static_assert(Bytes == 2 || Bytes == 4, "16/32-bit storage only");
  uintptr_t d = reinterpret_cast<uintptr_t>(dst), s = reinterpret_cast<uintptr_t>(src);
  if (!d || !s || (d | s | db | sb) % 128 || ds <= 0 || ss <= 0 ||
      dr < 0 || dc < 0 || sr < 0 || sc < 0 || rows <= 0 || cols <= 0 ||
      size_t(dc) + rows > size_t(ds) || size_t(sc) + cols > size_t(ss) ||
      d + db < d || s + sb < s || !(d + db <= s || s + sb <= d) ||
      (uint64_t(dr) + cols - 1) * ds + dc + rows > db / Bytes ||
      (uint64_t(sr) + rows - 1) * ss + sc + cols > sb / Bytes) return false;
  auto out = static_cast<unsigned char *>(dst);
  auto in = static_cast<const unsigned char *>(src);
  constexpr int N = 128 / Bytes;
  for (int r = 0; r < rows; r += N) {
    for (int c = 0; c < cols; c += N) {
      int h = rows - r < N ? rows - r : N;
      int w = cols - c < N ? cols - c : N;
#ifdef __hexagon__
      HVX_Vector v[N];
      for (int i = 0; i < N; ++i) {
        v[i] = Q6_V_vzero();
        if (i < h) {
          uintptr_t p = s + (size_t(sr + r + i) * ss + sc + c) * Bytes;
          int off = p & 127;
          auto base = reinterpret_cast<const HVX_Vector *>(p & ~uintptr_t(127));
          HVX_Vector lo = base[0];
          HVX_Vector hi = off + w * Bytes > 128 ? base[1] : lo;
          v[i] = Q6_V_valign_VVR(hi, lo, off);
        }
      }
      for (int step = 1; step < N; step *= 2) {
        for (int i = 0; i < N; ++i) if (!(i & step)) {
          HVX_VectorPair p = Q6_W_vshuff_VVR(v[i + step], v[i], -Bytes * step);
          v[i] = Q6_V_lo_W(p);
          v[i + step] = Q6_V_hi_W(p);
        }
      }
      for (int j = 0; j < w; ++j) {
        int rev = 0;
        for (int bit = 1; bit < N; bit *= 2) rev = (rev << 1) | !!(j & bit);
        uintptr_t p = d + (size_t(dr + c + j) * ds + dc + r) * Bytes;
        int off = p & 127, end = off + h * Bytes;
        auto base = reinterpret_cast<HVX_Vector *>(p & ~uintptr_t(127));
        HVX_Vector value = Q6_V_vlalign_VVR(v[rev], v[rev], off);
        auto upper = end >= 128 ? Q6_Q_not_Q(Q6_Q_vsetq_R(0)) : Q6_Q_vsetq_R(end);
        Q6_vmem_QRIV(Q6_Q_xor_QQ(upper, Q6_Q_vsetq_R(off)), base, value);
        if (end > 128) Q6_vmem_QRIV(Q6_Q_vsetq_R(end - 128), base + 1, value);
      }
#else
      // Host model of the same zip network, using integer storage exclusively.
      unsigned char v[N][128] = {};
      for (int i = 0; i < h; ++i)
        __builtin_memcpy(v[i], in + (size_t(sr+r+i)*ss+sc+c)*Bytes, w*Bytes);
      for (int step = 1; step < N; step *= 2)
        for (int i = 0; i < N; ++i) if (!(i & step)) {
          unsigned char pair[256];
          int chunk = Bytes * step;
          for (int k = 0; k < 128/chunk; ++k) {
            __builtin_memcpy(pair+2*k*chunk, v[i]+k*chunk, chunk);
            __builtin_memcpy(pair+(2*k+1)*chunk, v[i+step]+k*chunk, chunk);
          }
          __builtin_memcpy(v[i], pair, 128);
          __builtin_memcpy(v[i+step], pair+128, 128);
        }
      for (int j = 0; j < w; ++j) {
        int rev = 0;
        for (int bit = 1; bit < N; bit *= 2) rev = (rev << 1) | !!(j & bit);
        __builtin_memcpy(out+(size_t(dr+c+j)*ds+dc+r)*Bytes,v[rev],h*Bytes);
      }
#endif
    }
  }
  return true;
}
} // namespace tl
