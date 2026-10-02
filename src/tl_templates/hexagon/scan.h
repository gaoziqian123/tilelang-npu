#pragma once
#include "reduce.h"

namespace tl {
struct ScanSumOp {
  template <typename T> TL_DEVICE hvx_vec<T> operator()(hvx_vec<T> x, hvx_vec<T> y) { return SumOp()(x, y); }
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) { return x + y; }
};
struct ScanMaxOp {
  template <typename T> TL_DEVICE hvx_vec<T> operator()(hvx_vec<T> x, hvx_vec<T> y) { return MaxOp()(x, y); }
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) { return tl::max(x, y); }
};
// Each line has one physical worker owner; its 32 logical lanes are an HVX
// register. Stage arbitrary strides/reverse order and tails into local DDR,
// then perform a masked Hillis-Steele prefix scan in registers. No unaligned
// HVX loads or out-of-bounds tail reads. TODO: direct contiguous vector IO
// and vector gather for strided VTCM inputs (currently scalar staging).
template <class Reducer, bool reverse, typename T, int SEG = 64>
static TL_DEVICE void InclusiveScanLine(const T *__restrict__ src,
                                      T *__restrict__ dst, int extent,
                                      int src_stride, int dst_stride) {
  static_assert(SEG > 0, "positive scan segment required");
  if (extent <= 0) return;
  if constexpr (__is_same(T, float) || __is_same(T, int)) {
    T carry{};
    bool have_carry = false;
    for (int base = 0; base < extent; base += 32) {
      const int active = (extent - base < 32) ? extent - base : 32;
      alignas(128) T lanes[32] = {};
      for (int lane = 0; lane < active; ++lane) {
        int index = reverse ? extent - 1 - base - lane : base + lane;
        lanes[lane] = src[index * src_stride];
      }
      auto v = hvx_load(lanes);
      for (int off = 1; off < 32; off <<= 1) {
        hvx_vec<T> previous{Q6_V_vror_VR(v.raw, 128 - off * 4)};
        auto combined = Reducer()(previous, v);
        v.raw = Q6_V_vmux_QVV(Q6_Q_vsetq_R(off * 4), v.raw, combined.raw);
      }
      if (have_carry) {
        int bits;
        __builtin_memcpy(&bits, &carry, 4);
        v = Reducer()(hvx_vec<T>{Q6_V_vsplat_R(bits)}, v);
      }
      hvx_store(lanes, v);
      carry = lanes[active - 1];
      have_carry = true;
      for (int lane = 0; lane < active; ++lane) {
        int index = reverse ? extent - 1 - base - lane : base + lane;
        dst[index * dst_stride] = lanes[lane];
      }
    }
    return;
  }
  // Types without a native 32-lane arithmetic implementation retain the
  // scalar path; do not reinterpret half/BF16 bit patterns as FP32 values.
  int i = reverse ? extent - 1 : 0;
  T value = src[i * src_stride];
  dst[i * dst_stride] = value;
  for (int n = 1; n < extent; ++n) {
    i += reverse ? -1 : 1;
    value = Reducer()(value, src[i * src_stride]);
    dst[i * dst_stride] = value;
  }
}
template <class Reducer, int threads, bool reverse = false>
struct InclusiveScan1D {
  static_assert(threads > 0, "positive thread extent required");
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst, int N) {
    if (tl_hex_worker_id() == 0) InclusiveScanLine<Reducer, reverse, T, SEG>(src, dst, N, 1, 1);
    tl_hex_barrier();
  }
  template <typename T>
  static TL_DEVICE void run_auto(const T *__restrict__ src, T *__restrict__ dst, int N) { run<T>(src, dst, N); }
};
template <class Reducer, int threads, int Axis = 0, bool reverse = false>
struct InclusiveScan2D {
  static_assert(threads > 0 && (Axis == 0 || Axis == 1), "invalid scan configuration");
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                           int H, int W, int src_stride, int dst_stride) {
    const int count = tl_hex_num_workers();
    hex_require(count > 0);
    for (int line = tl_hex_worker_id(); line < (Axis == 1 ? H : W); line += count) {
      if constexpr (Axis == 1)
        InclusiveScanLine<Reducer, reverse, T, SEG>(src + line * src_stride, dst + line * dst_stride, W, 1, 1);
      else
        InclusiveScanLine<Reducer, reverse, T, SEG>(src + line, dst + line, H, src_stride, dst_stride);
    }
    tl_hex_barrier();
  }
  template <typename T>
  static TL_DEVICE void run_auto(const T *__restrict__ src, T *__restrict__ dst,
                                int H, int W, int src_stride, int dst_stride) {
    run<T>(src, dst, H, W, src_stride, dst_stride);
  }
};
template <int threads, bool reverse = false> struct CumSum1D {
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst, int N) {
    InclusiveScan1D<ScanSumOp, threads, reverse>::template run<T, SEG>(src, dst, N);
  }
};
template <int threads, int Axis = 0, bool reverse = false> struct CumSum2D {
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                           int H, int W, int src_stride, int dst_stride) {
    InclusiveScan2D<ScanSumOp, threads, Axis, reverse>::template run<T, SEG>(src, dst, H, W, src_stride, dst_stride);
  }
};
template <int threads, bool reverse = false> struct CumMax1D {
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst, int N) {
    InclusiveScan1D<ScanMaxOp, threads, reverse>::template run<T, SEG>(src, dst, N);
  }
};
template <int threads, int Axis = 0, bool reverse = false> struct CumMax2D {
  template <typename T, int SEG = 64>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                           int H, int W, int src_stride, int dst_stride) {
    InclusiveScan2D<ScanMaxOp, threads, Axis, reverse>::template run<T, SEG>(src, dst, H, W, src_stride, dst_stride);
  }
};
} // namespace tl
