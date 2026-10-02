#pragma once
#include "hvx.h"

namespace tl {
struct SumOp {
  TL_DEVICE hvx_vec<float> operator()(hvx_vec<float> x, hvx_vec<float> y) { return hvx_add(x, y); }
  TL_DEVICE hvx_vec<int> operator()(hvx_vec<int> x, hvx_vec<int> y) { return {Q6_Vw_vadd_VwVw(x.raw, y.raw)}; }
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) { return x + y; }
};
struct MaxOp {
  TL_DEVICE hvx_vec<float> operator()(hvx_vec<float> x, hvx_vec<float> y) { return hvx_max(x, y); }
  TL_DEVICE hvx_vec<int> operator()(hvx_vec<int> x, hvx_vec<int> y) { return {Q6_Vw_vmax_VwVw(x.raw, y.raw)}; }
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) { return tl::max(x, y); }
};
struct MinOp {
  TL_DEVICE hvx_vec<float> operator()(hvx_vec<float> x, hvx_vec<float> y) { return {Q6_Vsf_vmin_VsfVsf(x.raw, y.raw)}; }
  TL_DEVICE hvx_vec<int> operator()(hvx_vec<int> x, hvx_vec<int> y) { return {Q6_Vw_vmin_VwVw(x.raw, y.raw)}; }
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) { return tl::min(x, y); }
};
struct BitAndOp {
  TL_DEVICE hvx_vec<int> operator()(hvx_vec<int> x, hvx_vec<int> y) { return {Q6_V_vand_VV(x.raw, y.raw)}; }
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) { return x & y; }
};
struct BitOrOp {
  TL_DEVICE hvx_vec<int> operator()(hvx_vec<int> x, hvx_vec<int> y) { return {Q6_V_vor_VV(x.raw, y.raw)}; }
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) { return x | y; }
};
struct BitXorOp {
  TL_DEVICE hvx_vec<int> operator()(hvx_vec<int> x, hvx_vec<int> y) { return {Q6_V_vxor_VV(x.raw, y.raw)}; }
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) { return x ^ y; }
};

template <class Reducer, int threads, int scale, int thread_offset = 0,
          int batch_size = 1, int workspace_stride = 0>
struct AllReduce {
  static_assert(threads > 0 && scale > 0 && threads % scale == 0, "invalid reduction extent");
  static_assert(((threads / scale) & (threads / scale - 1)) == 0, "reduction width must be power of two");
  static_assert(batch_size > 0, "invalid batch size");
  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    if constexpr (threads == scale) return x;
    const int id = tl_hex_worker_id();
    hex_require(id >= thread_offset && id < thread_offset + threads);
    hex_require(tl_hex_num_workers() >= thread_offset + threads);
    if (!red_buf) red_buf = static_cast<T *>(tl_hex_collective_scratch(sizeof(T) * tl_hex_num_workers()));
    hex_require(red_buf != nullptr);
    // Caller buffer uses HIP scalar indexing (absolute worker id).
    for (int offset = threads / 2; offset >= scale; offset >>= 1) {
      red_buf[id] = x;
      tl_hex_barrier();
      x = Reducer()(x, red_buf[thread_offset + ((id - thread_offset) ^ offset)]);
      tl_hex_barrier(); // all readers finish before the next overwrite
    }
    return x;
  }
  template <typename T> static TL_DEVICE void run_batch(T *x, T *red_buf = nullptr) {
    // Serial batch reuse needs only one scalar collective workspace. Honor
    // the HIP batch buffer's relative indexing and stride when supplied.
    static_assert(workspace_stride == 0 || workspace_stride >= threads, "overlapping batch workspace");
    if constexpr (threads == scale) return;
    int id = tl_hex_worker_id() - thread_offset;
    hex_require(id >= 0 && id < threads);
    hex_require(tl_hex_num_workers() >= thread_offset + threads);
    if (!red_buf) red_buf = static_cast<T *>(tl_hex_collective_scratch(sizeof(T) * (threads + (batch_size - 1) * workspace_stride)));
    hex_require(red_buf != nullptr);
    for (int b = 0; b < batch_size; ++b) {
      T *buf = red_buf + b * workspace_stride;
      for (int offset = threads / 2; offset >= scale; offset >>= 1) {
        buf[id] = x[b];
        tl_hex_barrier();
        x[b] = Reducer()(x[b], buf[id ^ offset]);
        tl_hex_barrier();
      }
    }
  }
};

// One register holds 32 logical lanes on one worker. Return the reduced
// value in every lane, exactly as a logical-warp all-reduce does. This path
// has no worker-count constraint and no cross-worker synchronization.
template <typename T, typename ReduceOp>
TL_DEVICE hvx_vec<T> warp_reduce(hvx_vec<T> value, ReduceOp op) {
  static_assert(sizeof(T) == 4, "logical warp carrier needs 32 four-byte lanes");
  for (int bytes = 64; bytes >= 4; bytes >>= 1)
    value = op(value, hvx_vec<T>{Q6_V_vror_VR(value.raw, bytes)});
  return value;
}
// CUDA/HIP logical warp is 32 scalar lanes, NOT one Hexagon worker. The
// runtime must launch a cooperative group whose size is a multiple of 32.
// Never silently reinterpret a scalar as 32 identical HVX lanes.
template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op) {
  int id = tl_hex_worker_id(), count = tl_hex_num_workers();
  hex_require(count > 0 && count % 32 == 0 && id >= 0 && id < count);
  T *buf = static_cast<T *>(tl_hex_collective_scratch(sizeof(T) * count));
  hex_require(buf != nullptr);
  for (int offset = 16; offset; offset >>= 1) {
    buf[id] = value;
    tl_hex_barrier();
    value = op(value, buf[id ^ offset]);
    tl_hex_barrier();
  }
  return value;
}
template <typename T> TL_DEVICE T warp_reduce_sum(T value) { return warp_reduce(value, SumOp()); }
template <typename T> TL_DEVICE T warp_reduce_max(T value) { return warp_reduce(value, MaxOp()); }
template <typename T> TL_DEVICE T warp_reduce_min(T value) { return warp_reduce(value, MinOp()); }
template <typename T> TL_DEVICE T warp_reduce_bitand(T value) { return warp_reduce(value, BitAndOp()); }
template <typename T> TL_DEVICE T warp_reduce_bitor(T value) { return warp_reduce(value, BitOrOp()); }
} // namespace tl
