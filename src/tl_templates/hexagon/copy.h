#pragma once
#include "hvx.h"

namespace tl {
// Synchronous fallback: all copies are complete on return. These are NOT
// inter-worker barriers; consumers on other workers need tl_hex_barrier().
TL_DEVICE void cp_async_commit() {}
template <int N = 0> TL_DEVICE void cp_async_wait() {
  static_assert(N >= 0, "negative async group count");
}
template <int N>
TL_DEVICE void cp_async_gs(void *lds_base_ptr, void const *global_base_ptr) {
  static_assert(N == 4 || N == 8 || N == 16 || (N > 0 && N % 128 == 0), "unsupported copy size");
  if constexpr (N % 128 == 0) {
    hex_require(is_aligned<128>(lds_base_ptr) && is_aligned<128>(global_base_ptr));
    auto dst = static_cast<HVX_Vector *>(lds_base_ptr);
    auto src = static_cast<const HVX_Vector *>(global_base_ptr);
    for (int i = 0; i < N / 128; ++i) dst[i] = src[i];
  } else {
    __builtin_memcpy(lds_base_ptr, global_base_ptr, N);
  }
}
template <int N>
TL_DEVICE void cp_async_gs_conditional(void *lds_base_ptr, void const *global_base_ptr, bool cond) {
  if (cond) cp_async_gs<N>(lds_base_ptr, global_base_ptr);
  else __builtin_memset(lds_base_ptr, 0, N);
}
template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr, void const *global_ptr) {
  cp_async_gs<N>(const_cast<void *>(smem_addr), global_ptr);
}
template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr, void const *global_ptr, bool cond) {
  cp_async_gs_conditional<N>(const_cast<void *>(smem_addr), global_ptr, cond);
}
} // namespace tl
