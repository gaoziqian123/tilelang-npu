#pragma once
#include <stdint.h>
#include <stddef.h>

// Platform adapter contract, NOT a QuRT dynamic import. Backend supplies this
// symbol and validates its v79 descriptor limits before issuing native DMA.
// Returns only after all writes complete; zero success, negative failure.
extern "C" int tl_hex_dma_copy_2d_wait(void *dst, const void *src,
    uint32_t width, uint32_t rows, uint32_t src_stride, uint32_t dst_stride);
namespace tl {
inline int dma_copy_2d_wait(void *dst, uint64_t dst_offset, const void *src,
                          uint64_t src_offset, uint64_t width, uint64_t rows,
                          uint64_t src_stride, uint64_t dst_stride) {
  if (!dst || !src || !width || !rows || width>0xFFFFFF || rows>65535 ||
      src_stride>0xFFFFFF || dst_stride>0xFFFFFF ||
      src_stride<width || dst_stride<width ||
      ((uintptr_t(dst)+dst_offset)|(uintptr_t(src)+src_offset)|width|src_stride|dst_stride)&127u)
    return -81;
  if (src_offset>UINTPTR_MAX-uintptr_t(src) || dst_offset>UINTPTR_MAX-uintptr_t(dst) ||
      width>UINTPTR_MAX-uintptr_t(src)-src_offset || width>UINTPTR_MAX-uintptr_t(dst)-dst_offset ||
      rows-1>(UINTPTR_MAX-uintptr_t(src)-src_offset-width)/src_stride ||
      rows-1>(UINTPTR_MAX-uintptr_t(dst)-dst_offset-width)/dst_stride)
    return -82;
  int rc=tl_hex_dma_copy_2d_wait(static_cast<char*>(dst)+dst_offset,
      static_cast<const char*>(src)+src_offset,width,rows,src_stride,dst_stride);
  return rc;
}
} // namespace tl
