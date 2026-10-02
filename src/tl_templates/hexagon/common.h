#pragma once

#include <stddef.h>
#include <stdint.h>

#define TL_DEVICE __attribute__((always_inline)) inline
#define TL_DEVICE_NOINLINE __attribute__((noinline)) inline

using half_t = _Float16;

// Software BF16 storage/conversion, round-to-nearest-even, preserving NaNs.
struct bfloat16_t {
  uint16_t bits;
  bfloat16_t() = default;
  TL_DEVICE bfloat16_t(float value) {
    uint32_t u;
    __builtin_memcpy(&u, &value, 4);
    bits = ((u & 0x7fffffffU) > 0x7f800000U)
               ? uint16_t((u >> 16) | 0x40)
               : uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
  }
  TL_DEVICE operator float() const {
    uint32_t u = uint32_t(bits) << 16;
    float value;
    __builtin_memcpy(&value, &u, 4);
    return value;
  }
};

struct dim3 {
  unsigned int x, y, z;
};

// Runtime ABI, deliberately not a header-local pool or a fixed NWORKERS.
// The skel adapter must bind these to the current QuRT worker context. The
// existing attnops_pool_* callback argument is a JOB index, not a worker id.
// A cooperative launch must run every participant concurrently: inserting a
// barrier into independently queued jobs can deadlock. barrier is reusable,
// collective over num_workers, and has acquire/release memory semantics.
extern "C" {
int tl_hex_worker_id();
int tl_hex_num_workers();
int tl_hex_job_id();
int tl_hex_num_jobs();
void tl_hex_barrier();
// Job-local HAP-reserved VTCM slab; adapter bounds-checks every slice.
// Stable for the ENTIRE kernel launch, isolated from concurrent jobs and
// collective scratch. Repeated requests for an offset return the same slice.
// alignment is a power of two; the adapter rejects out-of-bounds/misaligned
// requests rather than silently returning ordinary DDR memory.
void *tl_hex_vtcm_slice(size_t offset, size_t bytes, size_t alignment);
// Per-cooperative-group, 128B aligned shared scratch, valid until next
// collective. All participants receive the same address; no allocation races.
// Unlike tl_hex_vtcm_slice, this is temporary storage: callers must finish
// all reads (collective barrier) before the next scratch request. It must not
// alias any live kernel-lifetime VTCM slice.
void *tl_hex_collective_scratch(size_t bytes);
dim3 tl_hex_grid_dim();
dim3 tl_hex_block_idx();
// One process-wide runtime lock for template atomics, never the HMX lock.
void tl_hex_atomic_lock();
void tl_hex_atomic_unlock();
}

TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  uint16_t a, b;
  __builtin_memcpy(&a, &x, 2);
  __builtin_memcpy(&b, &y, 2);
  return (unsigned(b) << 16) | a;
}

namespace tl {
// Optional diagnostic sink; must be provided by the diagnostic build. It may
// record the id but cannot turn a failed guard into a successful continuation.
#ifdef TL_HEX_GUARD_DIAGNOSTICS
extern "C" void tl_hex_guard_failure(unsigned id);
#endif
TL_DEVICE void hex_require_id(bool condition, unsigned id) {
  if (!condition) {
#ifdef TL_HEX_GUARD_DIAGNOSTICS
    tl_hex_guard_failure(id);
#else
    (void)id;
#endif
    __builtin_trap();
  }
}
TL_DEVICE void hex_require(bool condition) {
  if (!condition) __builtin_trap();
}
template <size_t Alignment> TL_DEVICE size_t align_up(size_t value) {
  static_assert(Alignment && !(Alignment & (Alignment - 1)), "power-of-two alignment required");
  return (value + Alignment - 1) & ~(Alignment - 1);
}
template <size_t Alignment> TL_DEVICE bool is_aligned(const void *ptr) {
  static_assert(Alignment && !(Alignment & (Alignment - 1)), "power-of-two alignment required");
  return !(reinterpret_cast<uintptr_t>(ptr) & (Alignment - 1));
}
template <typename T> TL_DEVICE T max(T x, T y) { return x > y ? x : y; }
TL_DEVICE float max(float x, float y) { return __builtin_fmaxf(x, y); }
TL_DEVICE double max(double x, double y) { return __builtin_fmax(x, y); }
template <typename T> TL_DEVICE T min(T x, T y) { return x < y ? x : y; }
TL_DEVICE float min(float x, float y) { return __builtin_fminf(x, y); }
TL_DEVICE double min(double x, double y) { return __builtin_fmin(x, y); }
template <typename T> TL_DEVICE bool Any(T *a, int size) {
  for (int i = 0; i < size; ++i) if (a[i]) return true;
  return false;
}
template <typename T> TL_DEVICE bool All(T *a, int size) {
  for (int i = 0; i < size; ++i) if (!a[i]) return false;
  return true;
}
} // namespace tl
