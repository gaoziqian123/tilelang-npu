#pragma once
#include "common.h"

// Runtime lock serializes all participants, including the owner thread.
// Acquire/release lock operations provide stronger-than-requested ordering.
// No HMX lock/barrier here: divergent atomic call counts must be legal.
template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 *ref, T2 val, int memory_order = 0) {
  tl::hex_require(memory_order >= 0 && memory_order <= 5);
  tl_hex_atomic_lock();
  T1 old = *ref;
  *ref = T1(old + static_cast<T1>(val));
  tl_hex_atomic_unlock();
  return old;
}
template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 val, int memory_order = 0) {
  (void)AtomicAddRet(address, val, memory_order);
}
template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 &address, T2 val, int memory_order = 0) {
  AtomicAdd(&address, val, memory_order);
}
template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 *address, T2 val, int memory_order = 0) {
  tl::hex_require(memory_order >= 0 && memory_order <= 5);
  tl_hex_atomic_lock();
  *address = tl::max(*address, static_cast<T1>(val));
  tl_hex_atomic_unlock();
}
template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 &address, T2 val, int memory_order = 0) {
  AtomicMax(&address, val, memory_order);
}
template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 *address, T2 val, int memory_order = 0) {
  tl::hex_require(memory_order >= 0 && memory_order <= 5);
  tl_hex_atomic_lock();
  *address = tl::min(*address, static_cast<T1>(val));
  tl_hex_atomic_unlock();
}
template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 &address, T2 val, int memory_order = 0) {
  AtomicMin(&address, val, memory_order);
}
