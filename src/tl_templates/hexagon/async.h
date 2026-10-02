#pragma once
#include "common.h"
#include <new>
extern "C" void *tl_hex_engine_reserve(int, unsigned);
extern "C" int tl_hex_engine_commit(int, void (*)(void *));
extern "C" int tl_hex_engine_wait_slot(int);
namespace tl {
// Bounded value capture lives in runtime-owned DDR until explicit wait. The
// compiler emits the body from IR; runtime knows no matrix or attention math.
template<int Slot, class F> TL_DEVICE void engine_submit(F f) {
  static_assert(sizeof(F)<=512,"engine closure exceeds descriptor payload");
  void *p=tl_hex_engine_reserve(Slot,sizeof(F));
  hex_require(p!=nullptr);
  new(p) F(f);
  hex_require(tl_hex_engine_commit(Slot,[](void *a){ (*static_cast<F*>(a))(); })==0);
}
TL_DEVICE void engine_wait(int slot) {
  if(tl_hex_worker_id()==0) hex_require(tl_hex_engine_wait_slot(slot)==0);
  tl_hex_barrier();
}
}
