#pragma once
#include "common.h"
#include "workergroup_abi.h"

namespace tl {
// Closed collective: no partial ticket escapes, and every scratch reader has
// finished before return. Caller supplies one exclusive DDR array per group;
// never use shared scratch across concurrent groups or overlap with live data.
// All active-group members must call in the same order inside an active stage.
template <unsigned Workers, bool Maximum>
TL_DEVICE float group_reduce_f32(const tl_wg_worker *w, unsigned group,
                                float value, float seed, float *scratch,
                                unsigned scratch_elements) {
  static_assert(Workers > 0 && Workers <= 64 && !(Workers & (Workers - 1)),
                "group reduction supports power-of-two counts from 1 to 64");
  hex_require(w && scratch && scratch_elements >= Workers);
  hex_require(w->group_id == group && w->local_size == Workers &&
              w->local_id < Workers);
  hex_require(is_aligned<alignof(float)>(scratch));
  const unsigned id = w->local_id;
  hex_require(__builtin_isfinite(value) && __builtin_isfinite(seed));
  scratch[id] = value;
  hex_require(tl_wg_group_barrier(w) == 0);
  // Ascending adjacent-pair tree. Only left endpoints update their partial;
  // barrier prevents the next level reading an unfinished partial.
  for (unsigned stride = 1; stride < Workers; stride *= 2) {
    if (id % (2 * stride) == 0) {
      float a = scratch[id], b = scratch[id + stride];
      scratch[id] = Maximum ? (a > b ? a : b) : a + b;
    }
    hex_require(tl_wg_group_barrier(w) == 0);
  }
  float result = scratch[0];
  // Seed post-combine exactly once per returned value, never per partial.
  result = Maximum ? (result > seed ? result : seed) : result + seed;
  hex_require(tl_wg_group_barrier(w) == 0); // scratch reuse only after all reads
  return result;
}
} // namespace tl
