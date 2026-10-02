#ifndef TILELANG_HEXAGON_WORKERGROUP_ABI_H_
#define TILELANG_HEXAGON_WORKERGROUP_ABI_H_

#include <stdint.h>
#include <stddef.h>

/* Single source of truth. New opt-in ABI; never cast an old kernel descriptor.
 * All reserved fields must be zero. All offsets are bytes, not pointers.
 * Descriptor arrays are immutable for the entire invocation. */
#define TL_WG_ABI_VERSION 3u
#define TL_WG_EFFECT_EXACTLY_ONCE 1u
#define TL_WG_EFFECT_ROUND_SYNC 2u
#define TL_WG_TIMEOUT (-4)
#define TL_WG_ENGINE_HVX 1u
#define TL_WG_ENGINE_HMX 2u
#define TL_WG_OK 0
#define TL_WG_INVALID (-1)
#define TL_WG_CANCELLED (-2)
#define TL_WG_RESOURCE (-3)

typedef struct tl_wg_group_desc {
  uint32_t engine, first_worker, worker_count, reserved;
} tl_wg_group_desc;

/* One single-producer/single-consumer edge per independently recycled buffer.
 * Fanout requires one edge per consumer; producer acquires ALL free events.
 * slot_desc indexes a separately budgeted multiversioned allocation. */
typedef struct tl_wg_edge_desc {
  uint32_t producer_group, consumer_group, slot_desc, depth;
  uint32_t ready_event_base, free_event_base, reserved0, reserved1;
} tl_wg_edge_desc;

typedef struct tl_wg_slot_desc {
  uint64_t byte_offset, bytes_per_slot;
  uint32_t depth, alignment, memory_kind, reserved;
} tl_wg_slot_desc;
#define TL_WG_MEMORY_DDR 1u
#define TL_WG_MEMORY_VTCM 2u

typedef struct tl_wg_plan_desc {
  uint32_t abi_version, descriptor_bytes, team_size, group_count;
  uint32_t edge_count, slot_count, event_count, flags;
  uint64_t ddr_scratch_bytes, vtcm_scratch_bytes, sync_bytes;
  /* Compiler proves uniform unconditional per-edge producer/consumer effects.
   * Each edge is published/consumed exactly once for every ordinal in this
   * half-open interval. No wrap; no reset across outer tiles. */
  uint64_t initial_ordinal, iteration_count;
  uint32_t effects, round_count;
} tl_wg_plan_desc;

/* V4 is a separate entrypoint, never read past a v3 descriptor. Stage IDs are
 * dense topological/serial order. Resource groups remain disjoint physical
 * groups. Rectangular static loops use depth-two slots; ASYNC advances each
 * physical owner independently, SERIAL retains a global debug stage token. */
#define TL_WG_ABI_VERSION_V4 4u
#define TL_WG_V4_SERIAL 1u
#define TL_WG_V4_ASYNC 2u
#define TL_WG_V4_MAX_STAGES 64u
#define TL_WG_V4_MAX_EVENTS 64u
typedef struct tl_wg_stage_desc {
  uint32_t stage_id, group_id, owner_local_order, reserved;
} tl_wg_stage_desc;
typedef struct tl_wg_edge_v4_desc {
  uint32_t producer_stage, consumer_stage, slot_desc, depth;
  uint32_t ready_event_base, free_event_base, reserved0, reserved1;
} tl_wg_edge_v4_desc;
typedef struct tl_wg_plan_v4_desc {
  tl_wg_plan_desc base;
  uint32_t stage_count, schedule, reserved0, reserved1;
} tl_wg_plan_v4_desc;

/* Supplied by runtime, not baked into the compiler (current device team cap 6).
 * Runtime must reject a plan exceeding ANY limit before starting workers. */
typedef struct tl_wg_caps {
  uint32_t abi_version, max_workers, max_groups, max_events;
  uint64_t max_ddr_bytes, max_vtcm_bytes, max_sync_bytes;
} tl_wg_caps;

typedef struct tl_wg_context tl_wg_context;
typedef struct tl_wg_worker {
  tl_wg_context *context;
  uint32_t physical_id, group_id, local_id, local_size;
} tl_wg_worker;

/* Every group member calls these in identical order. Successful wait is an
 * acquire for ALL members; publish aggregates ALL members' completed writes,
 * then the leader releases the generation, then completes a group barrier.
 * Events and group barriers reside in DDR, never VTCM. */
#ifdef __cplusplus
extern "C" {
#endif
int tl_wg_group_wait(const tl_wg_worker *, uint32_t event, uint64_t generation);
int tl_wg_group_publish(const tl_wg_worker *, uint32_t event, uint64_t generation);
int tl_wg_group_barrier(const tl_wg_worker *);
int tl_wg_team_barrier(const tl_wg_worker *);
/* Uniform callback-wide boundary. Backend must honor cancellation/deadline. */
int tl_wg_team_barrier(const tl_wg_worker *);
int tl_wg_error_poll(tl_wg_context *);
void tl_wg_cancel(tl_wg_context *, int error);
/* Completion means operations and stores have finished, not queue enqueue.
 * Runtime joins every worker and drains outstanding engines on success/error.
 * Even cancellation MUST NOT free scratch until join returns quiescent. */
int tl_wg_group_complete(const tl_wg_worker *);
int tl_wg_join(tl_wg_context *);
typedef int (*tl_wg_worker_entry)(const tl_wg_worker *, void *arguments);
int tl_wg_run(const tl_wg_caps *, const tl_wg_plan_desc *,
              const tl_wg_group_desc *, const tl_wg_edge_desc *,
              const tl_wg_slot_desc *, tl_wg_worker_entry, void *arguments,
               void *ddr_scratch, void *vtcm_scratch, void *sync_ddr);
int tl_wg_run_v4(const tl_wg_caps *, const tl_wg_plan_v4_desc *,
                 const tl_wg_group_desc *, const tl_wg_stage_desc *,
                 uint32_t stage_count, const tl_wg_edge_v4_desc *,
                 const tl_wg_slot_desc *, tl_wg_worker_entry, void *arguments,
                 void *ddr_scratch, void *vtcm_scratch, void *sync_ddr);
int tl_wg_stage_enter(const tl_wg_worker *, uint32_t stage_id);
int tl_wg_stage_exit(const tl_wg_worker *, uint32_t stage_id);
#ifdef __cplusplus
}
#endif

static inline uint32_t tl_wg_local_id(uint32_t physical_id,
                                       const tl_wg_group_desc *group) {
  return physical_id - group->first_worker;
}
static inline uint32_t tl_wg_local_size(const tl_wg_group_desc *group) {
  return group->worker_count;
}

#if defined(__cplusplus)
#define TL_WG_ASSERT(c) static_assert(c, #c)
#else
#define TL_WG_ASSERT(c) _Static_assert(c, #c)
#endif
TL_WG_ASSERT(sizeof(tl_wg_group_desc) == 16);
TL_WG_ASSERT(sizeof(tl_wg_edge_desc) == 32);
TL_WG_ASSERT(sizeof(tl_wg_slot_desc) == 32);
TL_WG_ASSERT(sizeof(tl_wg_plan_desc) == 80);
TL_WG_ASSERT(sizeof(tl_wg_plan_v4_desc) == 96);
TL_WG_ASSERT(sizeof(tl_wg_stage_desc) == 16);
TL_WG_ASSERT(sizeof(tl_wg_edge_v4_desc) == 32);
TL_WG_ASSERT(offsetof(tl_wg_plan_desc, initial_ordinal) == 56);
TL_WG_ASSERT(offsetof(tl_wg_plan_desc, iteration_count) == 64);
TL_WG_ASSERT(offsetof(tl_wg_plan_desc, effects) == 72);
TL_WG_ASSERT(offsetof(tl_wg_plan_desc, round_count) == 76);
TL_WG_ASSERT(sizeof(tl_wg_caps) == 40);
#undef TL_WG_ASSERT
#endif
