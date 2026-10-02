# Explicit worker pipelines — ABI v3

## v3 round-sync integration

Plan stays 80 bytes. Offset 76 is now explicitly `uint32_t round_count`, not
reserved; version 2 is rejected. `effects` is exactly one of EXACTLY_ONCE=1
or ROUND_SYNC=2, not a bitwise combination. Async EXACTLY_ONCE requires
round_count=0 and retains all existing edge publication checks. ROUND_SYNC
requires edge_count=event_count=0; slot descriptors still validate allocations.
Compiler must bound the total number of whole-team ticks to UINT32_MAX and
write it to round_count. Every lane must call the new
`int tl_wg_team_barrier(const tl_wg_worker *)` exactly that many times.
Group barriers remain group-local. Completion rejects missing/extra ticks.
No unused event checks are silently bypassed in async mode.

Backend blocking waits have a 60-second per-invocation soft deadline. A parked
waiter observing expiration sets TL_WG_TIMEOUT=-4 and wakes peers. QuRT uses
SDK qurt_timer_create targeting per-lane anysignals; host uses timed condvars.
The timers are deleted after join before their signals are destroyed. These
are new device imports requiring loader/device validation. A stuck hardware
leaf or callback that never reaches a wait is NOT forcibly interrupted; join
must not return/free active resources. No bounded hardware-recovery promise.

## 2026-09-29 runtime contract revision (primary-approved)

Version 2 appends `uint64_t initial_ordinal, iteration_count` and
`uint32_t effects, reserved` to `tl_wg_plan_desc` (80 bytes). Group/edge/slot,
caps and worker layouts and all public function signatures are unchanged.
Version 1 descriptors are rejected, never reinterpreted. `effects` must equal
`TL_WG_EFFECT_EXACTLY_ONCE`; reserved remains zero. This is a compiler proof
assertion: every edge producer publishes ready once and its consumer publishes
free once for every ordinal, with uniform unconditional participation of each
group member. Conditional/unbalanced stages must be rejected by the compiler;
the runtime cannot prove callback code from a descriptor. It validates the
assertion before launch and detects missing/duplicate publication at execution.

The invocation interval is `[initial_ordinal, initial_ordinal+iteration_count)`.
The sum must fit uint64; no ordinal or generation wraps. Zero trips are legal.
For depth D, slot s is initialized (both ready and free) to
`initial_ordinal / D + (s < initial_ordinal % D)`; this denotes a fully drained
prefix, not live payload. In particular nonzero starts do NOT initialize all
slots to the same epoch. With initial ordinal zero all events start at zero.
The final event bound uses the same expression at the interval end. Subsequent
outer tiles within one run never reset epochs. Wait remains `>=`; publication
must advance exactly one generation, within the interval bound.

TEST-CONTRACT-CHANGES: the earlier requested wrap-success test is replaced by
prelaunch overflow rejection, per primary decision. This is an explicit change
of semantics, not a weakened tolerance. Nonzero-start, zero-trip, short-trip and
all-slot reuse coverage remains required.

`tl_wg_run` is owner-side, blocking and not callable recursively from a worker.
It acquires a backend-private platform binding, checks real registered backing
bounds/ownership, initializes DDR sync, dispatches physical IDs 1..N-1 on a
persistent pool, and runs ID 0 synchronously on the owner. Worker callbacks use
the supplied `tl_wg_worker.local_id/local_size`; they never call nested legacy
launch. Each callback calls group_complete before returning. `tl_wg_join` is
owner-side only, not a worker collective; run invokes it before releasing any
resources. Group_complete and callback exit drain on the respective owner
thread; callback publish still requires its own scatter/DMA completion fence.

The current backend package implements common runtime and a real pthread host
adapter for synchronous CPU callbacks only. It is NOT a DSP/HMX execution
claim. QuRT persistent-pool, physical owner/VTCM reserve verification and real
hardware drain adapter are a subsequent package. No default adapter exists;
an unbound run fails closed. Private platform hooks are not portable ABI fields.
All barrier/event/lane state is DDR and charged to runtime-calculated sync_bytes.
Barriers use release/acquire arrivals and cancellation-polled epochs, not
uncancellable QuRT barrier waits. Platform join/drain must establish quiescence
even after partial launch or error; inability to establish it must block or
quarantine, never return resources as reusable.

## Implementation boundary

`T.pipeline_stage(name, engine=..., workers=... | weight=...)` is a C++
IR-builder-backed AttrStmt (`tl.pipeline_stage`, node: map of name, engine,
workers, weight; value: 1). Names are arbitrary labels, never operation selectors.
`T.Pipelined(..., num_stages=D)` containing these scopes owns
`tl.workergroup_depth=D` instead of `num_stages`: ordinary software-pipeline
versioning/reordering must not consume this loop a second time. Existing loops
without roles retain their current API and annotations, including CUDA.

Compiler planner/materializer and fail-closed whole-function validation are
developed independently. Do not launch generated device functions before that
package and the QuRT adapter land. The host runtime implementation described
above is executable but is not a device verification claim.

## Allocation and legality required of the planner

Use `with T.Kernel(..., threads=6)`, an outer `T.serial` loop over output
tiles, and inner `T.Pipelined(T.ceildiv(K,BK), num_stages=2)`. Inside use arbitrary
names with explicit engines: `hvx/workers=3`, `hmx/workers=1`,
`hvx/workers=2`. Counts must be positive integers, not booleans, and are exact.
There is no hard-coded two-role schema. Every group is nonempty, fixed,
contiguous and disjoint. At most one HMX role, exactly one worker at physical
ID 0 on the existing HMX-lock-owning RPC thread. Assign remaining groups in
lexical role order to contiguous physical ranges after that owner. Without
HMX, HVX allocation starts at zero. IDs passed to inline work partitioning
are **group-local** ID and size, not global team ID and size.

Optional `weight` is a positive integer ratio, mutually exclusive with
`workers` per role, supported for HVX only. Reserve exact-count roles first;
give each weighted role one worker. Distribute remaining workers by largest
remainder proportional to weights (ties lexical role order). Reject if fewer
workers remain than weighted roles. With no weighted roles the sum of exact
counts must equal the team size. `tl_wg_caps` is supplied explicitly: current
runtime max_workers=6 is a capability, not a compiler constant.

Reject scopes outside a pipeline, nested roles, nested worker pipelines,
duplicate names in one pipeline, role/operation mismatch (GEMM outside HMX,
HVX collective inside HMX), HMX owner multiplicity, oversubscribed teams,
nonuniform role control flow and unsupported cross-iteration dependencies.
Reject manual order/stage/sync/group scheduling mixed with worker roles.
Depth is any positive integer fitting event and byte budgets, including 1, 2,
3; not an enum restricted to 2. Reject arithmetic overflow, unknown resource
bounds, and nested resource double booking rather than claiming composition.

Infer dataflow from regions, not role names: A/B and materialized partial are
multiversioned; persistent fp32 `c` is not. HMX clears its hardware accumulator
per K tile, computes GEMM, and **reads out partial into memory**. HVX consumes
that partial and performs fp32 `c += cast(partial)`. Retaining the accumulated
answer invisibly in HMX is not equivalent and is forbidden. Store-owner HVX
workers partition `T.clear(c)` before first accumulate and final `T.copy(c,C)`
after last consumer drain; no full-array repetition per worker, no repeated
GEMM. Ordered updates to persistent c stay in K iteration order. K=0 still
initializes and stores c; it does not execute or publish any inner stage.

## Single ABI and dependency protocol

Canonical source: `src/tl_templates/hexagon/workergroup_abi.h`. Runtime must
include it (or mirror with static assertions for sizes AND field offsets),
use these exact symbols and validate version/descriptor_bytes/zero flags.
No existing kernel descriptor is reinterpreted. This is new opt-in ABI only.
Group/edge/slot arrays use plan counts; indices are zero-based. Check event
ranges are disjoint, group and slot indices in range, edge depth equals slot
depth, allocations aligned and nonoverlapping, and all budgets before launch.
`bytes_per_slot` is an aligned stride; allocation size is stride*depth.
Persistent c is outside recycled edge slots. Events and barriers consume
`sync_bytes` DDR, separately from data scratch. Runtime determines sync bytes
required for groups/events and rejects undersized supplied budgets.

Each directed producer→consumer edge has D ready and D free events. Fanout
uses separate edges and producer must acquire every consumer's free event
before reuse. A finite DAG of such edges is supported, not arbitrary cycles.
At ordinal zero all events begin at generation 0; wait means value >= requested generation.
Let q be the invocation-wide ordinal of inner iterations across successive
outer tiles: q = sum(previous inner trip counts) + current inner index.
Slot s=q mod D, epoch e=floor(q/D)+1. Producer waits free[s]>=e-1,
performs work, publishes ready[s]=e. Consumer waits ready[s]>=e, performs
work, publishes free[s]=e. q is never reset at outer-loop boundaries, even
for short trips (<D); zero trips advance it by zero. Reject overflow before
launch (q and generation uint64). Reset events only between joined invocations.

All members participate in each group call in the same dynamic sequence.
`group_publish` aggregates per-member arrival epochs via an acquire/release
group barrier; leader then release-publishes, followed by a group barrier.
Wait leader acquires event then broadcasts via an acquire/release barrier.
Thus **every producer write** happens-before **every consumer read**. A
leader-only release without aggregating worker writes is incorrect. Arrival
is internal to the collective publish/barrier primitive, not an unsynchronized
public counter. Each group barrier has monotonic invocation-wide sequence;
outer-loop transitions do not reset its sequence. Group complete is collective
and means no remaining memory/engine operations from any member. Outer tile
transition drains consumers before store/initialization or reuse of persistent c.

Async HMX completion means actual operation and readout completion, NOT queue
enqueue. Pack, scatter, asynchronous copies and all child work must finish
before publish. Free publication follows the last read/consumer completion.

## Failure and resource lifetime

`tl_wg_run` is blocking and owns context/event initialization and worker launch;
caller owns scratch, descriptor arrays and arguments until run returns.
All wait/barrier loops poll cancellation; first nonzero failure is retained.
Cancellation wakes peers; no participant may wait forever for an abandoned
generation. `tl_wg_join` waits for all workers and drains outstanding engines,
including partial-launch failure and failed invocations. A timeout is not proof
of quiescence: do not return scratch to caller/reuse/free while hardware or
workers may access it. Runtime must quarantine resources if quiescence cannot
be established. Successful run return implies join completed and all groups
completed; error return also requires quiescence. No fallback to old kernels.
