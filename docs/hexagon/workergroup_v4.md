# Stage-aware worker ABI v4 (asynchronous static loops)

Canonical declarations: `src/tl_templates/hexagon/workergroup_abi.h`.
Backend: `backend/npu/attn/skel/src/tl_workergroup_runtime.c`.

Compiler handoff: use `tl_wg_run_v4(caps, plan, physical_groups, stages,
stage_count, stage_edges, slots, callback, args, ddr, vtcm, sync)`.
`caps.abi_version=4`; `plan.base.abi_version=4`, descriptor_bytes=96;
base effects=EXACTLY_ONCE, round_count=0, flags=0. Extension fields are
stage_count, schedule=TL_WG_V4_ASYNC (2), reserved0=reserved1=0.
TL_WG_V4_SERIAL (1) is retained only as a globally serialized debug mode.
The separate explicit stage_count must equal the extension count.
V3 entrypoint accepts only ABI3/80B, and its physical DAG check stays intact.
No v3 worker fields or layout change; active stage is backend-private per lane.

Stages are `{stage_id, group_id, owner_local_order, reserved=0}`. Stage ID is
its array index, in global topological execution order. Each group's local
orders are dense starting at zero, in increasing stage-ID order. Every physical
group owns at least one stage. Edges are `{producer_stage, consumer_stage,
slot_desc, depth, ready_event_base, free_event_base, reserved0, reserved1}`.
All edges must go forward; depth is exactly two. Maximum stages/events are
64/64. Physical HMX group remains singleton worker 0; never duplicate it.

For each ordinal, each group executes its stages in order. All members call
`tl_wg_stage_enter(w,id)` before any event operations/body, then
`tl_wg_stage_exit(w,id)` after all event operations and completed engine work.
Enter validates the physical group membership, expected local stage and ordinal
collectively. In ASYNC it neither reads nor updates the global stage token.
Exit drains owning-thread work, checks incident event coverage collectively,
and advances only this group's next stage (or its next ordinal after its last
stage). Groups therefore run independently, blocked only on ready/free events
and their own collectives. SERIAL alone waits/transfers the global stage token.
Both modes permit HMX and HVX physical resources to be revisited without
physical cycles being mistaken for logical cycles. Neither stage entry nor exit
permits HMX preemption: enter makes no hardware completion promise, and the
platform's owning-thread drain hook must actually finish outstanding work.

## Restricted async progress proof / compiler obligations

This is a bounded-slot scheduler for rectangular static loops, not an arbitrary
DAG scheduler: every stage runs once per ordinal over the same invocation-wide
interval. The static union of data edges and consecutive same-owner resource
edges is acyclic because BOTH must strictly increase stage ID. The validator
checks data endpoints and dense ascending owner_local_order explicitly. Owners
must finish their entire local iteration before starting their next iteration;
QK cannot fill all ring slots before servicing PV on the same HMX owner.

For a blocked operation at ordinal q, a ready dependency points to an earlier
stage in the same ordinal; a free dependency points to its consumer at q-2
(or an initially free version). Local owner predecessors similarly precede the
operation in lexicographic (ordinal, stage ID) order. Thus the finite unrolled
dependency/resource union has no cycle, including when an edge's consumer is a
later stage on the same owner. This proves progress only for conforming callbacks
with finite engine work and the unconditional event effects below; arbitrary
user waits/branches, loop backedges, ragged iteration counts and recurrent
cross-stage writes are outside the contract and not made safe by this ABI.

For ordinal q, slot=q%2: producer waits free(base+slot,q/2), publishes
ready(base+slot,q/2+1); consumer waits ready(base+slot,q/2+1), publishes
free(base+slot,q/2+1). Each incident operation happens exactly once; stage exit
checks coverage. Wait-before-publication is checked for each edge. Compiler
must place producer writes after ALL free acquires (including fanout consumers),
ready publication after completed writes, and free publication after ALL reads
and engine users finish. Runtime cannot inspect memory accesses or prove these
body effects; publication is a completion promise, not an enqueue promise.
Release/acquire event ordering plus the group collectives exposes all members'
writes and protects WAR slot reuse, not just leader writes. Other
slot/generation/stage permissions reject and cancel all waiters. Error paths do
not manufacture successful ready/free publications: they broadcast cancellation,
drain and join before releasing resources. A stage enter is not a new physical
group arrival. After all ordinals,
call group_complete once per physical worker, including zero-trip execution.
Unknown modes, order conflicts and arithmetic overflow fail closed prelaunch.
Start failure/cancellation joins dispatched workers and drains before release.

## Native validation handoff

`/tmp/opencode/wg_v4_async_probe.c` runs the actual pthread runtime with 5 stages,
3 groups and 6 lanes, HMX and HVX owner revisits, separate depth-two edge buffers,
initial ordinal 7 and 0/1/32/97 trips. A mutex/condition rendezvous requires input
stage of ordinal 8 and compute stage of ordinal 7 to both arrive before either
can leave; this proves overlap without sleeps or timing inference. It is enabled
only for ASYNC. Both modes use an independent scalar oracle and canaries, plus
cancel, partial launch/reuse and malformed lifecycle/descriptor cases.

The earlier serial-only temporary probe treats mode 2 as unknown; that negative
case is obsolete because 2 is now the explicitly requested ASYNC mode. It was
not edited to manufacture a passing result; the new probe checks unknown mode 3
and covers both valid modes. Existing repository v3 tests remain unchanged.

This contract and host evidence do not establish FA numerical correctness,
device performance or the user's 10ms target. Device benchmarking is still due.
