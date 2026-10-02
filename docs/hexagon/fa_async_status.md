# Async attention compiler status

This is unfinished compiler work, not an accepted FA implementation.

Target: B=1, HQ=16, HKV=4, S=1024, D=256, causal, scale=1/16;
initial query/key tiles 128/256. Keep online maximum, denominator, output
accumulator and exp in FP32. Half HMX materialization is not permission to
replace the recurrence with half arithmetic.

The existing `fa_tile_schedule.py`, `fa_persistent.py`, and `fa.py` express
attention with ordinary DSL operations, not a retired whole-attention helper.
The first two currently apply 0.0625 after score readout; introducing prescaled
half Q would change a rounding boundary and must not be described as equivalent
without numerical validation.

## Implemented prerequisite

PlanWorkerPipeline accepts transform and reduce TileOps using their existing
access-region contracts. It accepts exp/exp2 in HVX roles and recursively
visits their operands, preserving input dependencies. Other calls remain
fail closed. No ABI or callback scheduling changes were made.

Host planner probe: `/tmp/opencode/fa_stage_effects_probe.py`. It checks a
transform producer followed by an exp/reduce consumer and requires the
producer-to-consumer slot edge; it is not a device numerical test.

## Required next implementation (not available yet)

1. Separate logical stage IDs from physical group IDs throughout ownership,
   physical allocation and callback construction. QK and PV must map to one
   singleton HMX group. Existing roles are still one-to-one with groups.
2. Build a stage dependency DAG, retaining producer and last-consumer stage
   identities for slot readiness/free events. Do not remove the current
   backedge rejection: a group revisit is not a loop-carried data recurrence.
3. Emit bounded per-job phase visitation for each physical group, so the HMX
   owner reaches PV before issuing an unbounded stream of QK jobs. Validate
   finite-slot progress across *all* groups, not merely stage DAG acyclicity.
   QK and PV cannot execute concurrently on the single owner.
4. Represent causal valid iteration counts per independent query job with
   checked ordinal prefix sums. Do not replace causal prefixes with full
   future-key computation. FP32 state belongs exclusively to its query job.
5. Connect immutable packed K/V lifetime, Q/K/V readonly and O write manifest,
   transform legality and output DMA. Account for every physical allocation,
   slot and fence against the actual 4MiB grant, not just tensor byte counts.
6. Generate standard-shape C/object/ASM, re-emit the standard GEMM baseline
   (expected C SHA256 3370fafc941321b9c0ca7d220178e2d200631dd569b0b52399e85900a83b38ea),
   then hand off to independent verification and device numerical gates.

Neither same-owner planner support nor a lowerable async FA milestone is
claimed by this prerequisite change. No backend, adapter or phone changes.
