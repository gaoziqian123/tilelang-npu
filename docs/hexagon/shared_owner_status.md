# Shared physical owners: v4 compiler integration candidate

## V4 continuation

The stage-aware ABI is now integrated into production callback/C emission.
Aliased plans use physical group descriptors, dense logical stage descriptors
with per-owner order, stage-indexed edges, `tl_wg_run_v4`, and ASYNC mode.
Every stage wraps all event operations and body in collective stage enter/exit.
Depth other than two, more than 64 stages/events, and resident extra-drain
events reject at compile time. The owner checks caps ABI4; sync bytes are supplied
by the caller and validated against the runtime-calculated requirement (there
is no compiler hardcoded private runtime-size estimate). Nonaliased plans stay
on the original v3 path.

`/tmp/opencode/shared_owner_native.py` emits unmodified portable scalar C from
stage TIR and links it to the real pthread runtime: six lanes, three two-lane
groups, five stages revisiting two owners, initial 7, trips 0/1/97, independent
oracle, poison and scratch canary. This validates collective stage/event
protocol, NOT HMX arithmetic. `/tmp/opencode/shared_owner_probe.py` additionally
lowers real QK/PV GEMM TileOps through production HMX codegen and emits
`/tmp/opencode/shared_owner_generated.c`. No native scalar substitute was added
to production HMX leaves. DSP object/assembly and standard FA remain unfinished.

The historical ABI3 integration boundary below is superseded by this section.

`T.pipeline_stage(..., physical_owner="resource", workers=N)` explicitly binds
logical stages to a physical resource. Equal owner labels require identical
engine and exact count. Unbound stages retain independent resources. Weighted
explicit bindings reject. Stage names never select implementation behavior.

The planner preserves `tl.workergroup_groups` as the logical stage table, so
existing slot edges, ownership scopes, callback event operations and merge keys
remain stage-indexed. Aliased plans additionally carry
`tl.workergroup_physical_groups` and a `physical_group` index per stage. Physical
worker accounting counts each resource once, including the singleton HMX owner.
Without aliases existing metadata and wrapper behavior remain unchanged.

Callback construction retains the lexical per-iteration stage sequence. A
shared HMX owner executes QK, waits for the middle stage, then executes PV before
advancing its iteration. It does not run QK and PV concurrently or fill the ring
with QK before servicing PV. Aliases require async scheduling; no team-round
fallback is introduced. Multiple writers and logical backedges still reject.
Private read/write FP32 state remains with its sole logical writer; sharing a
physical owner does NOT authorize cross-stage recurrent writes or last-read
reuse. General FA state recurrences and causal prefix ordinals remain unfinished.

## Historical ABI3 integration boundary (not executable FA)

ABI v3 has disjoint physical groups and group-indexed edges. Runtime validation
rejects self edges and cycles. QK(HMX)->middle(HVX)->PV(HMX) becomes a physical
cycle even though its logical stage graph is acyclic. Emitting duplicate HMX
groups violates disjointness; mapping logical edges to physical groups violates
cycle validation. Neither is safe. Therefore callback IR can be built, but the
production C wrapper and portable host emitter explicitly reject aliased plans.
No runtime files were changed and no unsafe v3 descriptor is emitted.

Next package needs an independently reviewed stage-aware ABI (stage-to-group
mapping plus stage-indexed event publisher/waiter validation and completion
accounting). Do not simply remove runtime cycle checks. Then enable wrapper
emission and execute the real native host runtime probe before device use.

Stage-aware ABI4 header/runtime now have a separate entrypoint with ASYNC static
loop scheduling and a SERIAL debug mode; see
`workergroup_v4.md` for the exact compiler handoff and restricted scheduling
contract. This does not remove ABI3's physical cycle check or make FA a verified
device kernel. Native runtime probe evidence is separate from the historical
Python simulation below; compiler integration is a parallel package.

Local implementation evidence: `/tmp/opencode/shared_owner_probe.py` constructs
real QK/PV TileOps, lowers to HMX/HVX callback IR, checks the launch rejection,
and executes the planned graph with Python condition-variable generation events
for zero/one/97 trips and nonzero initial ordinal. That simulation is NOT native
runtime or DSP validation. Callback IR is `/tmp/opencode/shared_owner_callback.tir`.
No new C/O/ASM or standard FA artifact exists in this package.
