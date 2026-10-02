# Ordered accumulator promotion (experimental, default off)

`tilelang.transform.PromoteOrderedAccumulator()` is a common C++ TIR pass,
registered as `tl.transform.PromoteOrderedAccumulator`. Explicit calls run the
pass; automatic Hexagon pipeline insertion requires
`tl.enable_ordered_accumulator_promotion=True`. Placement is after LowerTileOp
and LegalizeVectorizedLoop, before allocation planning, flattening and vector
materialization. Other targets can invoke the pass explicitly.

## Accepted contract

Version one accepts a bounded perfect nest `serial reduction -> vector lane`
or `serial reduction -> serial vector group -> vector lane`, ending in exactly
one unpredicated `dst[index] = dst[index] + contribution` float32 store.
All loop minima are zero and constant extents are in [1,4096]. The group must
carry `tl.vectorized_group=1`, emitted from an original vectorized loop by
LegalizeVectorizedLoop **only while the opt-in config is enabled**. It is
provenance, not permission to reorder arbitrary serial loops.

The analyzer must prove every load/store in bounds, the destination independent
of the reduction variable, and compact flattened destination indices equal to
an invariant base plus `group * lanes + lane`. Explicit strides are accepted
only when provably compact. Unknown/noncompact strides, nonzero offsets and
multiple Buffer views of the same data Var are rejected. This is deliberately
not a general affine layout transformation.

Contribution expressions accept constants, scalar variables, ordinary loads,
and Add/Sub/Mul only. They cannot read **any** region of the destination data
Var. Distinct lexical local allocations prove disjointness. External parameters
require an explicit, truthful `tir.noalias=True` caller contract; the pass
never adds it. A buffer name is not an alias proof. Multiple views of one data
Var remain rejected even with noalias. Unknown external aliases are a no-op.

The original version conservatively rejected any calls (including
address_of, opaque/atomic/barrier calls, even otherwise pure calls), naked
handle expressions, match-buffer aliases, non-local allocations, or attributes
other than provably single-thread thread_extent. Parallel/thread-bound loops
require extent provably one. Thus an address escape before/after a candidate,
volatile/async scopes and multiworker visibility do not slip through a local
nest-only check. Normal external memory must be exclusively owned for the call;
noalias is not a synchronization or volatile-memory contract.

### Candidate-region revision

Calls inside a candidate remain forbidden. Outside it, registered pure scalar
operations, storage synchronization, and audited backend operations are boundaries;
seed/final stores stay immediately around the candidate. Unknown calls (even
without explicit pointer arguments), naked handles, pointer aliases, volatile and
async scopes still reject the function. No unknown call is silently filtered.

Two generic Op attributes encode implementation facts, not user promises:
`tl.region_synchronous` means all accesses finish on return, no retained pointers,
callbacks or background consumers; `tl.region_bounded_effects` means effects,
including outstanding effects, are confined to explicit address_of operands.
For the latter every candidate read/write buffer must be proven disjoint from
every exposed operand across the **whole function**. HMX/scatter are deliberately
only bounded, not claimed synchronous. Transpose/require/profile helpers are
synchronous. Backend registrations reference the concrete helper implementations.
These attributes do not permit arithmetic calls inside the reduction.

Nonlocal allocations elsewhere no longer block a private/external candidate;
the candidate destination itself cannot be nonlocal. Generated accumulators
carry `tl.ordered_accumulator` block provenance to make repeated runs idempotent.

## Order and resources

Each independent group gets a local float32 buffer: seed once, update in the
original reduction order, write once. No reduction reassociation, changed
dtype, accumulator initialization to zero, or new fused arithmetic is inserted.
All original per-element Add/Mul nodes remain separate. Backend fast-math or
contraction options are outside this pass's contract; strict host verification
uses `-ffp-contract=off`. DSP numerical validation is a separate device gate.

The native lane width comes from the existing target-dependent vector legalizer,
not GDN dimensions. `tl.ordered_accumulator_budget_bytes` defaults to 128 and
limits one promoted accumulator; negative/zero/too-small budgets produce no
promotion. It does not promise register residency, bound whole-function stack
usage, or override the target's vector width. Allocation remains the downstream
planner's job. Only actual object/assembly inspection can establish spills.

Unsupported candidates retain their original body. No kernel names, shapes,
launch changes or generated-source edits are used.

## Current integration boundary

GDN main_stage now promotes external S/KF/RF with the above boundary analysis.
`examples/hexagon/gdn/layout.py` is shared by emitter and registry build: it checks
all slab intervals for alignment, positive lengths, bounds and pairwise absence
of overlap, and every PrimFunc parameter for matching dtype/compact byte extent
before adding noalias. Same-slab disjoint regions are legal; different variable
names alone remain insufficient. ABI, workers and DSL reduction are unchanged.
The build CLI adds `--ordered-accumulator` (default off), records it in layout.json
and seals the shared layout source and metadata. Device verification is separate.

TEST-CONTRACT-CHANGES (region revision): the prior GDN structural-no-op assertion
is replaced by a positive promotion assertion and generated-C reduction check
for absence of S accesses and preservation of Add/Mul. The requirement changed
from unsupported whole-function calls to checked region boundaries. All original
negative/bitwise/reference/default-off checks remain; region, escape, async
operand alias, idempotence and slab-overlap rejection coverage is added.

## TEST-CONTRACT-CHANGES

The initial WIP positives lacked the external noalias precondition required by
the fail-closed alias contract. `make()` remains unchanged: its original unknown
alias case now has an explicit structural no-op assertion. Positive callers add
truthful `tir.noalias=True` (the harness allocates distinct arrays), while
`test_ordered_accumulator_contract.py` separately covers independent local
allocations without requiring external noalias. Other negatives now start from
a legal noalias fixture, so alias rejection cannot mask their specific defect;
all negative cases remain, with structural equality rather than name-only checks.
The explicit-pass helper enables vector-group provenance during legalization.

Host tests clone away only the Hexagon target attribute before C lowering and
assert nonempty generated source and its real ABI. External fixtures use
`(dst,lhs,rhs)`; internal fixtures use `(a,b,out)`. Promotion is asserted before
execution. Baseline/promoted outputs are compared bitwise, including NaN bits;
an independent ordered, separately rounded NumPy fp32 multiply/add reference
checks every non-NaN bit (including signed zero/Inf) and the NaN mask. Subnormal
inputs are included. No tolerance or mathematical check was relaxed.

Real Hexagon C, object and assembly generation remains covered. The synthetic
assembly check requires one final vector store, seven ordered adds and no vector
stack accesses; it is not a claim of strict DSP IEEE behavior. The original GDN
no-op check is superseded by the region revision above. Default pipeline IR at the insertion point
must equal explicitly disabled IR on the same legalizer input, with no promotion
pass invoked; synthetic source must also match. Independent GDN lowerings
rebuild layout-map Buffer keys and can rename identical scale allocations,
so cross-lowering object identity/source text is not a determinism gate.
GDN device performance and mixed-precision correctness remain separate gates.

The old temporary `verify_ordered.py` HOST_EXACT_BITS_PASS line used the wrong
host argument order and remains invalid evidence. The corrected independent
`ordered_exact_abi.py` call/reference is now represented in the pytest harness.
