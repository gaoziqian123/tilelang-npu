# Resident async worker schedule — implementation candidate, not verified release

## CUDA comparison (sources read before implementation)

`src/transform/inject_pipeline.cc:ComputeBufferVersions` derives versions from
definition/last-use stages and overlapping reader/writer regions; external
definitions are not silently versioned. `src/cuda/transform/materialize_ws_schedule.cc`
binds accesses to acquire/wait phases and linearizes the original rectangular
outer-loop chain (`LinearPhase`), subtracting nonzero minima. Commit/release
close the ownership span. Guarded or nonrectangular schedules need separate
counters rather than resetting the phase at every inner loop.

Hexagon borrows those ownership principles, NOT CUDA hardware semantics:
`cp.async`/TMA/tcgen deferred arrivals and proxy fences do not prove HMX completion.
The Hexagon backend must guarantee that the synchronous HMX leaf has finished
input reads and partial stores before the callback publishes free/ready.
A singleton software barrier cannot establish that hardware guarantee.

## Current compiler candidate

- Explicit async selection, round_sync still selectable/default.
- Dependency coalescing is opt-in `tl.workergroup_merge_edges`, matching equal
  producer/consumer groups and depth, not names. `protected_slots` retains the
  full allocation membership in compiler IR and JSON. ABI v3 `slot_desc` is a
  representative allocation; all allocation descriptors remain budgeted.
- Conditional resident copy requires `tl.worker_resident_loop` naming the
  immediately enclosing reuse loop and an exact first-iteration guard.
  Allocation must enclose that loop; source is global, invariant across reuse;
  the write initializes the complete shared slot. Other conditionals reject.
- Current supported lifetime requires inner trips == depth, exactly one writer,
  and a cross-role consumer. Nonaligned initial ordinals reject at the owner.
- Lexical ordinal covers the entire outer nest. At a resident epoch boundary,
  producer additionally waits for every free slot of the preceding epoch before
  replacing any panel. Per-iteration ready/free publication remains unconditional.
- Sequential output staging and collective release use existing group ownership.
  No runtime fastpath changes or generated-C string patching.

## Runtime handoff (mandatory before device execution)

Async input and output DMA CAN overlap. The isolated candidate's native_dma.c
serializes the entire dmstart/dmwait interval with a mutex. Formal backend must
do likewise, alongside bidirectional registered-span/permission validation.
One leader per group is not one leader for the whole DMA engine. Retain exact
uint32 byte-address overflow, 24-bit width/stride, 16-bit row and 128B alignment
checks. Compiler introduces no new qurt imports.

Runtime must retain collective free publication, cancellation wakeup and join
quiescence. The separate singleton optimization and HMX completion audit belong
to the runtime owner. Do not substitute compiler host simulations for that audit.

## Artifacts and measured compile resources

`/tmp/opencode/compiler_async_formal/bn{256,512}.{c,json,o,s,su}` and `_pack.h`
are generated with M=1024 N=12288 K=2560 BM=64 BK=1280, `--transform-output
--schedule async --reuse-b --merge-edges --load 1`. Both plans have 4 workers,
3 groups, 2 edges, 8 events, 3 depth-two allocation slots, FP16 output ABI.
BN256 VTCM=1736960, compiler static callback frame=18176 bytes.
BN512 VTCM=3145984, compiler static callback frame=35584 bytes.
Frames exclude runtime caller/leaf stack requirements; compare with actual
worker stack budgets before launch. FP64 precision criteria are unchanged.

## Unfinished validation / known limitations

Historical focused tests had five fixture construction failures. Primary repaired
the root allocation/Then/evaluate-copy builder API usage without changing
assertions or thresholds (test API maintenance, not a numerical contract change).
The current focused plus neighboring suite was rerun: 34 passed in 8.21s.
Primary reports independent verifier PASS for that async version; this session
does not independently attest that report or extend it to the new cast changes.
The random schedule model treats consumer release collectively, not as native
runtime execution. Cancellation and overflow test expansion remains outstanding.
Whole-function auditing now rejects additional writes to resident storage or
its source, multiple Buffer views sharing a data variable, and unknown calls
with buffer/pointer escapes. This is conservative and still needs dedicated
negative-test execution. Distinct external parameter pointers can alias at
runtime; that external allocation-disjointness contract is not yet enforced by
the wrapper. Do not treat this as a complete arbitrary-pointer alias proof.
JSON now exports exact ABI slots by data identity against physical allocation
attributes. The owner rejects initial+iteration_count uint64 overflow before
runtime launch, in addition to resident initial alignment.

Current regenerated half-only artifacts retain the same source fingerprints.
BN256 source SHA is 3370fafc941321b9c0ca7d220178e2d200631dd569b0b52399e85900a83b38ea;
BN512 is 6098d4bb82be08fd7fcc612275f469d15bbde3d3041994d68419e4d805ab3c15.
Both were regenerated and compiled to O2 objects and assembly after the audit
and launch-check changes. These are candidates, not sealed release artifacts.

Transform now supports restricted half/float dtype conversion composed with
the supported permutations; see transform_cast_contract.md. General permutations
remain unsupported. Neither phone precision nor speed is
claimed. No independent verifier tool is available in this implementation session.
