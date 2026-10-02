# Active-group reduction: implementation status

`group_reduce.h` is an experimental template, NOT a supported DSL operation.
No frontend/IR lifecycle verifier is connected yet. Do not call it from FA or
claim that it closes the intrinsic matrix.

Implemented template contract:

* FP32 input/accumulator/result; sum or ordered max (`a > b ? a : b`).
  Max is not claimed to implement IEEE maximum/NaN propagation. Inputs must be
  finite until a numerical-domain verifier and nonfinite tests are added.
* Whole active worker group only, power-of-two count 1..64, additionally bounded
  by runtime capabilities (the device team cap is currently 6). No subgroup
  inside a group, no full-team barrier substituted for a group barrier.
* Adjacent-pair tree with strides 1,2,4,...; only left endpoints update partials.
  Seed is combined once after the tree, not into each worker's partial.
* Caller supplies exclusive DDR scratch, at least `Workers` FP32 elements,
  disjoint from other live data and concurrent groups. No internal allocation.
* Publish barrier, one barrier per tree level, then a final reader-completion
  barrier before return/reuse. There is no externally visible partial ticket.
* All group members call in identical order inside an active stage. Pointer,
  size, group ID, local ID/count and barrier return codes are runtime guarded.
  These guards DO NOT prove uniform invocation or scratch alias safety.

Development probes (2026-10-01): target O2 compile and a hand-written pthread
template oracle for independent groups 4+2, sum/max, 100 scratch reuses passed.
That oracle emulates the group barrier and is NOT generated callback execution
on the worker runtime. Nonmember, alias and invalid-lifecycle negative tests,
frontend/IR integration, and independent verification remain outstanding.

The legacy `reduce.h::AllReduce` has its original descending XOR/full-team
barrier semantics and has NOT been relabelled as this group operation.

## Other intrinsic limits (unchanged)

* Masked load needs a proven readable complete aligned 128-byte allocation;
  masking does not make an unpadded tail safe.
* FP32 single-round FMA is unsupported. FP16 FMA requires native mode.
* Reciprocal/division require the explicit bounded NR2 mode, not IEEE division.
* Exp/exp2 modes are explicit target contracts, not interchangeable with strict
  IEEE math. No new device numerical validation is claimed here.
* Shared row state uses compiler-fixed cyclic ownership for persistent stages;
  fragment recurrence is still unsupported. TileOp write regions participate in
  parallel partition decisions after the copy-effect fix.
