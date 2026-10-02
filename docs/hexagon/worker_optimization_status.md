# Shared numerical passes: implementation status

The ordinary pipeline's numerical order is extracted without reordering into
`_NumericalBeforeAllocation` and `_NumericalAfterFlatten`. Worker lowering calls
the same helpers after group-local TileOp lowering / flattening, respectively.
This is not evidence that every optimization pattern matches every layout.

| Pass family | Worker status / limitation |
|---|---|
| DecoupleTypeCast, LegalizeVectorizedLoop, LegalizeSafeMemoryAccess, LowerAccessPtr, Simplify, HoistNonRestrictParams | Shared pre-allocation helper |
| optional FusePointwiseStages, PromoteOrderedAccumulator | Shared helper with existing flags; matching/effectiveness not yet verified |
| optional PlanLocalRowReduce, FuseLocalRowMap, VectorizeLoop | Shared post-flatten helper; worker metadata stability needs further coverage |
| UnrollLoop | Added after numerical lowering |
| FuseCastCopy | Explicit worker compile error: edge/slot liveness remapping not implemented |
| PipelinePlanning, InjectSoftwarePipeline | Not applicable to worker schedule; would double-version |
| StorageRewrite, shared-memory reuse/merge | Not applied: cross-role temporal lifetimes must not alias |
| PlanAndUpdateBufferAllocationLocation, HoistGlobalBufferAllocations | Not shared yet: slot metadata remap required |
| NarrowDataType, ConfigIndexBitwidth | Not shared yet: ordinal must stay 64-bit |
| LoopUnswitching, HoistIfThenElse, MergeIfStmt | Not shared yet: collective uniformity proof required |
| reducer canonicalization/verification/materialization | Not yet connected; planner's unsupported TileOps still fail closed |
| ThreadSync | Generic whole-team insertion not used; explicit group sync conversion retained |
| packed API / SplitHostDevice / device-launch lowering | Separate worker owner ABI, not numerical passes |
| early bounds/init/race checks | Not fully shared yet; planner region proof is not a replacement for all checks |

Physical planning is rerun after flatten/vectorization/unrolling and final
simplification. The final pass seeds version counts from ABI slots by data Var
identity, recomputes shapes/strides/alignment and total bytes from actual final
allocations, rejects duplicate aliases or missing slot allocation identities,
and rechecks the capability budget. No storage reuse is performed. The wrapper
resolves every ABI slot against this final plan. No AH column-layout improvement
is claimed. This final allocation check is not a proof of numerical equivalence
for every optional transform.

ABI v3: round-sync emits ROUND_SYNC=2, zero edges/events, and the total number
of actual team barriers across outer tiles as uint32 round_count. Empty inner
loops contribute zero rounds; overflow is a compile error. Async retains
EXACTLY_ONCE=1 and round_count=0. Slots still validate allocation backing.

TEST-CONTRACT-CHANGES: ABI assertion now requires version 3 and round_count
offset 76 (plan size remains 80); same-body fixture supplies the required
Hexagon Target context. Structural/numerical assertions are not weakened.

Evidence so far: default round-sync GEMM source and Hexagon O2 object compile;
three fresh processes emit identical source. No role/nonrole byte comparison,
assembly attribution, optional-pass hit proof or device execution claim yet.
Runtime team-barrier implementation/contract remains outside this compiler task.
