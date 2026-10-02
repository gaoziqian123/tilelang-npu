# Opt-in local row map/reduce planning

`tl.hexagon.plan_local_row_reduce=True` creates the versioned physical row
plan after LowerTileOp / FlattenBuffer. This is independent of local.reducer
epochs. `tl.hexagon.fuse_local_row_map=True` then forwards producer SSA after
vectorization, before storage rewrite. Both options default to false.

The C++ pass accepts adjacent pure local-row map and planned sum/max, with
contiguous 1-lane or 32-lane stores. Consecutive partitions must provably cover
the entire row exactly once, in increasing order. Bounds cannot read mutable
memory. Dynamic complementary prefix/tail partitions are supported when the
analyzer proves their bounds. Opaque boundaries, address escapes, distinct
buffer views of the same data, stencils, partial domains and unsupported maps
retain the unfused plan. Row-loop alignment is allowed only for equal domains;
producer accesses to the reduction destination are rejected.

The map remains ordinary TIR arithmetic/cast/select/calls. A Bind computes its
value once; the original store and row_map_update use that SSA value. The leaf
implements only incremental reduction, not exp/softmax or an operator callback.
Independent consumers (including cast stores) still load the materialized row;
this version does not claim to eliminate those loads.

Numerical ABI: 32 neutral lane chains, the original rotating horizontal tree
16/8/4/2/1, seed after tree, ordered scalar tail. Classification shares producer
values. Exceptional rows replay the stored row in index order, never the map.
Materialization is mandatory. The eliminated read is the finite full-vector
row scan formerly performed by row_reduce_f32, NOT tail or exception replay.

`fa_persistent.make_fa(instrumentation=False)` / CLI `--no-instrumentation`
omits the entire region instrumentation at DSL construction. Default remains
instrumented. No pass removes extern calls or changes their effect annotation.
The noninstrumented persistent FA terminal mask/max and prefix-exp/zero-tail
sum can fuse; the instrumented version does not. Later fp16 consumers remain
intact. `tl.hexagon.row_map_reducers` selects reducer kinds (bit 0 sum, bit 1
max; default 3; valid range 0..3). The max-only experiment uses 2: sum across
exp is deliberately not selected, independent of operator name or shape. This
is explicit kind selection, not an automatic liveness/cost model.

A complete terminal map can follow earlier materialized full-row stages.
Only the terminal stage forwards its SSA value; prefix stages retain their
original order. Producer calls must be pure, aliases/escapes are rejected,
and the producer may neither read nor write the reduction destination.
Terminal source-row loads must use the same coordinate as the store. Constant
seed initialization is folded into finish only when it has no memory loads.
A vectorized unit-trip loop reduced to one 32-lane store is reconstructed for
the same domain verifier. The max-only implementation does not fuse exp and
therefore does not carry incremental sum state across its call.

Host bitwise tests compare the same physical tree, not a sequential reference.
Tests also compile actual generated persistent FA and generic vector map C++
with Hexagon clang -c. No device precision/performance acceptance is implied.
