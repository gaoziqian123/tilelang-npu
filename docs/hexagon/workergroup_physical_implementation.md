# Worker pipeline physical storage implementation (in progress)

Canonical runtime ABI remains `workergroup_abi.h` v2; this note changes no
runtime symbol or descriptor. The production representation is distinct from
the earlier `MaterializeWorkerSlots` rank-expansion diagnostic pass.

`PlanWorkerPhysicalAllocations(vtcm_budget)` runs on actual shared allocations
after layout and scratch-producing TileOp lowering. It consumes logical slot
depths by data Var identity, accounts for explicit physical strides and dtype
bytes, aligns each version and allocation to at least 128 bytes, and includes
lowering-created scratch such as HMX scales. It rejects missing versioned
allocations, aliases, nonstatic extents and budget/overflow violations. It
preserves the input body and buffer ranks.

Its function attributes are `tl.workergroup_physical_allocations` (buffer,
data Var, physical shape/strides, offset, bytes_per_slot, depth, alignment) and
`tl.workergroup_vtcm_bytes`. These are compiler IR, not another runtime ABI.

Production CodeGenHexagon accepts explicit callback VTCM and ordinal Vars as
`tl.workergroup_callback_vtcm` / `tl.workergroup_callback_ordinal`. For planned
shared allocations it emits a pointer view expression into that VTCM base,
including `(ordinal % depth) * bytes_per_slot`. The expression is evaluated at
each access, rather than capturing a slot pointer outside the loop. Logical
rank-two HMX buffers are therefore not expanded to rank three. No fallback to
`tl_hex_vtcm_slice` is allowed for callback shared allocations without a plan.

The production Hexagon pipeline now detects explicit role scopes and runs
planning, whole-function ownership, layout, TileOp lowering, group sync,
physical allocation planning and callback construction. No-role functions
continue through the preexisting branch. Capability function attributes
`tl.workergroup_max_workers`, `tl.workergroup_max_events` and
`tl.workergroup_max_vtcm_bytes` are mandatory. Currently only one function,
one planned pipeline and one logical launch job are supported; nested pipelines
and unsupported effects fail closed. Static outer serial loops are supported.

`examples/hexagon/gemm/gemm_worker_pipeline.py` emits through `engine.lower`,
not a private testing chain. Its owner symbol is `gemm_worker_owner(caps, ddr,
vtcm, sync, sync_bytes, initial, A, B, C)`, with A/B fp16 pointers and C fp32.
The caller binds the runtime platform on the HMX owner thread and supplies
exclusive aligned backing allocations. Query the backend's actual sync size;
the compiler does not duplicate its private DDR layout. The default 32x64x64
example uses 6 workers, 3 groups, 3 edges, 12 events, 12544 VTCM bytes, zero DDR
data bytes, 4 ordinals; sync bytes are supplied and checked by runtime. Only
the owner symbol is registered as the module entry; internal helper/adapter
must not be invoked as legacy per-worker kernels.

Slot/edge ordering follows deterministic IR first access, never pointer order
or buffer names. Current evidence covers source reproducibility and Hexagon
object compilation, NOT execution correctness, hardware completion or speed.
