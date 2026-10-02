from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from .cost_model import AnnotateLaunchEstimates
from tilelang.transform.fuse_pointwise_stages import FusePointwiseStages
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)


def HexagonPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    # TileOp macro lowering queries Target.current(), including vectorization
    # of fragment fills. engine.lower callers need not establish that context.
    with target:
        from .dma_verify import VerifyNativeDMA
        mod = VerifyNativeDMA()(mod)
        from .vector_verify import VerifyVectorRequired
        from .vector_lowering import VectorLowering
        return VerifyVectorRequired()(VectorLowering()(_HexagonPassPipelineBody(mod, target)))


def _HexagonPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    has_roles = False
    def detect_roles(node):
        nonlocal has_roles
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.pipeline_stage":
            has_roles = True
    for func in mod.functions.values():
        if isinstance(func, tirx.PrimFunc):
            tirx.stmt_functor.post_order_visit(func.body, detect_roles)
    if has_roles:
        return _WorkerPassPipeline(mod, target)
    # Fail closed until group-local layout and the single-launch adapter have
    # been materialized. AttrStmt is otherwise silently ignored by CodeGenC.
    from tvm import get_global_func
    get_global_func("tl.hexagon.transform.RejectUnmaterializedWorkerPipeline")()(mod)
    # Native LegalizeNegativeIndex preserves physical GEMM region indices for
    # GemmNode's enclosing-loop bounds proof (no Python-style wrapping).
    mod = tirx.transform.BindTarget(target)(mod)
    mod = AnnotateLaunchEstimates()(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    # Native ownership proof runs while TileOp read/write regions still exist.
    from tvm import get_global_func
    mod = get_global_func('tl.transform.VerifyHexagonAsync')()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.CanonicalizeLegacyReducer()(mod)
    mod = tilelang.transform.VerifyReducerEpoch()(mod)
    mod = tilelang.transform.VerifyBufferInit()(mod)

    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)

    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.ReducerPlanAndMaterialize()(mod)
    LayoutVisual(mod)
    if pass_ctx.config.get("tl.enable_fuse_cast_copy", False):
        mod = tilelang.transform.FuseCastCopy()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.VerifyReducerConsumed()(mod)

    mod = _NumericalBeforeAllocation(mod, pass_ctx)

    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = _NumericalAfterFlatten(mod, pass_ctx)
    # Async captures outlive lexical regions. Until token-aware interference
    # analysis exists, never coalesce their VTCM/local allocations.
    has_async = False
    def detect_async(node):
        nonlocal has_async
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "hexagon.async_scope":
            has_async = True
    for f in mod.functions.values():
        if isinstance(f, tirx.PrimFunc):
            tirx.stmt_functor.post_order_visit(f.body, detect_async)
    if not has_async:
        mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    # The requirement belongs to device compute, never to the generated host
    # packed-ABI marshaling wrapper. Use the standard device calling convention.
    for gv, func in list(mod.functions.items()):
        if isinstance(func, tirx.PrimFunc) and func.attrs and int(func.attrs.get("calling_conv", 0)) != 2:
            mod.update_func(gv, func.without_attr("tl.vector_required"))
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = has_async or should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge, disable_reuse=disable_reuse)(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


def _NumericalBeforeAllocation(mod, pass_ctx):
    """Preserve the ordinary path's exact numerical pass order."""
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    # Opt-in until the device numerical/performance gate has been repeated.
    if not pass_ctx.config.get("tl.disable_fuse_pointwise_stages", True):
        mod = FusePointwiseStages()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    if pass_ctx.config.get("tl.enable_ordered_accumulator_promotion", False):
        mod = tilelang.transform.PromoteOrderedAccumulator()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    return mod


def _NumericalAfterFlatten(mod, pass_ctx):
    if pass_ctx.config.get("tl.hexagon.plan_local_row_reduce", False):
        mod = tilelang.transform.PlanLocalRowReduce()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    from .vector_lowering import VectorLowering
    from .vector_verify import VerifyVectorRequired
    mod = VerifyVectorRequired()(VectorLowering()(mod))
    if pass_ctx.config.get("tl.hexagon.fuse_local_row_map", False):
        mod = tilelang.transform.FuseLocalRowMap()(mod)
    return mod


def _WorkerPassPipeline(mod: IRModule, target: Target) -> IRModule:
    """Opt-in fixed-group lowering; capabilities are explicit function attrs."""
    from tvm import get_global_func
    if len(mod.functions) != 1:
        raise ValueError("worker pipeline currently requires one function per module")
    func = next(iter(mod.functions.values()))
    keys = ("tl.workergroup_max_workers", "tl.workergroup_max_events", "tl.workergroup_max_vtcm_bytes")
    if not func.attrs or any(key not in func.attrs for key in keys):
        raise ValueError("worker pipeline requires explicit runtime capability attributes")
    workers, events, vtcm = (int(func.attrs[key]) for key in keys)
    from .collective_verify import VerifyGroupCollectives
    mod = VerifyGroupCollectives()(mod)
    pass_ctx = tilelang.transform.get_pass_context()
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = get_global_func("tl.hexagon.transform.PlanWorkerPipeline")(workers, events, vtcm)(mod)
    mod = get_global_func("tl.hexagon.transform.InferWorkerOwnership")()(mod)
    mod = tilelang.transform.LayoutInference()(mod)
    if pass_ctx.config.get("tl.enable_fuse_cast_copy", False):
        raise ValueError("worker FuseCastCopy requires slot/edge liveness remapping; unsupported")
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = _NumericalBeforeAllocation(mod, pass_ctx)
    mod = get_global_func("tl.hexagon.transform.LowerWorkerSync")()(mod)
    mod = get_global_func("tl.hexagon.transform.PlanWorkerPhysicalAllocations")(vtcm)(mod)
    mod = get_global_func("tl.hexagon.transform.BuildWorkerCallbacks")()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = _NumericalAfterFlatten(mod, pass_ctx)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = tirx.transform.Simplify()(mod)
    # Recompute from the FINAL allocations, not pre-vectorization estimates.
    # ABI slots retain data identity; the planner rejects any lost identity.
    mod = get_global_func("tl.hexagon.transform.PlanWorkerPhysicalAllocations")(vtcm)(mod)
    mod = get_global_func("tl.hexagon.transform.StageWorkerReads")()(mod)
    for gv, f in list(mod.functions.items()):
        mod.update_func(gv, f.with_attr("calling_conv", 2))
    return mod
