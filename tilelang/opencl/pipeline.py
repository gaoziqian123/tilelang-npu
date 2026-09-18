from __future__ import annotations

from tvm import IRModule, ir, s_tir, tirx
from tvm.target import Target
from tvm.tirx import Bind, BufferLoad, BufferStore, PyStmtExprMutator
from tvm.tirx.buffer import decl_buffer
from tvm.tirx.stmt import AllocBuffer, DeclBuffer
from tvm.tirx.transform import prim_func_pass

import tilelang
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)


def _clone_var_with_scope(var: tirx.Var, scope: str) -> tirx.Var:
    type_annotation = getattr(var, "type_annotation", None)
    if isinstance(type_annotation, ir.PointerType):
        return tirx.Var(str(var.name), ir.PointerType(type_annotation.element_type, scope), var.span)
    return tirx.Var(str(var.name), var.dtype, var.span)


def _clone_buffer_with_scope(buffer: tirx.Buffer, scope: str, data: tirx.Var | None = None) -> tirx.Buffer:
    return decl_buffer(
        buffer.shape,
        buffer.dtype,
        data=data if data is not None else buffer.data,
        name=str(buffer.name),
        strides=buffer.strides,
        elem_offset=buffer.elem_offset,
        scope=scope,
        data_alignment=buffer.data_alignment,
        offset_factor=buffer.offset_factor,
        buffer_type=str(buffer.buffer_type) if buffer.buffer_type is not None else "",
        axis_separators=buffer.axis_separators,
        span=buffer.span,
    )


@tirx.functor.mutator
class _SharedDynToSharedMutator(PyStmtExprMutator):
    def __init__(self, extra_bound_vars=()):
        super().__init__()
        self.scope_map = {"shared.dyn": "shared", "local.fragment": "local"}
        self.buffer_map: list[tuple[tirx.Buffer, tirx.Buffer]] = []
        self.var_map: list[tuple[tirx.Var, tirx.Var]] = []
        # Vars that already have storage elsewhere: function parameters and
        # Bind-introduced handle aliases (e.g. into the merged dynamic shared
        # allocation).  A DeclBuffer whose data var is NOT bound here is a
        # fresh thread-local allocation and must be materialized as
        # AllocBuffer for the legacy OpenCL codegen (its DeclBuffer visitor
        # emits nothing and never registers the var).
        self.bound_vars: list[tirx.Var] = list(extra_bound_vars)

    def _is_bound_var(self, var: tirx.Var) -> bool:
        return any(var.same_as(bound) for bound in self.bound_vars)

    def _map_buffer(self, buffer: tirx.Buffer) -> tirx.Buffer:
        for old, new in reversed(self.buffer_map):
            if buffer.same_as(old) or buffer.data.same_as(old.data):
                return new
        return buffer

    def _map_var(self, var: tirx.Var) -> tirx.Var:
        for old, new in reversed(self.var_map):
            if var.same_as(old):
                return new
        return var

    def visit_alloc_buffer_(self, op: AllocBuffer):
        new_scope = self.scope_map.get(op.buffer.scope())
        if new_scope is None:
            return super().visit_alloc_buffer_(op)
        new_data = _clone_var_with_scope(op.buffer.data, new_scope)
        new_buffer = _clone_buffer_with_scope(op.buffer, new_scope, new_data)
        self.buffer_map.append((op.buffer, new_buffer))
        self.var_map.append((op.buffer.data, new_data))
        return AllocBuffer(new_buffer, getattr(op, "annotations", None), getattr(op, "span", None))

    def visit_var_(self, op: tirx.Var):
        return self._map_var(op)

    def visit_decl_buffer_(self, op: DeclBuffer):
        buffer = op.buffer
        new_scope = self.scope_map.get(buffer.scope())
        if new_scope is not None:
            buffer = _clone_buffer_with_scope(buffer, new_scope)
        if self._is_bound_var(buffer.data):
            # Alias into another allocation (e.g. merged dynamic shared
            # memory): keep the DeclBuffer, only the scope is normalized.
            if buffer is op.buffer:
                return super().visit_decl_buffer_(op)
            return DeclBuffer(buffer, getattr(op, "span", None))
        # Fresh thread-local declaration: legacy OpenCL codegen emits nothing
        # for DeclBuffer, so materialize the storage as an AllocBuffer.
        return AllocBuffer(buffer, getattr(op, "annotations", None), getattr(op, "span", None))

    def visit_buffer_load_(self, op: BufferLoad):
        buffer = self._map_buffer(op.buffer)
        indices = [self.visit_expr(index) for index in op.indices]
        predicate = self.visit_expr(op.predicate) if op.predicate is not None else None
        return BufferLoad(buffer, indices, predicate, getattr(op, "span", None))

    def visit_buffer_store_(self, op: BufferStore):
        buffer = self._map_buffer(op.buffer)
        value = self.visit_expr(op.value)
        indices = [self.visit_expr(index) for index in op.indices]
        predicate = self.visit_expr(op.predicate) if op.predicate is not None else None
        return BufferStore(buffer, value, indices, predicate, getattr(op, "span", None))


def _OpenCLSharedDynToShared():
    @prim_func_pass(opt_level=0)
    def _pass(func: tirx.PrimFunc, _mod: IRModule, _ctx) -> tirx.PrimFunc:
        bound = list(func.params)
        # match_buffer / buffer_map introduced buffers reuse parameter vars;
        # also cover any explicitly mapped buffer data vars.
        buffer_map = getattr(func, "buffer_map", None)
        if buffer_map:
            for _var, buf in buffer_map.items():
                bound.append(buf.data)
        # Collect Bind-introduced handle aliases (e.g. into the merged dynamic
        # shared allocation) with a C++-side walk: overriding visit_bind_ in a
        # PyStmtExprMutator silently drops Bind nodes in this tirx fork.
        tirx.stmt_functor.post_order_visit(
            func.body, lambda node: bound.append(node.var) if isinstance(node, Bind) else None
        )
        return func.with_body(_SharedDynToSharedMutator(bound).visit_stmt(func.body), span=func.span)

    return _pass


def OpenCLPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    # OpenCL uses the same SIMT lowering shape as WebGPU for this minimal
    # backend: lower the launch nest to thread_extent bindings for codegen.
    # TODO(adreno): revisit this cloned WebGPU pipeline and trim/retune passes
    # once Adreno-specific kernels beyond L0 elementwise are enabled.
    # TVM's legacy codegen_opencl.cc PrintStorageScope accepts global/shared/
    # texture_* but not shared.dyn; normalize shared.dyn near the end below.
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
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
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tilelang.transform.VerifyReducerConsumed()(mod)

    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
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
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge, disable_reuse=disable_reuse)(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    # TVM's legacy OpenCL codegen recognizes the local address space only as
    # "shared".  TileLang's default T.alloc_shared scope is "shared.dyn", so
    # normalize it after shared.dyn ThreadSync has inserted barriers but before
    # OpenCL C emission chooses the storage qualifier.
    mod = _OpenCLSharedDynToShared()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod
