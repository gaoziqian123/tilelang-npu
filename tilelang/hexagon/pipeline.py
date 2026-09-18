"""Hexagon lowering pipeline 骨架。"""

from __future__ import annotations

from tvm.ir import CallingConv
from tvm import IRModule, tirx
from tvm.target import Target
from tvm.tirx import PrimFunc

import tilelang
from tilelang.transform import PassConfigKey, PassContext

from .passes import HexagonCopyPartition, ProductReduceFusion, StoragePlan, WScratchPlan, HexagonProfileConfig, HexagonVerify, WriteSet


def _mark_device_kernel(mod: IRModule) -> IRModule:
    """Mark Hexagon lowered PrimFuncs as device kernels for standard codegen split."""

    funcs = {}
    for gv, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            func = func.with_attr("calling_conv", CallingConv.DEVICE_KERNEL_LAUNCH)
        funcs[gv] = func
    new_mod = IRModule(funcs)
    if mod.attrs:
        new_mod = new_mod.with_attrs(mod.attrs)
    return new_mod


def HexagonPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    """v1 Python 侧 pass 钩子。

    Hexagon target-specific TileOp implementations lower T.copy/T.gemm into
    explicit ``hexagon.*`` intrinsics; subsequent Hexagon passes verify and plan
    those intrinsics before the C emitter mechanically prints recipes.
    """

    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)
    pass_config = PassContext.current().config
    hexagon_prof = bool(pass_config.get(PassConfigKey.TL_HEXAGON_PROF.value, False))
    with target:
        with PassContext(
            config={
                PassConfigKey.TL_LAYOUT_INFERENCE_ANNOTATE_PARALLEL_LOOPS.value: False,
                PassConfigKey.TL_LAYOUT_INFERENCE_FILL_DEFAULT_LAYOUT.value: True,
            }
        ):
            mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
    mod = HexagonCopyPartition()(mod)
    mod = ProductReduceFusion()(mod)
    mod = WriteSet()(mod)
    mod = StoragePlan()(mod)
    mod = WScratchPlan()(mod)
    mod = HexagonProfileConfig(hexagon_prof)(mod)
    mod = HexagonVerify()(mod)
    mod = _mark_device_kernel(mod)
    from .codegen import remember_lowered_mod

    mod = remember_lowered_mod(mod)

    return mod
