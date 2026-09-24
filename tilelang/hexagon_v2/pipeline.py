"""Hexagon-v2 pass pipeline skeleton.

Phase 0/1 intentionally implement the minimum pipeline necessary for GEMM
vertical slicing.  The functions below document the full design hooks while the
example path calls directly into :mod:`tilelang.hexagon_v2.codegen`.
"""

from __future__ import annotations

from tvm.ir import CallingConv
from tvm import IRModule, tirx
from tvm.target import Target
from tvm.tirx import PrimFunc

import tilelang
from tilelang.transform import PassConfigKey, PassContext


def _mark_device_kernel(mod: IRModule) -> IRModule:
    funcs = {}
    for gv, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            func = func.with_attr("calling_conv", CallingConv.DEVICE_KERNEL_LAUNCH)
        funcs[gv] = func
    new_mod = IRModule(funcs)
    if mod.attrs:
        new_mod = new_mod.with_attrs(mod.attrs)
    return new_mod


def NormalizeScopes(mod: IRModule) -> IRModule:
    """Phase-1 stub for first-class Hexagon scope/layout normalization."""

    return mod


def LowerHexagonAsyncCopy(mod: IRModule) -> IRModule:
    """Phase-1 stub; pooled copy/overlap materialization remains in runtime helpers."""

    return mod


def HexagonWorkerPlan(mod: IRModule) -> IRModule:
    return mod


def HexagonABIPlan(mod: IRModule) -> IRModule:
    return mod


def HexagonStoragePlan(mod: IRModule) -> IRModule:
    """Phase-1 stub; codegen bridge owns fixed VTCM offsets for GEMM_NT."""

    return mod


def HexagonV2Verify(mod: IRModule) -> IRModule:
    return mod


def HexagonV2PassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    """First Hexagon-v2 lowering pipeline.

    The first round intentionally keeps async-copy, worker-plan and ABI-plan as
    no-op stubs, but it runs the same engine-facing pass envelope as the future
    target codegen path and lets ``LowerTileOp`` resolve ``T.gemm`` to the v2 HMX
    intrinsic surface.
    """

    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = NormalizeScopes(mod)
    with target:
        with PassContext(
            config={
                PassConfigKey.TL_LAYOUT_INFERENCE_ANNOTATE_PARALLEL_LOOPS.value: False,
                PassConfigKey.TL_LAYOUT_INFERENCE_FILL_DEFAULT_LAYOUT.value: True,
            }
        ):
            mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
    mod = LowerHexagonAsyncCopy(mod)
    mod = HexagonWorkerPlan(mod)
    mod = HexagonABIPlan(mod)
    mod = HexagonStoragePlan(mod)
    mod = HexagonV2Verify(mod)
    return _mark_device_kernel(mod)


class HexagonV2VerifyError(ValueError):
    """Raised by future v2 verifier passes."""
