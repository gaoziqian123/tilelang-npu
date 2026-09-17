"""Hexagon lowering pipeline 骨架。"""

from __future__ import annotations

import os
from pathlib import Path

from tvm import IRModule, tirx
from tvm.target import Target

import tilelang
from tilelang.transform import PassConfigKey, PassContext

from .emitter import emit_hexagon_c
from .passes import HexagonCopyPartition, ProductReduceFusion, StoragePlan, WScratchPlan, HexagonProfileConfig, HexagonVerify, WriteSet


def default_emit_path() -> str:
    return str(Path(__file__).with_name("hexagon_kernel.c"))


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

    out = os.environ.get("TILELANG_HEXAGON_EMIT_C", default_emit_path())
    emit_hexagon_c(mod, target, out)
    return mod
