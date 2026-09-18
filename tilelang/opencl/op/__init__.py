"""OpenCL TileOp registrations."""

from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from .gemm_fma import GEMM_INST_FMA, GemmFMA


def _target_is_opencl(target) -> bool:
    kind = getattr(target, "kind", None)
    return kind is not None and getattr(kind, "name", None) == "opencl"


register_gemm_impl("opencl.fma", GEMM_INST_FMA, _target_is_opencl, GemmFMA)
