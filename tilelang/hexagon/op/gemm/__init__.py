from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl

from .gemm_hmx import GEMM_INST_HMX, GemmHMX


def _match_hmx(target) -> bool:
    return target.kind.name == "hexagon" and target.tag != "hexagon_v2"


register_gemm_impl("hexagon.hmx", GEMM_INST_HMX, _match_hmx, GemmHMX)
