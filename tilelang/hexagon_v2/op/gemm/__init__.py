from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl

from ...target import target_is_hexagon_v2
from .gemm_hmx_v2 import GEMM_INST_HMX_V2, GemmHMXv2


register_gemm_impl("hexagon_v2.hmx", GEMM_INST_HMX_V2, target_is_hexagon_v2, GemmHMXv2)
