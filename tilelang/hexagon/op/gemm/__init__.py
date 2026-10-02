from tilelang.tileop.gemm.registry import register_gemm_impl
from ...target import target_is_hexagon
from .gemm_hmx import GemmHMX

register_gemm_impl("hexagon.hmx", "hexagon.hmx", target_is_hexagon, GemmHMX)
