from __future__ import annotations

from tilelang import language as T
from tilelang.tileop.gemm.gemm_base import GemmBase
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_HMX = "hexagon.hmx"


class GemmHMX(GemmBase):
    """Hexagon HMX GEMM marker lowered to a Hexagon intrinsic call."""

    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        A_data = self.A.data
        B_data = self.B.data
        C_data = self.C.data
        M, N, K = self.M, self.N, self.K

        @T.prim_func
        def _gemm_hmx() -> None:
            T.evaluate(T.call_pure_extern("handle", "hexagon.gemm_hmx", A_data, B_data, C_data, M, N, K))

        return _gemm_hmx.with_attr("global_symbol", "hexagon.gemm_hmx")
