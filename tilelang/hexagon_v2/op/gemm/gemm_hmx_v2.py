from __future__ import annotations

from tilelang import language as T
from tilelang.hexagon.language.layout import make_ah_layout, make_wh_layout
from tilelang.tileop.gemm.gemm_base import GemmBase
from tvm import DataType, tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_HMX_V2 = "hexagon.hmx"


class GemmHMXv2(GemmBase):
    """Lower ss ``T.gemm`` to the v2 HMX intrinsic surface.

    Phase-1 codegen still emits the final GEMM kernel from static shape attrs,
    but this lowering makes the engine path exercise target op resolution and
    produces canonical intrinsic calls for golden inspection.
    """

    def infer_layout(self, target: Target, thread_nums: int):
        self._validate_supported()
        return {self.A: make_ah_layout(self.A), self.B: make_wh_layout(self.B)}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        self._validate_supported()
        A_data = self.A.data
        B_data = self.B.data
        K = self.K
        clear_accum = self.clear_accum
        @T.prim_func
        def _gemm_hmx_v2() -> None:
            if clear_accum:
                T.evaluate(T.call_pure_extern("handle", "tl.hexagon_v2.hmx_acc_clear", 0))
            for ko in T.serial(0, K // 1024):
                T.evaluate(T.call_pure_extern("handle", "tl.hexagon_v2.hmx_mma_deep", A_data, B_data, 32, 2048, 2048))

        return _gemm_hmx_v2.with_attr("global_symbol", "hexagon_v2.gemm_hmx_v2")

    def _validate_supported(self) -> None:
        if self.trans_A or not self.trans_B:
            raise ValueError("hexagon_v2.hmx Phase-1 supports NT GEMM only (transpose_B=True)")
        if str(self.a_dtype) != "float16" or str(self.b_dtype) != "float16":
            raise ValueError("hexagon_v2.hmx Phase-1 supports fp16 operands only")
        if str(self.accum_dtype) not in ("float32", "float16"):
            raise ValueError("hexagon_v2.hmx Phase-1 supports fp32/fp16 accumulator markers only")
        if self.M % 32 or self.N % 256 or self.K % 64 or self.K > 4096:
            raise ValueError("hexagon_v2.hmx requires M%32==0, N%256==0, K%64==0, K<=4096")
        for buf in (self.A, self.B):
            if not str(buf.scope()).startswith("shared") and not str(buf.scope()).startswith("vtcm"):
                raise ValueError("hexagon_v2.hmx operands must be shared/vtcm buffers")
            if DataType(buf.dtype).bits != 16:
                raise ValueError("hexagon_v2.hmx operand buffers must be fp16")
