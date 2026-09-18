"""OpenCL scalar FMA fallback GEMM."""

from __future__ import annotations

from tilelang import language as T
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.transform.simplify import _Simplify
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_FMA = "opencl.fma"


class GemmFMA(GemmBase):
    """Portable OpenCL FMA GEMM lowering.

    This keeps A/B tiles in OpenCL local memory, while each work-item accumulates
    through a thread-private scalar before writing its destination slot.
    """

    def infer_layout(self, target: Target, thread_nums: int):
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        M, N, K = self.M, self.N, self.K
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        assert len(A_region.region) >= 2, "FMA GEMM requires at least 2D A"
        assert len(B_region.region) >= 2, "FMA GEMM requires at least 2D B"
        assert len(C_region.region) >= 2, "FMA GEMM requires at least 2D C"

        A_other = [r.min for r in A_region.region[:-2]]
        B_other = [r.min for r in B_region.region[:-2]]
        C_other = [r.min for r in C_region.region[:-2]]
        a0 = A_region.region[-2].min
        a1 = A_region.region[-1].min
        b0 = B_region.region[-2].min
        b1 = B_region.region[-1].min
        c0 = C_region.region[-2].min
        c1 = C_region.region[-1].min

        trans_A = self.trans_A
        trans_B = self.trans_B
        clear_accum = self.clear_accum
        accum_dtype = self.accum_dtype
        thread_nums = thread_bounds.extent

        @T.prim_func
        def _gemm_fma() -> None:
            accum = T.alloc_local((1,), accum_dtype)

            for c_flat in T.serial(thread_var, M * N, thread_nums):
                i = c_flat // N
                j = c_flat % N
                if clear_accum:
                    accum[0] = T.cast(0, accum_dtype)
                else:
                    accum[0] = T.cast(C_buf[tuple(C_other) + (c0 + i, c1 + j)], accum_dtype)
                for k in T.serial(K):
                    a_i = k if trans_A else i
                    a_j = i if trans_A else k
                    b_i = j if trans_B else k
                    b_j = k if trans_B else j
                    accum[0] += T.cast(A_buf[tuple(A_other) + (a0 + a_i, a1 + a_j)], accum_dtype) * T.cast(
                        B_buf[tuple(B_other) + (b0 + b_i, b1 + b_j)], accum_dtype
                    )
                C_buf[tuple(C_other) + (c0 + i, c1 + j)] = accum[0]

        return _Simplify(_gemm_fma, inline_let=True)
