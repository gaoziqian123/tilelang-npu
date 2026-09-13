from __future__ import annotations

from tilelang import language as T
from tilelang.hexagon.emitter import hexagon_fold_view_base_rc
from tilelang.hexagon.language.layout import make_ah_layout, make_wh_layout
from tilelang.tileop.gemm.gemm_base import GemmBase
from tvm import DataType, tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_HMX = "hexagon.hmx"


class GemmHMX(GemmBase):
    """Hexagon HMX GEMM marker lowered to a Hexagon intrinsic call."""

    def infer_layout(self, target: Target, thread_nums: int):
        layout_map = {}
        if _can_use_hmx_vtcm_layout(self.A):
            layout_map[self.A] = make_ah_layout(self.A)
        if _can_use_hmx_vtcm_layout(self.B):
            layout_map[self.B] = make_wh_layout(self.B)
        return layout_map

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
        A_base = _flatten_view_base(self.A_base_offsets, self.A)
        B_base = _flatten_view_base(self.B_base_offsets, self.B)
        M, N, K = self.M, self.N, self.K

        @T.prim_func
        def _gemm_hmx() -> None:
            T.evaluate(T.call_pure_extern("handle", "hexagon.gemm_hmx", A_data, B_data, C_data, A_base[0], A_base[1], B_base[0], B_base[1], M, N, K))

        return _gemm_hmx.with_attr("global_symbol", "hexagon.gemm_hmx")


def _flatten_view_base(offsets, buffer):
    """Fold leading-dim view mins into the row index of the trailing 2-D plane.

    A point view like ``state_ah[hv, 0, 0]`` on a rank-3 buffer denotes the
    trailing [shape[-2], shape[-1]] tile rooted at that point.  The AH/WH byte
    offset must therefore see ``hv * shape[-2]`` extra rows, not just the last
    two mins.  Whole-buffer operands keep [0, 0] and stay byte-identical.
    """

    row, col = hexagon_fold_view_base_rc(list(offsets), buffer)
    return [row, col]


def _can_use_hmx_vtcm_layout(buffer: tirx.Buffer) -> bool:
    """Return whether ``buffer`` satisfies the Python-side AH/WH prechecks."""

    if not str(buffer.scope()).startswith("vtcm"):
        return False
    dtype = DataType(buffer.dtype)
    if dtype.bits != 16:
        return False
    if len(buffer.shape) < 2:
        return False
    return all(_is_const_multiple_of_32(dim) for dim in buffer.shape[-2:])


def _is_const_multiple_of_32(dim) -> bool:
    value = None
    if isinstance(dim, int):
        value = dim
    elif isinstance(dim, tirx.IntImm):
        value = int(dim.value)
    return value is not None and value % 32 == 0
