"""OpenCL scalar FMA fallback GEMM."""

from __future__ import annotations

from tilelang import language as T
from tilelang.layout import Fragment
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.utils.language import is_fragment, is_shared
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

    def _rr_b_layout(self) -> str:
        """Best-effort RR B layout recovery.

        The frontend records the intended layout in ``tl.opencl.b_layout`` for
        example kernels, but TileOp instances do not currently expose their
        enclosing PrimFunc attrs.  Prefer the actual B operand shape / trans_B
        contract, then fall back to the documented default (kn).
        """
        if self.trans_B:
            return "nk"
        try:
            b_shape = tuple(int(x) for x in self.B.shape)
            if len(b_shape) >= 2:
                if b_shape[-2] == self.K and b_shape[-1] == self.N:
                    return "kn"
                if b_shape[-2] == self.N and b_shape[-1] == self.K:
                    return "nk"
        except Exception:
            pass
        return "kn"

    def _is_gemm_gr(self) -> bool:
        """Direct-operand vector GEMM: A/B tiles read straight from global or
        shared memory, C accumulated in a fragment.  On Adreno, staging
        replicated A/B fragments in private memory costs one memory round-trip
        per FMA (measured 0.012-0.043 TFLOPS across bk=64/16/8, register file
        cannot hold replicate-amplified tiles), so the fast path keeps
        operands where they are and only the accumulator in registers.  Shared
        operands are fine: OpenCL vector loads work on __local pointers and
        Adreno local memory has real cross-thread reuse (e.g. flash-attention
        P tiles)."""
        return (
            not is_fragment(self.A)
            and not is_fragment(self.B)
            and is_fragment(self.C)
        )

    def infer_layout(self, target: Target, thread_nums: int):
        if self._is_gemm_gr():
            bm, bn, bk = self.M, self.N, self.K
            if bm % 8 != 0 or bn % 8 != 0 or bk % 4 != 0:
                return {}
            tiles_n = bn // 8
            expected_threads = (bm // 8) * tiles_n
            if int(thread_nums) != expected_threads:
                return {}
            frag_c = Fragment(
                (bm, bn),
                forward_fn=lambda i, j: ((i // 8) * tiles_n + (j // 8), (i % 8) * 8 + (j % 8)),
            )
            return {self.C: frag_c}
        if not self.is_gemm_rr():
            return {}

        bm, bn, bk = self.M, self.N, self.K
        if bm % 8 != 0 or bn % 8 != 0 or bk % 4 != 0:
            return {}
        tiles_n = bn // 8
        tiles_m = bm // 8
        expected_threads = (bm // 8) * tiles_n
        if int(thread_nums) != expected_threads:
            return {}

        frag_c = Fragment(
            (bm, bn),
            forward_fn=lambda i, j: ((i // 8) * tiles_n + (j // 8), (i % 8) * 8 + (j % 8)),
        )
        frag_a = Fragment(
            (bm, bk),
            forward_fn=lambda i, k, rep: ((i // 8) * tiles_n + rep.var, (i % 8) * bk + k),
            replicate=tiles_n,
        )
        if self._rr_b_layout() == "nk":
            frag_b = Fragment(
                (bn, bk),
                forward_fn=lambda j, k, rep: (rep.var * tiles_n + (j // 8), (j % 8) * bk + k),
                replicate=tiles_m,
            )
        else:
            frag_b = Fragment(
                (bk, bn),
                forward_fn=lambda k, j, rep: (rep.var * tiles_n + (j // 8), k * 8 + (j % 8)),
                replicate=tiles_m,
            )
        return {self.A: frag_a, self.B: frag_b, self.C: frag_c}

    def _lower_rr(
        self,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
    ):
        M, N, K = self.M, self.N, self.K
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        a0 = A_region.region[-2].min
        a1 = A_region.region[-1].min
        b0 = B_region.region[-2].min
        b1 = B_region.region[-1].min
        c0 = C_region.region[-2].min
        c1 = C_region.region[-1].min

        tiles_n = N // 8
        b_layout = self._rr_b_layout()
        clear_accum = self.clear_accum
        tid = thread_var - thread_bounds.min

        # Full-fp16 mode: when the C fragment is fp16 the whole inner chain
        # stays in half (no Convert nodes anywhere): half4 A loads, half8 B
        # loads, float16x8 accumulators, and a cast-free C store.  Otherwise
        # keep the fp32 inner chain (half loads widened via Cast).
        half_mode = str(self.accum_dtype) == "float16"
        v4_dtype = "float16x4" if half_mode else "float32x4"
        v8_dtype = "float16x8" if half_mode else "float32x8"
        zero = T.float16(0.0) if half_mode else T.float32(0.0)

        def as_v4(expr):
            return expr if half_mode else T.Cast(v4_dtype, expr)

        def as_v8(expr):
            return expr if half_mode else T.Cast(v8_dtype, expr)

        # The K loop stays serial in the TIR; the lowering PassContext enables
        # tl.UnrollLoop.unroll_local_access, which force-unrolls it only when
        # its index feeds local (fragment) buffers — i.e. exactly the staged
        # RR path.  GR reads global memory directly, so its K loop remains
        # rolled like the 1.35T anchor kernel.
        @T.prim_func
        def _gemm_rr_fma() -> None:
            acc = T.alloc_local((8,), v8_dtype)
            tm = tid // tiles_n
            tn = tid - tm * tiles_n
            ar = a0 + tm * 8
            bc = b0 + tn * 8
            cr = c0 + tm * 8
            cc = c1 + tn * 8

            for ii in T.unroll(8, explicit=True):
                if clear_accum:
                    acc[ii] = T.Broadcast(zero, 8)
                else:
                    acc[ii] = C_buf[cr + ii, T.Ramp(cc, 1, 8)]

            av = T.alloc_local((8,), v4_dtype)
            if b_layout == "kn":
                bv = T.alloc_local((4,), v8_dtype)
            else:
                bg = T.alloc_local((8,), v4_dtype)
            for pos4 in T.serial(K // 4):
                k4 = pos4 * 4
                av[0] = as_v4(A_buf[ar + 0, T.Ramp(a1 + k4, 1, 4)])
                av[1] = as_v4(A_buf[ar + 1, T.Ramp(a1 + k4, 1, 4)])
                av[2] = as_v4(A_buf[ar + 2, T.Ramp(a1 + k4, 1, 4)])
                av[3] = as_v4(A_buf[ar + 3, T.Ramp(a1 + k4, 1, 4)])
                av[4] = as_v4(A_buf[ar + 4, T.Ramp(a1 + k4, 1, 4)])
                av[5] = as_v4(A_buf[ar + 5, T.Ramp(a1 + k4, 1, 4)])
                av[6] = as_v4(A_buf[ar + 6, T.Ramp(a1 + k4, 1, 4)])
                av[7] = as_v4(A_buf[ar + 7, T.Ramp(a1 + k4, 1, 4)])
                if b_layout == "kn":
                    bv[0] = as_v8(B_buf[b0 + k4 + 0, T.Ramp(b1 + tn * 8, 1, 8)])
                    bv[1] = as_v8(B_buf[b0 + k4 + 1, T.Ramp(b1 + tn * 8, 1, 8)])
                    bv[2] = as_v8(B_buf[b0 + k4 + 2, T.Ramp(b1 + tn * 8, 1, 8)])
                    bv[3] = as_v8(B_buf[b0 + k4 + 3, T.Ramp(b1 + tn * 8, 1, 8)])
                    acc[0] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[0]], [0]), 8), bv[0], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[0]], [1]), 8), bv[1], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[0]], [2]), 8), bv[2], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[0]], [3]), 8), bv[3], acc[0]))))
                    acc[1] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[1]], [0]), 8), bv[0], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[1]], [1]), 8), bv[1], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[1]], [2]), 8), bv[2], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[1]], [3]), 8), bv[3], acc[1]))))
                    acc[2] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[2]], [0]), 8), bv[0], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[2]], [1]), 8), bv[1], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[2]], [2]), 8), bv[2], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[2]], [3]), 8), bv[3], acc[2]))))
                    acc[3] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[3]], [0]), 8), bv[0], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[3]], [1]), 8), bv[1], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[3]], [2]), 8), bv[2], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[3]], [3]), 8), bv[3], acc[3]))))
                    acc[4] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[4]], [0]), 8), bv[0], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[4]], [1]), 8), bv[1], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[4]], [2]), 8), bv[2], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[4]], [3]), 8), bv[3], acc[4]))))
                    acc[5] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[5]], [0]), 8), bv[0], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[5]], [1]), 8), bv[1], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[5]], [2]), 8), bv[2], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[5]], [3]), 8), bv[3], acc[5]))))
                    acc[6] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[6]], [0]), 8), bv[0], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[6]], [1]), 8), bv[1], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[6]], [2]), 8), bv[2], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[6]], [3]), 8), bv[3], acc[6]))))
                    acc[7] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[7]], [0]), 8), bv[0], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[7]], [1]), 8), bv[1], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[7]], [2]), 8), bv[2], T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[7]], [3]), 8), bv[3], acc[7]))))
                else:
                    bg[0] = as_v4(B_buf[bc + 0, T.Ramp(b1 + k4, 1, 4)])
                    bg[1] = as_v4(B_buf[bc + 1, T.Ramp(b1 + k4, 1, 4)])
                    bg[2] = as_v4(B_buf[bc + 2, T.Ramp(b1 + k4, 1, 4)])
                    bg[3] = as_v4(B_buf[bc + 3, T.Ramp(b1 + k4, 1, 4)])
                    bg[4] = as_v4(B_buf[bc + 4, T.Ramp(b1 + k4, 1, 4)])
                    bg[5] = as_v4(B_buf[bc + 5, T.Ramp(b1 + k4, 1, 4)])
                    bg[6] = as_v4(B_buf[bc + 6, T.Ramp(b1 + k4, 1, 4)])
                    bg[7] = as_v4(B_buf[bc + 7, T.Ramp(b1 + k4, 1, 4)])
                    vec8_ctor = "(half8)" if half_mode else "(float8)"
                    bv0 = T.call_extern(v8_dtype, vec8_ctor, T.Shuffle([bg[0]], [0]), T.Shuffle([bg[1]], [0]), T.Shuffle([bg[2]], [0]), T.Shuffle([bg[3]], [0]), T.Shuffle([bg[4]], [0]), T.Shuffle([bg[5]], [0]), T.Shuffle([bg[6]], [0]), T.Shuffle([bg[7]], [0]))
                    bv1 = T.call_extern(v8_dtype, vec8_ctor, T.Shuffle([bg[0]], [1]), T.Shuffle([bg[1]], [1]), T.Shuffle([bg[2]], [1]), T.Shuffle([bg[3]], [1]), T.Shuffle([bg[4]], [1]), T.Shuffle([bg[5]], [1]), T.Shuffle([bg[6]], [1]), T.Shuffle([bg[7]], [1]))
                    bv2 = T.call_extern(v8_dtype, vec8_ctor, T.Shuffle([bg[0]], [2]), T.Shuffle([bg[1]], [2]), T.Shuffle([bg[2]], [2]), T.Shuffle([bg[3]], [2]), T.Shuffle([bg[4]], [2]), T.Shuffle([bg[5]], [2]), T.Shuffle([bg[6]], [2]), T.Shuffle([bg[7]], [2]))
                    bv3 = T.call_extern(v8_dtype, vec8_ctor, T.Shuffle([bg[0]], [3]), T.Shuffle([bg[1]], [3]), T.Shuffle([bg[2]], [3]), T.Shuffle([bg[3]], [3]), T.Shuffle([bg[4]], [3]), T.Shuffle([bg[5]], [3]), T.Shuffle([bg[6]], [3]), T.Shuffle([bg[7]], [3]))
                    acc[0] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[0]], [0]), 8), bv0, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[0]], [1]), 8), bv1, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[0]], [2]), 8), bv2, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[0]], [3]), 8), bv3, acc[0]))))
                    acc[1] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[1]], [0]), 8), bv0, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[1]], [1]), 8), bv1, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[1]], [2]), 8), bv2, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[1]], [3]), 8), bv3, acc[1]))))
                    acc[2] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[2]], [0]), 8), bv0, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[2]], [1]), 8), bv1, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[2]], [2]), 8), bv2, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[2]], [3]), 8), bv3, acc[2]))))
                    acc[3] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[3]], [0]), 8), bv0, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[3]], [1]), 8), bv1, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[3]], [2]), 8), bv2, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[3]], [3]), 8), bv3, acc[3]))))
                    acc[4] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[4]], [0]), 8), bv0, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[4]], [1]), 8), bv1, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[4]], [2]), 8), bv2, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[4]], [3]), 8), bv3, acc[4]))))
                    acc[5] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[5]], [0]), 8), bv0, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[5]], [1]), 8), bv1, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[5]], [2]), 8), bv2, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[5]], [3]), 8), bv3, acc[5]))))
                    acc[6] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[6]], [0]), 8), bv0, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[6]], [1]), 8), bv1, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[6]], [2]), 8), bv2, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[6]], [3]), 8), bv3, acc[6]))))
                    acc[7] = T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[7]], [0]), 8), bv0, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[7]], [1]), 8), bv1, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[7]], [2]), 8), bv2, T.call_extern(v8_dtype, "mad", T.Broadcast(T.Shuffle([av[7]], [3]), 8), bv3, acc[7]))))

            for ii in T.unroll(8, explicit=True):
                for jj in T.unroll(8, explicit=True):
                    C_buf[cr + ii, cc + jj] = T.Shuffle([acc[ii]], [jj])

        return _Simplify(_gemm_rr_fma, inline_let=True)

    def infer_layout_old(self, target: Target, thread_nums: int):
        return {}

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        if self.is_gemm_rr() or self._is_gemm_gr():
            return self._lower_rr(target, thread_bounds, thread_var)

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
