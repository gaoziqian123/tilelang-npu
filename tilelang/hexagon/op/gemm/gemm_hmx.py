"""HMX shared/shared GEMM, expanded as TIR rather than target source text."""

from tilelang import language as T
from tilelang.hexagon import _ffi_api
from tilelang.layout import Fragment
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.utils.language import is_shared, is_full_region
from tvm.ir import Op
from tvm.ir import structural_equal
from tvm import tirx
from tvm.script.ir_builder import IRBuilder
from tvm import arith

GEMM_INST_HMX = "hexagon.hmx"


class GemmHMX(GemmBase):
    def _validate(self):
        analyzer = arith.Analyzer()
        for region in (self.ARegion, self.BRegion, self.CRegion):
            if region is None:  # Also permit focused storage-validation tests.
                continue
            for axis, shape in zip(region.region, region.buffer.shape):
                for condition in (axis.extent > 0, axis.min >= 0,
                                  axis.min + axis.extent <= shape):
                    if analyzer.can_prove(~condition):
                        raise ValueError("HMX region bounds are invalid")
        buffers = (self.A, self.B, self.C)
        for i, buffer in enumerate(buffers):
            if any(buffer.data.same_as(other.data) for other in buffers[:i]):
                raise ValueError("HMX operands must not alias the same data Var")
            if len(buffer.shape) != 2:
                raise ValueError("HMX requires rank-two backing buffers")
            if not analyzer.can_prove(buffer.elem_offset == 0):
                raise ValueError("HMX backing buffer must have zero element offset")
            if buffer.strides:
                expected = (buffer.shape[1], 1)
                if not all(analyzer.can_prove(s == e) for s, e in zip(buffer.strides, expected)):
                    raise ValueError("HMX backing strides must be compact")
            if any(not isinstance(analyzer.simplify(x), tirx.IntImm)
                   or int(analyzer.simplify(x)) <= 0 or int(analyzer.simplify(x)) % 32
                   for x in buffer.shape):
                raise ValueError("HMX backing shape must be positive static and tile aligned")
        # Full bounds (including symbolic slices) are checked before this
        # lowering by GemmNode with the enclosing-loop analyzer. Inference's
        # analyzer lacks loop bindings. Never replace bounds with alignment.
        if not (is_shared(self.A) and is_shared(self.B)):
            raise ValueError("HMX requires shared/shared operands")
        if self.a_dtype != "float16" or self.b_dtype != "float16":
            raise ValueError("HMX currently requires fp16 operands")
        if self.trans_A or not self.trans_B:
            raise ValueError("HMX currently requires A[M,K] and B[N,K]")
        if any(int(dim) <= 0 or int(dim) % 32 for dim in (self.M, self.N, self.K)):
            raise ValueError("HMX tile extents must be multiples of 32")
        if not all(is_full_region(region) for region in (self.ARegion, self.CRegion)):
            raise ValueError("HMX currently requires full A/C regions")
        if not is_full_region(self.BRegion):
            if not is_shared(self.C):
                raise ValueError('HMX B subregions require shared output')
            analyzer=arith.Analyzer()
            for axis in self.BRegion.region:
                if not analyzer.can_prove(axis.min % 32 == 0):
                    raise ValueError('HMX B subregion must start at a tile boundary')
            if any(int(x)%32 for x in self.B.shape):
                raise ValueError('HMX backing shape must be tile aligned')

    def infer_layout(self, target, thread_nums):
        self._validate()
        workers = int(thread_nums)
        n = int(self.N)
        if is_shared(self.C):
            if self.C.dtype != "float16" or not self.clear_accum:
                raise ValueError("shared HMX output requires fp16 and clear_accum=True")
            return {self.A: _ffi_api.make_layout("ah", self.A),
                    self.B: _ffi_api.make_layout("wh", self.B),
                    self.C: _ffi_api.make_layout("ah", self.C)}
        if int(self.M) % workers == 0:
            # A cooperative worker owns complete rows, not every fourth lane.
            # Row reductions are local HVX work, never cross-worker collectives.
            return {
                self.A: _ffi_api.make_layout("ah", self.A),
                self.B: _ffi_api.make_layout("wh", self.B),
                self.C: Fragment(self.C.shape,
                    forward_thread_fn=lambda i, j: i % workers,
                    forward_index_fn=lambda i, j: (i // workers) * n + j),
            }
        return {
            self.A: _ffi_api.make_layout("ah", self.A),
            self.B: _ffi_api.make_layout("wh", self.B),
            self.C: Fragment(
                self.C.shape,
                forward_thread_fn=lambda i, j: (i * n + j) % workers,
                forward_index_fn=lambda i, j: (i * n + j) // workers,
            ),
        }

    def lower(self, layout_map, target, thread_bounds, thread_index, mbar_phase_expr=None):
        self._validate()
        for buffer, kind in ((self.A, 'ah'), (self.B, 'wh'), (self.C, 'ah')):
            if is_shared(buffer):
                expected = _ffi_api.make_layout(kind, buffer)
                if buffer not in layout_map or not structural_equal(layout_map[buffer], expected):
                    raise ValueError('HMX backing layout does not match required ' + kind)
        A, B, C = self.A, self.B, self.C
        m_tiles, n_tiles, k_tiles = int(self.M) // 32, int(self.N) // 32, int(self.K) // 32
        full_chains, tail = divmod(k_tiles, 32)
        clear_accum = self.clear_accum
        workers = int(thread_bounds.extent)
        n = int(self.N)
        clear = Op.get("tl.hexagon.hmx_clear_acc")
        mma = Op.get("tl.hexagon.hmx_mma_deep")
        unpack = Op.get("tl.hexagon.unpack_ah_tile")

        if not is_shared(C) and int(self.M) % workers == 0 and n % 64 == 0:
            # Full shared readout: issue the whole GEMM before publishing it.
            # AH -> RM leaf copies cover pairs of adjacent tiles using HVX.
            # The private row staging is vector-loaded; no scalar VTCM access.
            m = int(self.M)
            @T.prim_func
            def hmx_rows():
                tile_output = T.alloc_shared((m * n,), "float16")
                row_scales = T.alloc_shared((128,), "float16")
                row_values = T.alloc_local((n,), "float16")
                T.sync_threads()
                if thread_index == 0:
                    T.evaluate(T.call_intrin("handle", Op.get("tl.hexagon.hmx_init_scale"),
                        T.address_of(row_scales[0]), T.float16(1)))
                    for mi, ni in T.grid(m_tiles, n_tiles):
                        T.evaluate(T.call_intrin("handle", clear))
                        for ki in T.serial(full_chains):
                            T.evaluate(T.call_intrin("handle", mma,
                                T.address_of(A[mi * 32, ki * 1024]),
                                T.address_of(B[ni * 32, ki * 1024]), 32))
                        if tail:
                            T.evaluate(T.call_intrin("handle", mma,
                                T.address_of(A[mi * 32, full_chains * 1024]),
                                T.address_of(B[ni * 32, full_chains * 1024]), tail))
                        T.evaluate(T.call_intrin("handle", Op.get("tl.hexagon.hmx_store_after"),
                            T.address_of(tile_output[(mi * n_tiles + ni) * 1024]),
                            T.address_of(row_scales[0])))
                T.sync_threads()
                for ri in T.serial(m // workers):
                    T.evaluate(T.call_extern("handle", "tl::gemm_unpack_ah_row",
                        T.address_of(row_values[0]), T.address_of(tile_output[0]),
                        n, ri * workers + thread_index))
                    for col in T.vectorized(n):
                        if clear_accum:
                            C[ri * workers + thread_index, col] = T.cast(row_values[col], C.dtype)
                        else:
                            C[ri * workers + thread_index, col] += T.cast(row_values[col], C.dtype)
                T.sync_threads()
            def name_rows(node):
                if isinstance(node, tirx.SBlock):
                    for buffer in node.alloc_buffers:
                        IRBuilder.name(str(C.name) + "_" + str(buffer.name), buffer)
            tirx.stmt_functor.post_order_visit(hmx_rows.body, name_rows)
            return hmx_rows

        if is_shared(C):
            init = Op.get("tl.hexagon.hmx_init_scale")
            store = Op.get("tl.hexagon.hmx_store_after")
            br,bk = [axis.min for axis in self.BRegion.region]

            @T.prim_func
            def hmx_shared():
                scales = T.alloc_shared((128,), "float16")
                T.sync_threads()
                if thread_index == 0:
                    T.evaluate(T.call_intrin("handle", init, T.address_of(scales[0]), T.float16(1)))
                    for ni, mi in T.grid(n_tiles, m_tiles):
                        T.evaluate(T.call_intrin("handle", clear))
                        for ki in T.serial(full_chains):
                            T.evaluate(T.call_intrin("handle", mma,
                                T.address_of(A[mi * 32, ki * 1024]),
                                T.address_of(B[br + ni * 32, bk + ki * 1024]), 32))
                        if tail:
                            T.evaluate(T.call_intrin("handle", mma,
                                T.address_of(A[mi * 32, full_chains * 1024]),
                                T.address_of(B[br + ni * 32, bk + full_chains * 1024]), tail))
                        T.evaluate(T.call_intrin("handle", store,
                            T.address_of(C[mi * 32, ni * 32]), T.address_of(scales[0])))
                T.sync_threads()
            return hmx_shared

        @T.prim_func
        def hmx_gemm():
            fragment_scales = T.alloc_shared((128,), "float16")
            # Native HMX has fp16 readout from its internal 37-bit accumulator,
            # matching production attnops_gemm_nt. Casting to C's fp32 dtype
            # does not recover fp32 readout precision. TODO: separate fp32 path
            # if stronger precision than the production contract is required.
            # Physical AH tile written by asm store-after; unpack uses physical
            # storage because the macro is expanded after LayoutInference.
            readout = T.alloc_shared((1024,), "float16")
            # Compact RM scratch is in shared VTCM: the HVX unpack helper
            # requires aligned storage and writes complete 128-byte vectors.
            readout_rm = T.alloc_shared((32, 32), "float16")
            # A contiguous register-sized copy uses the generic vector load
            # lowering. Distribute from private storage, not scalar VTCM loads.
            readout_local = T.alloc_local((64,), "float16")
            T.sync_threads()
            if thread_index == 0:
                T.evaluate(T.call_intrin("handle", Op.get("tl.hexagon.hmx_init_scale"),
                    T.address_of(fragment_scales[0]), T.float16(1)))
            for mi, ni in T.grid(m_tiles, n_tiles):
                if thread_index == 0:
                    # Scale=1 was bound above; each tile starts a new
                    # accumulator session and uses asm store-after.
                    T.evaluate(T.call_intrin("handle", clear))
                    for ki in T.serial(full_chains):
                        T.evaluate(T.call_intrin("handle", mma,
                            T.address_of(A[mi * 32, ki * 1024]),
                            T.address_of(B[ni * 32, ki * 1024]), 32))
                    if tail:
                        T.evaluate(T.call_intrin("handle", mma,
                            T.address_of(A[mi * 32, full_chains * 1024]),
                            T.address_of(B[ni * 32, full_chains * 1024]), tail))
                    T.evaluate(T.call_intrin("handle", Op.get("tl.hexagon.hmx_store_after"),
                        T.address_of(readout[0]), T.address_of(fragment_scales[0])))
                    T.evaluate(T.call_intrin("handle", unpack,
                        T.address_of(readout_rm[0, 0]), T.address_of(readout[0]), 1))
                T.sync_threads()
                for pair in T.serial(16):
                    for lane in T.vectorized(64):
                        readout_local[lane] = readout_rm[pair * 2 + lane // 32, lane % 32]
                    for row in T.serial(2):
                        for column in T.serial(T.ceildiv(32, workers)):
                            i = pair * 2 + row
                            j = column * workers + (thread_index - ((mi * 32 + i) * n + ni * 32)) % workers
                            if j < 32:
                                if clear_accum:
                                    C[mi * 32 + i, ni * 32 + j] = T.cast(readout_local[row * 32 + j], C.dtype)
                                else:
                                    C[mi * 32 + i, ni * 32 + j] += T.cast(readout_local[row * 32 + j], C.dtype)
                T.sync_threads()

        def name_scratch(node):
            # Stable names disambiguate scratch from multiple fragment GEMMs;
            # shared-output GEMM (including the 8T artifact) is untouched.
            if isinstance(node, tirx.SBlock):
                for buffer in node.alloc_buffers:
                    IRBuilder.name(str(C.name) + "_" + str(buffer.name), buffer)
        tirx.stmt_functor.post_order_visit(hmx_gemm.body, name_scratch)
        return hmx_gemm
