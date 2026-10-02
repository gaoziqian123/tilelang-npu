"""Explicit HMX live-accumulator sum of two products.

This is NOT fp32 T.gemm accumulation: fp16 operands, hardware live accumulator,
one fp16 materialization. No intermediate rounded product is observable.
Operands must be distinct from output; caller stages complete valid panels.
"""
from tilelang import language as T
from tilelang.hexagon import _ffi_api
from tvm.ir import Op


def _check_buffers(buffers,shapes):
    for b,shape in zip(buffers,shapes):
        if b.dtype!="float16" or b.scope() not in ("shared","shared.dyn"):
            raise ValueError("live HMX scope requires shared fp16 buffers, not fp32 accumulation")
        if tuple(int(v) for v in b.shape)!=shape:
            raise ValueError("live HMX scope requires full, exactly shaped operand panels")


def transpose_packed(m,n):
    """AH[M,N] -> WH[N,M], per-tile vector transpose, disjoint buffers."""
    @T.macro
    def transpose(A,B):
        T.sync_threads()
        for tile in T.serial(T.ceildiv((m//32)*(n//32),T.get_thread_extent())):
            index=tile*T.get_thread_extent()+T.get_thread_binding()
            if index<(m//32)*(n//32):
                T.evaluate(T.call_extern("handle","tl::gemm_transpose_packed_tile",
                    T.address_of(B[(index%(n//32))*32,(index//(n//32))*32]),
                    T.address_of(A[(index//(n//32))*32,(index%(n//32))*32])))
        T.sync_threads()
    return transpose


def product_sum_f16(m, n, k0, k1):
    if any(x <= 0 or x % 32 for x in (m,n,k0,k1)):
        raise ValueError("HMX dimensions must be positive multiples of 32")

    @T.macro
    def product_sum(A, B, X, Y, C):
        _check_buffers((A,B,X,Y,C),((m,k0),(n,k0),(m,k1),(n,k1),(m,n)))
        if any(C.same_as(v) for v in (A,B,X,Y)):
            raise ValueError("product_sum output must not alias an operand")
        scales=T.alloc_shared((128,),"float16")
        T.sync_threads()
        if T.get_thread_binding()==0:
            T.evaluate(T.call_intrin("handle",Op.get("tl.hexagon.hmx_init_scale"),T.address_of(scales[0]),T.float16(1)))
            for ni,mi in T.grid(n//32,m//32):
                T.evaluate(T.call_intrin("handle",Op.get("tl.hexagon.hmx_clear_acc")))
                for ki in T.serial(k0//32):
                    T.evaluate(T.call_intrin("handle",Op.get("tl.hexagon.hmx_mma_deep"),
                        T.address_of(A[mi*32,ki*32]),T.address_of(B[ni*32,ki*32]),1))
                for ki in T.serial(k1//32):
                    T.evaluate(T.call_intrin("handle",Op.get("tl.hexagon.hmx_mma_deep"),
                        T.address_of(X[mi*32,ki*32]),T.address_of(Y[ni*32,ki*32]),1))
                T.evaluate(T.call_intrin("handle",Op.get("tl.hexagon.hmx_store_after"),
                    T.address_of(C[mi*32,ni*32]),T.address_of(scales[0])))
        T.sync_threads()
    return product_sum


def block_diagonal_product_sum_f16(m,n,k):
    """C=blockdiag(D)@C+X@Y.T, live accumulator then one fp16 store.

    D[M,32] explicitly stores M/32 dense diagonal blocks (not arbitrary sparse
    inputs). C AH tile is already the WH encoding of its logical transpose;
    each output tile reads only its own old C tile, so in-place is safe.
    Caller annotates D/X/C AH and Y WH. No zero padding or identity reload.
    """
    if any(v<=0 or v%32 for v in (m,n,k)):
        raise ValueError("HMX dimensions must be positive multiples of 32")
    @T.macro
    def update(D,X,Y,C):
        _check_buffers((D,X,Y,C),((m,32),(m,k),(n,k),(m,n)))
        if any(C.same_as(v) for v in (D,X,Y)):
            raise ValueError("block update only permits explicit old-C alias")
        scales=T.alloc_shared((128,),"float16")
        T.sync_threads()
        if T.get_thread_binding()==0:
            T.evaluate(T.call_intrin("handle",Op.get("tl.hexagon.hmx_init_scale"),T.address_of(scales[0]),T.float16(1)))
            for ni,mi in T.grid(n//32,m//32):
                T.evaluate(T.call_intrin("handle",Op.get("tl.hexagon.hmx_clear_acc")))
                T.evaluate(T.call_intrin("handle",Op.get("tl.hexagon.hmx_mma_deep"),
                    T.address_of(D[mi*32,0]),T.address_of(C[mi*32,ni*32]),1))
                for ki in T.serial(k//32):
                    T.evaluate(T.call_intrin("handle",Op.get("tl.hexagon.hmx_mma_deep"),
                        T.address_of(X[mi*32,ki*32]),T.address_of(Y[ni*32,ki*32]),1))
                T.evaluate(T.call_intrin("handle",Op.get("tl.hexagon.hmx_store_after"),
                    T.address_of(C[mi*32,ni*32]),T.address_of(scales[0])))
        T.sync_threads()
    return update
