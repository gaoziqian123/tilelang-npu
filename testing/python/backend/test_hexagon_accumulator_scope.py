"""Explicit live hardware accumulation is distinct from rounded fp16 products."""
import numpy as np
import pytest
import tilelang
from tilelang import language as T
from tilelang.hexagon import _ffi_api
from tilelang.hexagon.language.accumulator import product_sum_f16,block_diagonal_product_sum_f16


@pytest.mark.parametrize("m,n,k0,k1",[(32,64,32,64),(64,128,64,96),(128,256,128,256)])
def test_product_sum_lifecycle(m,n,k0,k1):
    fused=product_sum_f16(m,n,k0,k1)
    @T.prim_func
    def f(A:T.Tensor((m,k0),"float16"),B:T.Tensor((n,k0),"float16"),
          X:T.Tensor((m,k1),"float16"),Y:T.Tensor((n,k1),"float16"),C:T.Tensor((m,n),"float16")):
        with T.Kernel(1,threads=4):
            a=T.alloc_shared((m,k0),"float16")
            b=T.alloc_shared((n,k0),"float16")
            x=T.alloc_shared((m,k1),"float16")
            y=T.alloc_shared((n,k1),"float16")
            c=T.alloc_shared((m,n),"float16")
            T.annotate_layout({a:_ffi_api.make_layout("ah",a),b:_ffi_api.make_layout("wh",b),
                x:_ffi_api.make_layout("ah",x),y:_ffi_api.make_layout("wh",y),c:_ffi_api.make_layout("ah",c)})
            T.copy(A,a);T.copy(B,b);T.copy(X,x);T.copy(Y,y)
            fused(a,b,x,y,c)
            T.copy(c,C)
    src=tilelang.engine.lower(f,target="hexagon",enable_host_codegen=False,enable_device_compile=False).kernel_source
    start=src.index("tl::hmx_clear_f16")
    end=src.index("tl::hmx_store_after_f16",start)
    assert src[start:end].count("hmx_mma_deep_f16")==2
    assert "hmx_acc_read" not in src
    assert src.count("hmx_store_after_f16")==1
    # Full independent algebra reference distinguishes single vs double rounding.
    rng=np.random.default_rng(701)
    a=rng.normal(size=(m,k0)).astype(np.float16).astype(np.float64)
    b=rng.normal(size=(n,k0)).astype(np.float16).astype(np.float64)
    x=rng.normal(size=(m,k1)).astype(np.float16).astype(np.float64)
    y=rng.normal(size=(n,k1)).astype(np.float16).astype(np.float64)
    ref=(a@b.T+x@y.T).astype(np.float16)
    packed=(np.concatenate((a,x),axis=1)@np.concatenate((b,y),axis=1).T).astype(np.float16)
    np.testing.assert_array_equal(ref,packed)
    rounded=((a@b.T).astype(np.float16).astype(np.float64)+(x@y.T).astype(np.float16)).astype(np.float16)
    assert np.any(ref!=rounded)


def test_reject_incomplete_panels():
    with pytest.raises(ValueError):product_sum_f16(32,64,31,64)


@pytest.mark.parametrize("m,n,k",[(32,64,32),(64,128,96),(128,256,256)])
def test_inplace_block_diagonal_lifecycle(m,n,k):
    update=block_diagonal_product_sum_f16(m,n,k)
    @T.prim_func
    def f(D:T.Tensor((m,32),"float16"),X:T.Tensor((m,k),"float16"),
          Y:T.Tensor((n,k),"float16"),C:T.Tensor((m,n),"float16")):
        with T.Kernel(1,threads=4):
            d=T.alloc_shared((m,32),"float16")
            x=T.alloc_shared((m,k),"float16")
            y=T.alloc_shared((n,k),"float16")
            c=T.alloc_shared((m,n),"float16")
            T.annotate_layout({d:_ffi_api.make_layout("ah",d),x:_ffi_api.make_layout("ah",x),
                y:_ffi_api.make_layout("wh",y),c:_ffi_api.make_layout("ah",c)})
            T.copy(D,d);T.copy(X,x);T.copy(Y,y);T.copy(C,c)
            update(d,x,y,c)
            T.copy(c,C)
    src=tilelang.engine.lower(f,target="hexagon",enable_host_codegen=False,enable_device_compile=False).kernel_source
    a=src.index("tl::hmx_clear_f16");b=src.index("tl::hmx_store_after_f16",a)
    assert src[a:b].count("hmx_mma_deep_f16")==2
    assert src.count("hmx_store_after_f16")==1
    assert "transpose_packed_tile" not in src
    rng=np.random.default_rng(702)
    d=rng.normal(size=(m,32)).astype(np.float16).astype(np.float64)
    c=rng.normal(size=(m,n)).astype(np.float16).astype(np.float64)
    x=rng.normal(size=(m,k)).astype(np.float16).astype(np.float64)
    y=rng.normal(size=(n,k)).astype(np.float16).astype(np.float64)
    dense=np.zeros((m,m))
    for i in range(m//32):dense[i*32:(i+1)*32,i*32:(i+1)*32]=d[i*32:(i+1)*32]
    ref=(dense@c+x@y.T).astype(np.float16)
    got=np.empty((m,n),np.float16)
    for i in range(0,m,32):
        for j in range(0,n,32):
            got[i:i+32,j:j+32]=(d[i:i+32]@c[i:i+32,j:j+32]+x[i:i+32]@y[j:j+32].T).astype(np.float16)
    np.testing.assert_array_equal(got,ref)


def test_ah_to_transposed_wh_same_tile_bytes():
    for r in range(32):
        for c in range(32):
            ah=(r//2)*64+c*2+r%2
            wh_transpose=(r//2)*64+c*2+r%2
            assert ah==wh_transpose
