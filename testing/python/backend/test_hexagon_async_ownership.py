import pytest
import tilelang
from tilelang import language as T
from tilelang.hexagon.language.async_scope import submit, wait
from tvm import IRModule, get_global_func


def case(mode):
    @T.prim_func
    def f(X:T.Tensor((32,32),'float16'), Z:T.Tensor((32,32),'float16')):
        with T.Kernel(1,threads=1):
            a=T.alloc_shared((32,32),'float16')
            b=T.alloc_shared((32,32),'float16')
            c=T.alloc_shared((32,32),'float16')
            d=T.alloc_shared((32,32),'float16')
            T.copy(X,a)
            T.copy(X,b)
            with submit(0):
                T.gemm(a,b,c,transpose_B=True,clear_accum=True)
            if mode == 1:
                T.copy(c,Z)
            if mode == 2:
                T.copy(X,a)
            if mode == 3:
                with submit(0):
                    T.gemm(a,b,d,transpose_B=True,clear_accum=True)
            if mode != 4:
                wait(0)
            if mode == 5:
                wait(0)
            if mode == 6:
                T.gemm(a,b,d,transpose_B=True,clear_accum=True)
            if mode == 0:
                T.copy(c,Z)
    return f


@pytest.mark.parametrize('mode',[1,2,3,4,5,6])
def test_reject_lifetime(mode):
    with pytest.raises(Exception,match='async'):
        tilelang.engine.lower(case(mode),target='hexagon',
            enable_host_codegen=False,enable_device_compile=False)


def test_retired_token_allows_consumer():
    src=tilelang.engine.lower(case(0),target='hexagon',
        enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert 'engine_submit<0>' in src and 'engine_wait' in src
