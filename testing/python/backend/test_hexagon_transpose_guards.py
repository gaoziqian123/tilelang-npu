import pytest
import tilelang
from tilelang import language as T
from tvm.ir.transform import PassContext

def lower(f, enabled):
    with PassContext(config={'tl.hexagon.affine_transpose':enabled}):
        return tilelang.engine.lower(f,target='hexagon',enable_host_codegen=False,
                                     enable_device_compile=False).kernel_source

def test_alias_rejected():
    @T.prim_func
    def f(A:T.Tensor((64,64),'float16')):
        with T.Kernel(1,threads=1):
            T.transpose(A,A)
    with pytest.raises(Exception,match='cannot alias'):
        lower(f,True)

def test_unpadded_vtcm_rejected():
    @T.prim_func
    def f(A:T.Tensor((33,33),'float16')):
        with T.Kernel(1,threads=1):
            b=T.alloc_shared((33,33),'float16')
            T.transpose(A,b)
            T.evaluate(T.call_extern('handle','consume',T.address_of(b[0,0])))
    with pytest.raises(Exception,match='requires proven bounds'):
        lower(f,True)

def test_optout_and_region():
    @T.prim_func
    def f(A:T.Tensor((128,128),'float32'),B:T.Tensor((128,128),'float32')):
        with T.Kernel(1,threads=1):
            T.transpose(A[1:66,3:34],B[2:33,5:70])
    assert 'transpose_2d' not in lower(f,False)
    assert 'transpose_2d<4>' in lower(f,True)
