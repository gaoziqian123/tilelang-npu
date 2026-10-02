import pytest
import tilelang
from tilelang import language as T
from tvm import tirx as tir


@T.macro
def reduced(A,B):
    k=tir.IterVar((0,64),'k',tir.IterVar.CommReduce)
    lhs=tir.Var('lhs','float32')
    rhs=tir.Var('rhs','float32')
    comb=tir.CommReducer([lhs],[rhs],[lhs+rhs],[tir.const(0,'float32')])
    value=tir.Reduce(comb,[A[k.var]],[k],tir.const(True,'bool'),0)
    T.evaluate(0)
    for i in T.vectorized(32):
        B[i]=value


def test_reduce_dsl():
    @T.prim_func
    def kernel(A:T.Tensor((64,),'float32'),B:T.Tensor((32,),'float32')):
        T.func_attr({'tl.vector_required':1,'tl.hvx_arithmetic_mode':'native_target_v1',
                     'tl.hvx_reduce_order':'chunks_asc_tree_asc_seed_post_v1'})
        with T.Kernel(1,threads=1):
            reduced(A,B)
    src=tilelang.engine.lower(kernel,target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert 'sum32_asc' in src
