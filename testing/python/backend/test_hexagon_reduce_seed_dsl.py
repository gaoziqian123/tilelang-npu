from pathlib import Path
import subprocess
import pytest
import tilelang
from tilelang import language as T
from tvm import tirx as tir


def reduction(A, maximum, n, seed):
    k=tir.IterVar((0,n),'k',tir.IterVar.CommReduce)
    a,b=tir.Var('a','float32'),tir.Var('b','float32')
    comb=tir.CommReducer([a],[b],[tir.Max(a,b) if maximum else a+b],
                        [tir.const(float('-inf') if maximum else 0,'float32')])
    return tir.Reduce(comb,[tir.Cast('float32',A[k.var])],[k],tir.const(True,'bool'),0,[tir.const(seed,'float32')])


@pytest.mark.parametrize('dtype,n,maximum',[('float16',128,False),('float16',128,True),('float32',96,False),('float32',96,True)])
def test_reduce_target(tmp_path,dtype,n,maximum):
    @T.prim_func
    def kernel(A:T.Tensor((n,),dtype),B:T.Tensor((32,),'float32')):
        T.func_attr({'tl.vector_required':1,'tl.hvx_arithmetic_mode':'native_target_v1',
                     'tl.hvx_reduce_order':'chunks_asc_tree_asc_seed_post_v1'})
        with T.Kernel(1,threads=1):
            value=reduction(A,maximum,n,2.5)
            for i in T.vectorized(32):
                B[i]=value
    src=tilelang.engine.lower(kernel,target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert ('max32_asc' if maximum else 'sum32_asc') in src
    cpp=tmp_path/'reduce.cpp';cpp.write_text(src)
    root=Path(__file__).resolve().parents[3]
    cmd=['/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++','-mv79','-mhvx','-mhvx-length=128B','-O2','-std=c++17','-I'+str(root/'src'),'-c',str(cpp),'-o',str(tmp_path/'reduce.o')]
    print('COMMAND:', ' '.join(cmd))
    r=subprocess.run(cmd,text=True,capture_output=True);print(r.stdout+r.stderr)
    assert r.returncode==0
