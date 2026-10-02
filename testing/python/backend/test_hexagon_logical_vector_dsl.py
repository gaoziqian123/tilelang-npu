"""Full public DSL pipeline, not a hand-constructed IR module."""
from pathlib import Path
import subprocess
import pytest
import tilelang
from tilelang import language as T


@pytest.mark.parametrize('bits,n',[(16,128),(32,96)])
def test_dsl_pipeline(tmp_path,bits,n):
    dtype=f'float{bits}'
    @T.prim_func
    def kernel(A:T.Tensor((n,),dtype),B:T.Tensor((n,),dtype)):
        T.func_attr({'tl.vector_required':1,'tl.hvx_arithmetic_mode':'native_target_v1'})
        with T.Kernel(1,threads=1):
            for i in T.vectorized(n):
                B[i]=A[i]*T.cast(0.5,dtype)+T.cast(1,dtype)
    src=tilelang.engine.lower(kernel,target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert 'tl::hvx_leaf::mul' in src
    cc=Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')
    if not cc.exists(): pytest.skip('Hexagon compiler unavailable')
    root=Path(__file__).resolve().parents[3]
    source=tmp_path/'dsl.cpp'; source.write_text(src)
    cmd=[str(cc),'-mv79','-mhvx','-mhvx-length=128B','-O2','-std=c++17','-I'+str(root/'src'),'-c',str(source),'-o',str(tmp_path/'dsl.o')]
    print('COMMAND:', ' '.join(cmd))
    result=subprocess.run(cmd,text=True,capture_output=True)
    print(result.stdout+result.stderr)
    assert result.returncode==0


def test_dsl_strict_rejects():
    @T.prim_func
    def kernel(A:T.Tensor((64,),'float32'),B:T.Tensor((64,),'float32')):
        T.func_attr({'tl.vector_required':1})
        with T.Kernel(1,threads=1):
            for i in T.vectorized(64):
                B[i]=A[i]+T.float32(1)
    with pytest.raises((ValueError,RuntimeError),match='strict unavailable'):
        tilelang.engine.lower(kernel,target='hexagon',enable_host_codegen=False,enable_device_compile=False)
