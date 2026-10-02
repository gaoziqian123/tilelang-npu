from pathlib import Path
import subprocess
import pytest
import tilelang
from tilelang import language as T


@pytest.mark.parametrize('op,dtype,mode',[('exp','float32','hvx_math32_v1'),('exp2','float16','ggml_qf16_clamp24_v1'),('div','float32','nr2_positive_bounded_v1')])
def test_explicit_math(tmp_path,op,dtype,mode):
    @T.prim_func
    def kernel(A:T.Tensor((128,),dtype),B:T.Tensor((128,),dtype)):
        T.func_attr({'tl.vector_required':1,'tl.hvx_'+op+'_mode':mode})
        with T.Kernel(1,threads=1):
            for i in T.vectorized(128):
                if op=='exp':
                    B[i]=T.exp(A[i])
                elif op=='exp2':
                    B[i]=T.exp2(A[i])
                else:
                    B[i]=T.cast(1,dtype)/A[i]
    src=tilelang.engine.lower(kernel,target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
    cpp=tmp_path/'math.cpp';cpp.write_text(src)
    root=Path(__file__).resolve().parents[3]
    cmd=['/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++','-mv79','-mhvx','-mhvx-length=128B','-O2','-ffp-contract=off','-std=c++17','-I'+str(root/'src'),'-c',str(cpp),'-o',str(tmp_path/'math.o')]
    print('COMMAND:', ' '.join(cmd))
    r=subprocess.run(cmd,text=True,capture_output=True);print(r.stdout+r.stderr)
    assert r.returncode==0
