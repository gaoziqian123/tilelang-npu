from pathlib import Path
import subprocess
import pytest
import tilelang
from tilelang import language as T
from tilelang.hexagon.language.dma import submit,wait


def kernel(case):
    @T.prim_func
    def dma(A:T.Tensor((64,),'float16'),B:T.Tensor((64,),'float16')):
        with T.Kernel(1,threads=1):
            T.evaluate(submit(0,T.address_of(B[0]),T.address_of(A[0]),128,1,128,128))
            if case == 'duplicate':
                T.evaluate(submit(0,T.address_of(B[0]),T.address_of(A[0]),128,1,128,128))
            if case == 'read':
                B[0]=A[0]
            T.evaluate(wait(0))
            if case == 'stale':
                T.evaluate(wait(0))
    return dma


@pytest.mark.parametrize('case',['duplicate','read','stale'])
def test_negative(case):
    with pytest.raises(ValueError,match='DMA'):
        tilelang.engine.lower(kernel(case),target='hexagon',enable_host_codegen=False,enable_device_compile=False)


def test_emit(tmp_path):
    src=tilelang.engine.lower(kernel('ok'),target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert 'tl_hex_dma_copy_2d_submit' in src and 'tl_hex_dma_wait' in src
    assert '__builtin_trap' in src
    cpp=tmp_path/'dma.cpp'; cpp.write_text(src)
    root=Path(__file__).resolve().parents[3]
    cmd=['/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++','-mv79','-mhvx','-mhvx-length=128B','-O2','-std=c++17','-I'+str(root/'src'),'-I/root/project/backend/npu/attn/skel/src','-c',str(cpp),'-o',str(tmp_path/'dma.o')]
    print('COMMAND:', ' '.join(cmd))
    r=subprocess.run(cmd,text=True,capture_output=True); print(r.stdout+r.stderr)
    assert r.returncode==0
