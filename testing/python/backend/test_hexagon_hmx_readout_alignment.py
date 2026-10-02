"""HMX crouton readout alignment, independent of any operator DSL/name."""
import re
import pytest
import tilelang
from tilelang import language as T


@pytest.mark.parametrize('m,n,k', [(32,32,32),(32,64,128),(64,64,128),(128,128,64)])
def test_fragment_readout_crouton_alignment(m,n,k):
    @T.prim_func
    def kernel(A:T.Tensor((m,k),'float16'), B:T.Tensor((n,k),'float16'),
               O:T.Tensor((m,n),'float16')):
        with T.Kernel(1,threads=1):
            a=T.alloc_shared((m,k),'float16')
            b=T.alloc_shared((n,k),'float16')
            c=T.alloc_fragment((m,n),'float16')
            T.copy(A,a)
            T.copy(B,b)
            T.gemm(a,b,c,transpose_B=True,clear_accum=True)
            T.copy(c,O)
    src=tilelang.engine.lower(kernel,target='hexagon',enable_host_codegen=False,
                              enable_device_compile=False).kernel_source
    offsets={name:int(offset) for name,offset in re.findall(
        r'void\* (\w+) = .*?buf_dyn_shmem \+ (\d+)\)',src)}
    outputs=re.findall(r'hmx_store_after_f16\(\(&\(\(\(__fp16\*\)(\w+)\)',src)
    assert outputs,src
    for name in outputs:
        assert offsets[name]%2048==0,(name,offsets[name],src)
