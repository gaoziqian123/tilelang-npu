"""Extern dependency registration: real objects, no forced header include."""
import os
from pathlib import Path
import subprocess
import pytest
import tilelang
from tilelang import language as T

@pytest.mark.parametrize('columns', [32, 64, 128])
@pytest.mark.parametrize('op', ['gemm_unpack_ah_row', 'gemm_unpack_ah_rows', 'gemm_pack_ah_rows'])
def test_row_extern_header(tmp_path, columns, op):
    clang=Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')
    if not clang.exists(): pytest.skip('Hexagon compiler unavailable')
    @T.prim_func
    def row_copy(A:T.Tensor((32,columns),'float16'), B:T.Tensor((32,columns),'float16')):
        with T.Kernel(1,threads=1):
            if op == 'gemm_unpack_ah_row':
                T.evaluate(T.call_extern('int32','tl::'+op,T.address_of(B[0,0]),T.address_of(A[0,0]),columns,0))
            else:
                T.evaluate(T.call_extern('int32','tl::'+op,T.address_of(B[0,0]),T.address_of(A[0,0]),columns,0,1))
    src=tilelang.engine.lower(row_copy,target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert '#include <tl_templates/hexagon/gemm.h>' in src
    cpp=tmp_path/'row.cpp'; cpp.write_text(src)
    root=Path(__file__).resolve().parents[3]
    subprocess.run([str(clang),'-mv79','-mhvx','-mhvx-length=128B','-mhmx','-O2','-std=c++17','-I'+str(root/'src'),'-c',str(cpp),'-o',str(tmp_path/'row.o')],check=True)
