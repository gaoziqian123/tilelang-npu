"""Permutation layouts must use remapped storage exactly once."""
import subprocess
from pathlib import Path
import pytest
import tilelang
from tilelang import language as T
from tvm.ir.transform import PassContext

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_permutation_layout(tmp_path, dtype):
    @T.prim_func
    def f(A: T.Tensor((64, 128), dtype), B: T.Tensor((64, 128), dtype)):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((64, 128), dtype)
            T.annotate_layout({s: tilelang.layout.Layout((64, 128), lambda i, j: (j, i))})
            T.copy(A, s)
            T.copy(s, B)

    with PassContext(config={'tl.hexagon.affine_transpose': True}):
        source = tilelang.engine.lower(f, target='hexagon', enable_host_codegen=False,
                                      enable_device_compile=False).kernel_source
    assert source.count('tl::transpose_2d<') == 2
    cpp = tmp_path/'layout.cpp'
    cpp.write_text(source)
    subprocess.run(['/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++',
                    '-mv79', '-mhvx', '-mhvx-length=128B', '-O2', '-std=c++17',
                    '-I'+str(ROOT/'src'), '-c', str(cpp), '-o', str(tmp_path/'layout.o')], check=True)
