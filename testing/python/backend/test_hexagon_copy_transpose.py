"""Physical-address transpose recognition through standard strided buffers."""
import subprocess
from pathlib import Path

import pytest
import tilelang
from tilelang import language as T
from tvm.ir.transform import PassContext

ROOT = Path(__file__).resolve().parents[3]
CLANG = Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')


def lower(f):
    with PassContext(config={'tl.hexagon.affine_transpose': True}):
        return tilelang.engine.lower(f, target='hexagon', enable_host_codegen=False,
                                     enable_device_compile=False).kernel_source


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
@pytest.mark.parametrize('reverse', [False, True])
def test_strided_view(tmp_path, dtype, reverse):
    ss = (1, 192) if reverse else (256, 1)
    ds = (256, 1) if reverse else (1, 192)

    @T.prim_func
    def f(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (192, 256), dtype, strides=ss)
        B = T.match_buffer(b, (192, 256), dtype, strides=ds)
        with T.Kernel(1, threads=1):
            T.copy(A[3:68, 5:38], B[7:72, 9:42])

    source = lower(f)
    assert 'tl::transpose_2d<' in source
    cpp = tmp_path / 'generated.cpp'
    cpp.write_text(source)
    subprocess.run([str(CLANG), '-mv79', '-mhvx', '-mhvx-length=128B', '-O2',
                    '-std=c++17', '-I'+str(ROOT/'src'), '-c', str(cpp),
                    '-o', str(tmp_path/'generated.o')], check=True)


def test_identity_not_transpose():
    @T.prim_func
    def f(A: T.Tensor((64, 128), 'float16'), B: T.Tensor((64, 128), 'float16')):
        with T.Kernel(1, threads=1):
            T.copy(A, B)
    assert 'transpose_2d' not in lower(f)
