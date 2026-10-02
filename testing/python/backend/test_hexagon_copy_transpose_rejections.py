"""Unproven copies must not select the physical transpose helper."""
import pytest
import tilelang
from tilelang import language as T
from tvm.ir.transform import PassContext


def lower(f):
    with PassContext(config={'tl.hexagon.affine_transpose': True}):
        return tilelang.engine.lower(f, target='hexagon', enable_host_codegen=False,
                                     enable_device_compile=False).kernel_source


@pytest.mark.parametrize('case', ['cast', 'oob', 'unpadded', 'negative'])
def test_not_applicable(case):
    dtype = 'float32' if case == 'cast' else 'float16'
    shape = (33, 33) if case == 'unpadded' else (64, 64)
    stride = (1, -64) if case == 'negative' else (1, shape[0])
    extent = 65 if case == 'oob' else shape[0]

    @T.prim_func
    def f(a: T.handle, b: T.handle):
        A = T.match_buffer(a, shape, 'float16')
        B = T.match_buffer(b, shape, dtype, strides=stride)
        with T.Kernel(1, threads=1):
            T.copy(A[0:extent, 0:shape[1]], B[0:extent, 0:shape[1]])

    assert 'transpose_2d' not in lower(f)


def test_alias_rejected():
    @T.prim_func
    def f(a: T.handle):
        A = T.match_buffer(a, (64, 64), 'float16')
        with T.Kernel(1, threads=1):
            B = T.decl_buffer((64, 64), 'float16', data=A.data, strides=(1, 64))
            T.copy(A, B)
    with pytest.raises(Exception, match='cannot alias'):
        lower(f)
