"""Conservative applicability and worker-scope checks."""
import pytest
import tilelang
from tilelang import language as T
from tvm.ir.transform import PassContext


def lower(f):
    with PassContext(config={'tl.hexagon.affine_transpose': True}):
        return tilelang.engine.lower(f, target='hexagon', enable_host_codegen=False,
                                     enable_device_compile=False).kernel_source


def test_unknown_dimension():
    @T.prim_func
    def f(a: T.handle, b: T.handle, n: T.int32):
        A = T.match_buffer(a, (n, 64), 'float32')
        B = T.match_buffer(b, (n, 64), 'float32', strides=(1, n))
        with T.Kernel(1, threads=1):
            T.copy(A, B)
    assert 'transpose_2d' not in lower(f)


def test_nonzero_offset():
    @T.prim_func
    def f(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (64, 64), 'float32', elem_offset=32)
        B = T.match_buffer(b, (64, 64), 'float32', strides=(1, 64))
        with T.Kernel(1, threads=1):
            T.copy(A, B)
    assert 'transpose_2d' not in lower(f)


def test_unknown_layout():
    @T.prim_func
    def f(A: T.Tensor((64, 64), 'float32'), B: T.Tensor((64, 64), 'float32')):
        with T.Kernel(1, threads=1):
            s = T.alloc_local((64, 64), 'float32')
            T.annotate_layout({s: tilelang.layout.Layout((64, 64), lambda i, j: (i, j ^ 1))})
            T.copy(A, s)
            T.copy(s, B)
    assert 'transpose_2d' not in lower(f)


def test_zero_fill_predicate():
    @T.prim_func
    def f(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (32, 64), 'float32')
        B = T.match_buffer(b, (64, 64), 'float32', strides=(1, 64))
        with T.Kernel(1, threads=1):
            T.copy(A[0:64, 0:64], B)
    source = lower(f)
    assert 'transpose_2d' not in source
    assert '32' in source


def test_unpadded_shared_rejected():
    @T.prim_func
    def f(A: T.Tensor((33, 33), 'float32'), B: T.Tensor((33, 33), 'float32')):
        with T.Kernel(1, threads=1):
            s = T.alloc_shared((33, 33), 'float32')
            T.annotate_layout({s: tilelang.layout.Layout((33, 33), lambda i, j: (j, i))})
            T.copy(A, s)
            T.copy(s, B)
    with pytest.raises(Exception, match='requires proven bounds'):
        lower(f)


def test_multiworker_local_rejected():
    @T.prim_func
    def f(A: T.Tensor((64, 64), 'float32'), B: T.Tensor((64, 64), 'float32')):
        with T.Kernel(1, threads=2):
            s = T.alloc_local((64, 64), 'float32')
            T.annotate_layout({s: tilelang.layout.Layout((64, 64), lambda i, j: (j, i))})
            T.copy(A, s)
            T.copy(s, B)
    with pytest.raises(Exception, match='requires a single worker'):
        lower(f)
