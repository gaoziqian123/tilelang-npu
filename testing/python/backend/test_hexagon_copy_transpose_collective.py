import tilelang
from tilelang import language as T
from tvm.ir.transform import PassContext


def test_collective_owner_and_barriers():
    @T.prim_func
    def f(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (64, 64), 'float32')
        B = T.match_buffer(b, (64, 64), 'float32', strides=(1, 64))
        with T.Kernel(1, threads=2):
            T.copy(A, B)
    with PassContext(config={'tl.hexagon.affine_transpose': True}):
        source = tilelang.engine.lower(f, target='hexagon', enable_host_codegen=False,
                                      enable_device_compile=False).kernel_source
    assert source.count('tl_hex_barrier();') == 2
    assert 'if (tx == 0)' in source
    assert source.count('tl::transpose_2d<4>') == 1
