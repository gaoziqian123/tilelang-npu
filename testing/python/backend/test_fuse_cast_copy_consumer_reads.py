"""Consumer descriptor reads must survive cast forwarding, including aliases."""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tilelang
from tilelang import language as T
from tilelang.transform import PassConfigKey
from tvm import IRModule, tirx
from tvm.ir.transform import PassContext
from tvm.target import Target


def make(kind):
    @T.prim_func
    def kernel(A: T.Tensor((2, 64), 'float32'), B: T.Tensor((4, 64), 'float16')):
        with T.Kernel(1, threads=1):
            f = T.alloc_local((2, 64), 'float32')
            h = T.alloc_local((2, 64), 'float16')
            T.copy(A, f)
            for r in T.serial(2):
                for j in T.vectorized(64):
                    h[r, j] = T.cast(f[r, j], 'float16')
            if kind == 'offset':
                T.copy(h, B[T.min(T.max(T.cast(h[0, 0], 'int32'), 0), 2):
                            T.min(T.max(T.cast(h[0, 0], 'int32'), 0), 2)+2, :])
            elif kind == 'extent':
                T.copy(h, B[0:2, 0:T.min(T.max(T.cast(h[0, 0], 'int32'), 1), 2)*32])
            elif kind == 'indirect':
                T.copy(h, B[T.min(T.max(T.cast(h[0, T.min(T.max(T.cast(h[0, 1], 'int32'), 0), 63)], 'int32'), 0), 2):
                            T.min(T.max(T.cast(h[0, T.min(T.max(T.cast(h[0, 1], 'int32'), 0), 63)], 'int32'), 0), 2)+2, :])
            elif kind == 'select':
                T.copy(h, B[T.Select(h[0, 0] > 0, 2, 0):T.Select(h[0, 0] > 0, 2, 0)+2, :])
            elif kind == 'source_extent':
                T.copy(h[0:2, 0:T.Select(h[0, 0] > 0, 64, 32)], B[0:2, :])
            elif kind == 'source_offset':
                T.copy(h[T.min(T.max(T.cast(h[0, 0], 'int32'), 0), 1):2, :], B[0:2, :])
            else:
                T.copy(h, B[0:2, :])
    return kernel


def alias_offset(f):
    """Same data Var, distinct Buffer in a destination offset expression."""
    def rewrite(node):
        if not isinstance(node, tirx.Call) or node.op.name != 'tl.tileop.copy':
            return None
        src, dst = node.args
        anchor = src.args[0]
        if anchor.buffer.dtype != 'float16':
            return None
        h = anchor.buffer
        view = tirx.decl_buffer(h.shape, h.dtype, name='extra_view', data=h.data)
        offset = tirx.Min(tirx.Max(tirx.Cast('int32', tirx.BufferLoad(view, [0, 0])), 0), 2)
        args = list(dst.args)
        args[0] = tirx.BufferLoad(args[0].buffer, [offset, 0])
        dst = tirx.Call(dst.dtype, dst.op, args)
        return tirx.Call(node.dtype, node.op, [src, dst])
    return f.with_body(tirx.stmt_functor.ir_transform(f.body, None, rewrite, ['tirx.Call']))


CASES = ['offset', 'extent', 'indirect', 'select', 'source_extent', 'source_offset', 'extra_view']


@pytest.mark.parametrize('kind', CASES)
def test_consumer_dependency_host(tmp_path, kind):
    base = alias_offset(make('plain')) if kind == 'extra_view' else make(kind)
    fused = tilelang.transform.FuseCastCopy()(IRModule({'main': base}))['main']
    assert 'h[r, j] = T.Cast("float16", f[r, j])' in str(fused)
    libs = []
    root = Path(__file__).resolve().parents[3]
    for name, func in [('base', base), ('fused', fused)]:
        with Target('c'), PassContext(config={'tirx.disable_vectorize': True}):
            src = tilelang.engine.lower(func, target='c', enable_host_codegen=False,
                                       enable_device_compile=False).kernel_source
        cpp, so = tmp_path/(name+'.cpp'), tmp_path/(name+'.so')
        cpp.write_text(src)
        subprocess.run(['g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-I',
                        str(root/'src'), str(cpp), '-o', str(so)], check=True)
        libs.append(ctypes.CDLL(str(so)))
    for value in [0., 1., 2.]:
        a = np.arange(128, dtype='float32').reshape(2, 64)/128
        a[0, 0], a[0, 1] = value, 0.
        outputs = []
        for lib in libs:
            b = np.full((4, 64), -7., dtype='float16')
            fn = lib.kernel_kernel
            fn.argtypes = [ctypes.c_void_p]*2
            fn(a.ctypes.data, b.ctypes.data)
            outputs.append(b.copy())
        np.testing.assert_array_equal(outputs[0].view('uint16'), outputs[1].view('uint16'))
        if kind in ['offset', 'extra_view']:
            expected = np.full((4, 64), -7., dtype='float16')
            expected[int(value):int(value)+2] = a.astype('float16')
            np.testing.assert_array_equal(outputs[1], expected)


@pytest.mark.parametrize('field', ['shape', 'stride', 'elem_offset', 'source_select'])
def test_source_metadata_rejected(field):
    def rewrite(node):
        if not isinstance(node, tirx.Call) or node.op.name != 'tl.tileop.copy':
            return None
        src, dst = node.args
        h = src.args[0].buffer
        if h.dtype != 'float16':
            return None
        dep = tirx.Cast('int32', tirx.BufferLoad(h, [0, 0]))
        shape, strides, offset = list(h.shape), [], 0
        if field == 'shape': shape[0] = tirx.Select(dep > 0, 2, 2)
        if field == 'stride': strides = [64, tirx.Select(dep > 0, 1, 1)]
        if field == 'elem_offset': offset = tirx.Select(dep > 0, 0, 0)
        args = list(src.args)
        if field == 'source_select':
            args[2] = tirx.Select(dep > 0, 2, 2)
        else:
            view = tirx.decl_buffer(shape, h.dtype, data=h.data, strides=strides,
                                   elem_offset=offset, name='metadata_view')
            args[0] = tirx.BufferLoad(view, [0, 0])
        return tirx.Call(node.dtype, node.op, [tirx.Call(src.dtype, src.op, args), dst])
    f = make('plain')
    f = f.with_body(tirx.stmt_functor.ir_transform(f.body, None, rewrite, ['tirx.Call']))
    fused = tilelang.transform.FuseCastCopy()(IRModule({'main': f}))
    assert 'h[r, j] = T.Cast("float16", f[r, j])' in str(fused)


def test_pass_config_keys():
    assert PassConfigKey.TL_ENABLE_FUSE_CAST_COPY.value == 'tl.enable_fuse_cast_copy'
    assert PassConfigKey.TL_HEXAGON_TYPED_LAYOUT_COPY.value == 'tl.hexagon.typed_layout_copy'
    with PassContext(config={PassConfigKey.TL_ENABLE_FUSE_CAST_COPY: True,
                             PassConfigKey.TL_HEXAGON_TYPED_LAYOUT_COPY: True}):
        assert callable(tilelang.transform.FuseCastCopy())
