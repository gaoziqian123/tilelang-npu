"""Execute unmodified generated kernels, not reconstructed helper calls."""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tilelang
from tilelang import language as T
from tvm.ir.transform import PassContext

ROOT = Path(__file__).resolve().parents[3]


def compile_host(tmp_path, f):
    with PassContext(config={'tl.hexagon.affine_transpose': True}):
        source = tilelang.engine.lower(f, target='hexagon', enable_host_codegen=False,
                                      enable_device_compile=False).kernel_source
    assert 'tl::transpose_2d<' in source
    (tmp_path/'kernel.cpp').write_text(source)
    # Only the runtime ABI is stubbed. Real common.h and transpose.h supply the
    # host model; all emitted addresses, offsets, strides and barriers are intact.
    (tmp_path/'runtime.cpp').write_text('''
#include <stddef.h>
#include <stdlib.h>
alignas(128) static unsigned char slab[262144];
extern "C" int tl_hex_worker_id() { return 0; }
extern "C" int tl_hex_num_workers() { return 1; }
extern "C" int tl_hex_job_id() { return 0; }
extern "C" int tl_hex_num_jobs() { return 1; }
extern "C" void tl_hex_barrier() {}
extern "C" void* tl_hex_vtcm_slice(size_t o,size_t n,size_t a) {
  if (o+n>sizeof(slab) || o%a) abort();
  return slab+o;
}
''')
    so = tmp_path/'kernel.so'
    subprocess.run(['g++', '-shared', '-fPIC', '-O2', '-std=c++17',
                    # GCC 11 has no _Float16: pure-copy storage ABI only.
                    '-D__fp16=uint16_t', '-D_Float16=uint16_t', '-I'+str(ROOT/'src'),
                    str(tmp_path/'kernel.cpp'), str(tmp_path/'runtime.cpp'),
                    '-o', str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.f_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return lib


def allocation(count, dtype):
    item = np.dtype(dtype).itemsize
    raw = np.full(count*item+512, 0xa5, np.uint8)
    off = 128+(-raw.ctypes.data)%128
    return raw, raw[off:off+count*item].view(dtype)


@pytest.mark.parametrize('dtype,bits', [('float16', np.uint16), ('float32', np.uint32)])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('layout', [False, True])
def test_generated_kernel_whole_storage(tmp_path, dtype, bits, reverse, layout):
    ss = (1, 192) if reverse else (256, 1)
    ds = (256, 1) if reverse else (1, 192)
    if layout:
        @T.prim_func
        def f(a: T.handle, b: T.handle):
            A = T.match_buffer(a, (192, 256), dtype, strides=ss)
            B = T.match_buffer(b, (192, 256), dtype, strides=ds)
            with T.Kernel(1, threads=1):
                s = T.alloc_shared((192, 256), dtype)
                T.annotate_layout({s: tilelang.layout.Layout((192, 256), lambda i, j: (j, i))})
                T.copy(A[3:68, 5:38], s[11:76, 13:46])
                T.copy(s[11:76, 13:46], B[7:72, 9:42])
    else:
        @T.prim_func
        def f(a: T.handle, b: T.handle):
            A = T.match_buffer(a, (192, 256), dtype, strides=ss)
            B = T.match_buffer(b, (192, 256), dtype, strides=ds)
            with T.Kernel(1, threads=1):
                T.copy(A[3:68, 5:38], B[7:72, 9:42])
    lib = compile_host(tmp_path, f)
    raw_s, src = allocation(192*256, bits)
    raw_d, dst = allocation(192*256, bits)
    rng = np.random.default_rng(915)
    src[:] = rng.integers(0, np.iinfo(bits).max, src.size, dtype=bits, endpoint=True)
    before = raw_s.copy()
    expected = raw_d.copy()
    offset = dst.ctypes.data-raw_d.ctypes.data
    reference = expected[offset:offset+dst.nbytes].view(bits)
    # Independent scalar logical-copy reference, with no helper/layout code.
    for i in range(65):
        for j in range(33):
            reference[(7+i)*ds[0]+(9+j)*ds[1]] = src[(3+i)*ss[0]+(5+j)*ss[1]]
    lib.f_kernel(src.ctypes.data, dst.ctypes.data)
    np.testing.assert_array_equal(raw_s, before)
    np.testing.assert_array_equal(raw_d, expected)
