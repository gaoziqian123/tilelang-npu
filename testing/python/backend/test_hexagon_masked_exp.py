"""Standard DSL masked FP32 exp: source, actual target object, host semantics.

Host execution models only the runtime/HVX primitive ABI with scalar libm. It
does not claim to validate the target exp polynomial's numerical accuracy.
"""
import ctypes
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
import tilelang
from tilelang import language as T

ROOT = Path(tilelang.__file__).resolve().parents[1]
HEX = Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')
HOST = '/root/autodl-tmp/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++'


def kernel(n, mode):
    @T.prim_func
    def f(A: T.Tensor((3, n), 'float32'), B: T.Tensor((3, n), 'float32'), bound: T.int32):
        with T.Kernel(1, threads=1):
            for chunk in T.serial(3):
                for j in T.vectorized(n):
                    if mode == 'select':
                        B[chunk, j] = T.Select(T.And(j < bound - chunk, j >= chunk), T.exp(A[chunk, j]), T.float32(0))
                    elif mode == 'lazy':
                        B[chunk, j] = T.if_then_else(T.And(j < bound - chunk, j >= chunk), T.exp(A[chunk, j]), T.float32(0))
                    else:
                        if T.And(j < bound - chunk, j >= chunk):
                            B[chunk, j] = T.exp(A[chunk, j])
                        else:
                            B[chunk, j] = T.float32(0)
    return f


def lower(f):
    return tilelang.engine.lower(f, target='hexagon', enable_host_codegen=False,
                                enable_device_compile=False).kernel_source


def actual_object(tmp_path, source):
    if not HEX.exists():
        pytest.skip('Hexagon SDK unavailable')
    cpp = tmp_path/'kernel.cpp'
    cpp.write_text(source)
    flags = [str(HEX), '-mv79', '-mhvx', '-mhvx-length=128B', '-mhmx',
             '-O2', '-std=c++17', '-I'+str(ROOT/'src')]
    subprocess.run(flags + ['-c', str(cpp), '-o', str(tmp_path/'kernel.o')], check=True)
    subprocess.run(flags + ['-S', str(cpp), '-o', str(tmp_path/'kernel.s')], check=True)
    return (tmp_path/'kernel.s').read_text()


@pytest.mark.parametrize('n', [32, 64])
@pytest.mark.parametrize('mode', ['select', 'lazy', 'control'])
def test_masked_exp_object(tmp_path, n, mode):
    source = lower(kernel(n, mode))
    assert 'tl::native_exp32(' in source
    assert 'expf(' not in source
    assert 'exp_f16' not in source
    asm = actual_object(tmp_path, source)
    assert 'vmpy' in asm and 'vmux' in asm
    # Libm in the existing cold exceptional repair is permitted, not in the
    # generated hot kernel or per-normal-lane helpers.
    hot = asm.split('f_kernel:', 1)[1].split('.Lfunc_end', 1)[0]
    assert not re.search(r'call\s+.*\bexpf?\b', hot)
    assert 'native_exp32_impl' in hot


def host_library(tmp_path, source):
    # Include shadowing stubs the runtime only: kernel.cpp is byte-identical to
    # the target input, including masks, control flow, pointers and loop bounds.
    inc = tmp_path/'tl_templates'/'hexagon'
    inc.mkdir(parents=True)
    stub = r'''
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
typedef int32_t HVX_Vector __attribute__((ext_vector_type(32)));
inline HVX_Vector Q6_V_vsplat_R(int v) { return (HVX_Vector)v; }
inline HVX_Vector Q6_V_vzero() { return (HVX_Vector)0; }
inline int tl_hex_num_jobs() { return 1; }
inline int tl_hex_job_id() { return 0; }
inline int tl_hex_num_workers() { return 1; }
inline int tl_hex_worker_id() { return 0; }
namespace tl {
inline void hex_require(bool b) { if (!b) abort(); }
template<int N> inline bool is_aligned(const void* p) { return (uintptr_t)p%N==0; }
template<class T> struct hvx_vec { HVX_Vector raw; };
template<class T> inline void hvx_store(T* p, hvx_vec<T> v) { memcpy(p,&v.raw,128); }
inline HVX_Vector native_exp32(HVX_Vector v) {
  float x[32]; memcpy(x,&v,128);
  for(int i=0;i<32;i++) x[i]=expf(x[i]);
  memcpy(&v,x,128); return v;
}
}
'''
    (tmp_path/'abi.h').write_text(stub)
    for name in ['hexagon_types.h', 'hvx_hexagon_protos.h']:
        (tmp_path/name).write_text('#include "abi.h"\n')
    for name in ['common.h', 'hvx.h', 'exp.h']:
        (inc/name).write_text('#include "abi.h"\n')
    (tmp_path/'kernel.cpp').write_text(source)
    subprocess.run([HOST, '-shared', '-fPIC', '-O2', '-std=c++17', '-Wno-psabi',
                    '-I'+str(tmp_path), str(tmp_path/'kernel.cpp'), '-o', str(tmp_path/'kernel.so')], check=True)
    lib = ctypes.CDLL(str(tmp_path/'kernel.so'))
    lib.f_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    return lib


def aligned(n):
    raw = np.full(n*4+256, 0xa5, np.uint8)
    offset = (-raw.ctypes.data)%128
    return raw, raw[offset:offset+n*4].view(np.float32)


@pytest.mark.parametrize('n', [32, 64, 33])
@pytest.mark.parametrize('mode', ['select', 'lazy', 'control'])
def test_generated_host_semantics(tmp_path, n, mode):
    lib = host_library(tmp_path, lower(kernel(n, mode)))
    raw_a, a = aligned(3*n)
    raw_b, b = aligned(3*n)
    values = np.array([np.nan, np.inf, -np.inf, 0., -0., -104., -90.,
                       -87., -0.5, 0.25, 12., 87., 89.], np.float32)
    for bound in [-1, 0, 1, 17, 31, 32, 33, 63, 64, 65]:
        for shift in range(len(values)):
            a[:] = np.resize(np.roll(values, shift), a.size)
            before_a = raw_a.copy()
            raw_b[:] = 0xa5
            expected_storage = raw_b.copy()
            expected = expected_storage[b.ctypes.data-raw_b.ctypes.data:][:b.nbytes].view(np.float32)
            # Independent logical scalar oracle; inactive nonfinite lanes must
            # produce positive zero, never multiply-by-zero NaNs.
            with np.errstate(over='ignore', invalid='ignore'):
                for row in range(3):
                    for j in range(n):
                        expected[row*n+j] = np.exp(np.float64(a[row*n+j])) if row <= j < bound-row else 0.
            lib.f_kernel(a.ctypes.data, b.ctypes.data, bound)
            np.testing.assert_array_equal(raw_a, before_a)
            np.testing.assert_allclose(b, expected, rtol=2e-5, atol=0, equal_nan=True)
            inactive = expected == 0
            np.testing.assert_array_equal(b.view(np.uint32)[inactive], np.zeros(inactive.sum(), np.uint32))
            offset = b.ctypes.data-raw_b.ctypes.data
            np.testing.assert_array_equal(raw_b[:offset], expected_storage[:offset])
            np.testing.assert_array_equal(raw_b[offset+b.nbytes:], expected_storage[offset+b.nbytes:])


def test_lazy_oob_guard_remains_scalar(tmp_path):
    @T.prim_func
    def f(A: T.Tensor((17,), 'float32'), B: T.Tensor((64,), 'float32'), bound: T.int32):
        with T.Kernel(1, threads=1):
            for j in T.vectorized(64):
                B[j] = T.if_then_else(T.And(j < 17, j < bound), T.exp(A[j]), T.float32(0))
    source = lower(f)
    assert 'tl::native_exp32(' not in source
    assert 'expf(' in source
    actual_object(tmp_path, source)
    lib = host_library(tmp_path, source)
    # Place the last valid float directly before a PROT_NONE page. Speculative
    # vector loads past A[16] then fault, rather than merely reading padding.
    import mmap
    page = mmap.PAGESIZE
    region = mmap.mmap(-1, 2*page)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    libc = ctypes.CDLL(None)
    assert libc.mprotect(ctypes.c_void_p(addr+page), page, 0) == 0
    a = np.ctypeslib.as_array((ctypes.c_float*17).from_address(addr+page-68))
    a[:] = 0
    raw, b = aligned(64)
    try:
        for bound in [0, 1, 16, 17, 64]:
            lib.f_kernel(a.ctypes.data, b.ctypes.data, bound)
            np.testing.assert_array_equal(b, np.arange(64) < min(bound, 17))
    finally:
        assert libc.mprotect(ctypes.c_void_p(addr+page), page, 3) == 0
        region.close()
