"""Bit-exact memory permutation; no floating-point reference conversions."""
import ctypes
import subprocess
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
CLANG = Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')

@pytest.mark.parametrize('item', [2, 4])
def test_bits_and_object(tmp_path, item):
    cpp = tmp_path / 'transpose.cpp'
    cpp.write_text('#include <tl_templates/hexagon/transpose.h>\n'
        'extern "C" bool run(void*d,const void*s,size_t db,size_t sb,int ds,int ss,'
        'int dr,int dc,int sr,int sc,int h,int w){'
        f'return tl::transpose_2d<{item}>(d,s,db,sb,ds,ss,dr,dc,sr,sc,h,w);}}\n')
    so = tmp_path / 'transpose.so'
    subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-I'+str(ROOT/'src'),str(cpp),'-o',str(so)],check=True)
    lib = ctypes.CDLL(str(so))
    lib.run.argtypes = [ctypes.c_void_p]*2+[ctypes.c_size_t]*2+[ctypes.c_int]*8
    lib.run.restype = ctypes.c_bool
    def allocation():
        raw = np.full(256*256*item+256,0xa5,np.uint8)
        offset = (-raw.ctypes.data)%128
        return raw, raw[offset:offset+256*256*item].view(np.uint16 if item==2 else np.uint32).reshape(256,256)
    rng = np.random.default_rng(93)
    for h,w in [(64,128),(128,128),(1,1),(31,65),(65,33),(127,129)]:
        for sr,sc,dr,dc in [(0,0,0,0),(1,3,2,5),(2,63,1,31)]:
            raw_s,s = allocation(); raw_d,d = allocation()
            s[:] = rng.integers(0,2**(item*8),s.shape,dtype=s.dtype)
            if item==2:
                s.flat[:65536] = np.arange(65536,dtype=np.uint16) # every half encoding
            else:
                s[sr,sc:sc+min(w,4)] = np.array([0,0x80000000,0x7fc12345,0x7f812345],np.uint32)[:min(w,4)]
            before = raw_d.copy(); expected = d.copy()
            expected[dr:dr+w,dc:dc+h] = s[sr:sr+h,sc:sc+w].T
            assert lib.run(d.ctypes.data,s.ctypes.data,d.nbytes,s.nbytes,256,256,dr,dc,sr,sc,h,w)
            np.testing.assert_array_equal(d,expected)
            offset = d.ctypes.data-raw_d.ctypes.data
            np.testing.assert_array_equal(raw_d[:offset],before[:offset])
            np.testing.assert_array_equal(raw_d[offset+d.nbytes:],before[offset+d.nbytes:])
            saved = raw_s.copy()
            assert not lib.run(s.ctypes.data,s.ctypes.data,s.nbytes,s.nbytes,256,256,0,0,0,0,h,w)
            np.testing.assert_array_equal(raw_s,saved)
            saved = raw_d.copy()
            assert not lib.run(d.ctypes.data+item,s.ctypes.data,d.nbytes,s.nbytes,256,256,0,0,0,0,h,w)
            assert not lib.run(d.ctypes.data,s.ctypes.data,d.nbytes,s.nbytes,256,256,0,255,0,0,2,2)
            np.testing.assert_array_equal(raw_d,saved)
    assert CLANG.exists()
    flags = [str(CLANG),'-mv79','-mhvx','-mhvx-length=128B','-O2','-std=c++17','-I'+str(ROOT/'src')]
    subprocess.run(flags+['-c',str(cpp),'-o',str(tmp_path/'transpose.o')],check=True)
    subprocess.run(flags+['-S',str(cpp),'-o',str(tmp_path/'transpose.s')],check=True)
    asm = (tmp_path/'transpose.s').read_text()
    assert 'vshuff' in asm and 'vmem' in asm
    assert 'memuh(' not in asm and 'memh(' not in asm

@pytest.mark.parametrize('dtype', ['float16','float32'])
def test_dsl(tmp_path, dtype):
    import tilelang
    from tilelang import language as T
    from tvm.ir.transform import PassContext
    @T.prim_func
    def kernel(A:T.Tensor((128,128),dtype), B:T.Tensor((128,128),dtype)):
        with T.Kernel(1,threads=1):
            T.transpose(A,B)
    with PassContext(config={'tl.hexagon.affine_transpose':True}):
        src = tilelang.engine.lower(kernel,target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert 'tl::transpose_2d<' in src
    cpp = tmp_path/'generated.cpp'; cpp.write_text(src)
    subprocess.run([str(CLANG),'-mv79','-mhvx','-mhvx-length=128B','-O2','-std=c++17',
                    '-I'+str(ROOT/'src'),'-c',str(cpp),'-o',str(tmp_path/'generated.o')],check=True)

@pytest.mark.parametrize('item', [2, 4])
def test_asymmetric_strides_and_rejection(tmp_path, item):
    cpp = tmp_path/'guard.cpp'
    cpp.write_text('#include <tl_templates/hexagon/transpose.h>\n'
        'extern "C" bool run(void*d,const void*s,size_t db,size_t sb,int ds,int ss,'
        'int dr,int dc,int sr,int sc,int h,int w){'
        f'return tl::transpose_2d<{item}>(d,s,db,sb,ds,ss,dr,dc,sr,sc,h,w);}}\n')
    so = tmp_path/'guard.so'
    subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-I'+str(ROOT/'src'),str(cpp),'-o',str(so)],check=True)
    lib = ctypes.CDLL(str(so))
    lib.run.argtypes = [ctypes.c_void_p]*2+[ctypes.c_size_t]*2+[ctypes.c_int]*8
    lib.run.restype = ctypes.c_bool
    dtype = np.uint16 if item == 2 else np.uint32
    def allocate(stride):
        raw = np.full(256*stride*item+256,0xa5,np.uint8)
        off = (-raw.ctypes.data)%128
        return raw, raw[off:off+256*stride*item].view(dtype).reshape(256,stride)
    for ss,ds in [(197,211),(211,197)]:
        raw_s,s = allocate(ss); raw_d,d = allocate(ds)
        bits = ([0,0x8000,0x7c00,0xfc00,0x7c01,0x7e35,1,0x3c00] if item==2 else
                [0,0x80000000,0x7f800000,0xff800000,0x7f800001,0x7fc12345,1,0x3f800000])
        s[:] = np.resize(np.array(bits,dtype=dtype),s.shape)
        for h,w in [(1,1),(31,65),(65,33),(127,129)]:
            sr,sc,dr,dc = 1,3,2,5
            expected_raw = raw_d.copy()
            off = d.ctypes.data-raw_d.ctypes.data
            expected = expected_raw[off:off+d.nbytes].view(dtype).reshape(d.shape)
            expected[dr:dr+w,dc:dc+h] = s[sr:sr+h,sc:sc+w].T
            source_before = raw_s.copy()
            args = [d.ctypes.data,s.ctypes.data,d.nbytes,s.nbytes,ds,ss,dr,dc,sr,sc,h,w]
            assert lib.run(*args)
            # Entire backing storage: exterior guards, row gaps and tail lanes.
            np.testing.assert_array_equal(raw_d,expected_raw)
            np.testing.assert_array_equal(raw_s,source_before)
            for index,value in [(0,None),(1,None),(2,d.nbytes-1),(3,s.nbytes-1),
                                (2,0),(3,0),(4,0),(5,0),(10,0),(11,0)]:
                invalid = args.copy(); invalid[index] = value
                before = raw_d.copy()
                assert not lib.run(*invalid)
                np.testing.assert_array_equal(raw_d,before)
                np.testing.assert_array_equal(raw_s,source_before)
