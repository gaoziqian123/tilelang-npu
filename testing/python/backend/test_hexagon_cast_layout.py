"""Typed layout copy: independent numeric reference and real target compilation."""
import ctypes
from pathlib import Path
import subprocess
import numpy as np
import pytest
import tilelang
from tilelang import language as T
from tilelang.hexagon import _ffi_api
from tvm.ir.transform import PassContext

ROOT = Path(__file__).resolve().parents[3]
CLANG = Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')

@pytest.mark.parametrize('wh', [False, True])
@pytest.mark.parametrize('columns', [32, 64, 96, 256])
def test_numeric_and_object(tmp_path, wh, columns):
    cpp = tmp_path/'copy.cpp'
    cpp.write_text('#include <tl_templates/hexagon/cast_layout.h>\n'
        'extern "C" void run(void*d,const float*s,int n,int stride,int r,int c,int h,int w){'
        f'tl::cast_pack_f32_f16<{str(wh).lower()}>(d,s,n,stride,r,c,h,w);}}\n')
    so = tmp_path/'copy.so'
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I',str(ROOT/'src'),str(cpp),'-o',str(so)],check=True)
    lib = ctypes.CDLL(str(so))
    lib.run.argtypes = [ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*6
    rng = np.random.default_rng(42)
    # Random bit patterns exercise the full exponent range, not just normals.
    x = rng.integers(0,2**32,(64,columns+7),dtype=np.uint32).view(np.float32)
    special = np.array([0.,-0.,np.inf,-np.inf,np.nan,2**-25,3*2**-25,
                        1+2**-11,1+3*2**-11,65504.,65520.],dtype=np.float32)
    x[0,:len(special)] = special
    raw = np.full(64*columns*2+256,0xa5,dtype=np.uint8)
    offset = (-raw.ctypes.data)%128
    out = raw[offset:offset+64*columns*2].view(np.uint16)
    ref = out.copy()
    row,col,h,w = (32,2,32,columns-2) if wh else (2,0,34,columns)
    lib.run(out.ctypes.data,x.ctypes.data,columns,columns+7,row,col,h,w)
    with np.errstate(over='ignore',invalid='ignore'):
        halves = x[:h,:w].astype(np.float16).view(np.uint16)
    for r in range(h):
        for c in range(w):
            i,j = row+r,col+c
            tile = (i//32)*(columns//32)+j//32
            lane = (j%32//2)*64+(i%32)*2+j%2 if wh else (i%32//2)*64+(j%32)*2+i%2
            v = halves[r,c]
            if np.isnan(x[r,c]): v |= 0x200
            ref[tile*1024+lane] = v
    np.testing.assert_array_equal(out,ref)
    assert np.all(raw[:offset]==0xa5) and np.all(raw[offset+out.nbytes:]==0xa5)
    if not CLANG.exists(): pytest.fail('Hexagon compiler required')
    subprocess.run([str(CLANG),'-mv79','-mhvx','-mhvx-length=128B','-O2','-std=c++17',
                    '-I',str(ROOT/'src'),'-c',str(cpp),'-o',str(tmp_path/'copy.o')],check=True)

def make(mode='ah', rows=2, col=0, width=64, dtype='float32', alignment=128, scope='shared.dyn'):
    @T.prim_func
    def kernel(A:T.Tensor((32,64),dtype)):
        with T.Kernel(1,threads=1):
            a=T.alloc_local((32,64),dtype)
            b=T.sblock_alloc_buffer((32,64),'float16',scope=scope,align=alignment)
            T.annotate_layout({b:_ffi_api.make_layout(mode,b)})
            T.copy(A,a)
            T.copy(a[0:rows,0:width],b[0:rows,col:col+width])
            T.evaluate(T.call_extern('handle','consume',T.address_of(b[0,0])))
    return kernel

def lower(f, enabled):
    with PassContext(config={'tl.hexagon.typed_layout_copy':enabled}):
        return tilelang.engine.lower(f,target='hexagon',enable_host_codegen=False,
                                     enable_device_compile=False).kernel_source

@pytest.mark.parametrize('mode,rows,col,width', [('ah',2,0,64),('ah',4,32,32),('wh',32,2,62)])
def test_lowering(mode,rows,col,width):
    f=make(mode,rows,col,width)
    assert 'cast_pack_f32_f16' in lower(f,True)
    assert 'cast_pack_f32_f16' not in lower(f,False)

@pytest.mark.parametrize('mode,rows,col,width,dtype', [
    ('ah',3,0,64,'float32'),('ah',2,1,32,'float32'),
    ('wh',31,0,64,'float32'),('wh',32,1,62,'float32'),
    ('ah',2,0,64,'int32')])
def test_decline(mode,rows,col,width,dtype):
    assert 'cast_pack_f32_f16' not in lower(make(mode,rows,col,width,dtype),True)

def test_unknown_alignment_declines():
    assert 'cast_pack_f32_f16' not in lower(make(alignment=64,scope='shared'),True)
