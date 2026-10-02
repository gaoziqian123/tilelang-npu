import importlib.util
from pathlib import Path
import tilelang
import ctypes
import subprocess
import numpy as np
from tilelang import language as T
from tvm import IRModule, instrument
from tvm.ir.transform import PassContext
from tvm.target import Target

def make(extra=False, carry=False, dependent=False, escape=False, fence=False):
    @T.prim_func
    def kernel(A:T.Tensor((2,64),'float32'), B:T.Tensor((2,64),'float16'), D:T.Tensor((2,),'float32')):
        with T.Kernel(1,threads=1):
            f=T.alloc_local((2,64),'float32')
            h=T.alloc_local((2,64),'float16')
            d=T.alloc_local((2,),'float32')
            T.copy(A,f)
            T.copy(D,d)
            T.copy(B,h)
            if escape:
                T.evaluate(T.call_extern('handle','escape',T.address_of(h[0,0])))
            for k in T.serial(3):
                if carry:
                    D[0]=T.cast(h[0,0],'float32')
                for r in T.serial(2):
                    if dependent:
                        f[r,0]=f[r,0]+1
                    else:
                        d[r]=d[r]+1
                    for j in T.vectorized(64):
                        h[r,j]=T.cast(f[r,j],'float16')
                if fence:
                    T.evaluate(T.call_extern('handle','fence'))
                T.copy(h,B)
                if extra:
                    D[0]=T.cast(h[0,0],'float32')
            T.copy(d,D)
    return kernel

def test_forward_and_counterexamples():
    for extra,carry,dependent in [(False,False,False),(True,False,False),(False,True,False),(False,False,True)]:
        f=make(extra,carry,dependent)
        result=str(tilelang.transform.FuseCastCopy()(IRModule({'main':f})))
        hit='T.region(f[0, 0], 1, 2, 64), T.region(B' in result
        assert hit == (not(extra or carry or dependent)), result
        if not dependent:
            assert 'd[r] = d[r] + T.float32(1.0)' in result

def test_escape_and_fence():
    for f in [make(escape=True),make(fence=True)]:
        result=str(tilelang.transform.FuseCastCopy()(IRModule({'main':f})))
        assert 'h[r, j] = T.Cast("float16", f[r, j])' in result

def test_host_compiled_semantics(tmp_path):
    # Standard C target uses the same native pass, independent of Hexagon
    # helpers. Disable vectorization because host GCC lacks ext_vector_type.
    f=make()
    fused=tilelang.transform.FuseCastCopy()(IRModule({'main':f}))['main']
    libs=[]
    for name,func in [('base',f),('fused',fused)]:
        with Target('c'), PassContext(config={'tirx.disable_vectorize':True}):
            src=tilelang.engine.lower(func,target='c',enable_host_codegen=False,
                                     enable_device_compile=False).kernel_source
        cpp=tmp_path/(name+'.cpp'); cpp.write_text(src)
        so=tmp_path/(name+'.so')
        root=Path(__file__).resolve().parents[3]
        subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I',str(root/'src'),
                        str(cpp),'-o',str(so)],check=True)
        lib=ctypes.CDLL(str(so)); libs.append(lib)
    rng=np.random.default_rng(1)
    for exceptional in [None, np.nan, np.inf, -np.inf, -0., 2**-25, 1+2**-11]:
        a=rng.normal(size=(2,64)).astype('float32')
        if exceptional is not None: a[:,::3]=exceptional
        outputs=[]
        for lib in libs:
            b=np.zeros((2,64),dtype='float16'); d=np.array([1.,-0.],dtype='float32')
            fn=lib.kernel_kernel; fn.argtypes=[ctypes.c_void_p]*3
            fn(a.ctypes.data,b.ctypes.data,d.ctypes.data)
            outputs.append((b.view('uint16').copy(),d.view('uint32').copy()))
            np.testing.assert_array_equal(d,np.array([4.,3.],dtype='float32'))
        for a,b in zip(*outputs): np.testing.assert_array_equal(a,b)

@instrument.pass_instrument
class Capture:
    def __init__(self): self.before=self.after=''
    def run_before_pass(self,mod,info):
        if info.name=='tl.FuseCastCopy': self.before=str(mod)
    def run_after_pass(self,mod,info):
        if info.name=='tl.FuseCastCopy': self.after=str(mod)

def test_real_fa(tmp_path):
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('persistent',root/'examples/hexagon/fa/fa_persistent.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cap=Capture()
    with PassContext(config={'tl.enable_fuse_cast_copy':True,'tl.hexagon.typed_layout_copy':True},instruments=[cap]):
        src=tilelang.engine.lower(module.make_fa(),target='hexagon',enable_host_codegen=False,
                                  enable_device_compile=False).kernel_source
    (tmp_path/'before.tir').write_text(cap.before)
    (tmp_path/'after.tir').write_text(cap.after)
    (tmp_path/'fa.cpp').write_text(src)
    assert 'T.region(rowf[0, 0], 1, 2, 256), T.region(p[' in cap.after
    assert cap.after.count('denominator[')==cap.before.count('denominator[')
    assert 'rowh[r, j] = T.Cast("float16", rowf[r, j])' not in cap.after
    assert 'cast_pack_f32_f16' in src
    import subprocess
    clang='/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++'
    subprocess.run([clang,'-mv79','-mhvx','-mhvx-length=128B','-mhmx','-O2','-std=c++17',
                    '-I',str(root/'src'),'-c',str(tmp_path/'fa.cpp'),'-o',str(tmp_path/'fa.o')],check=True)
    subprocess.run([clang,'-mv79','-mhvx','-mhvx-length=128B','-mhmx','-O2','-g','-std=c++17',
                    '-I',str(root/'src'),'-S',str(tmp_path/'fa.cpp'),'-o',str(tmp_path/'fa.s')],check=True)
    asm=(tmp_path/'fa.s').read_text(encoding='latin1')
    assert 'cast_layout.h' in asm and 'vmem(' in asm
