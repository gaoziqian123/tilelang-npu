"""Bitwise checks compare the SAME row tree, never a sequential sum reference."""
import ctypes
import importlib.util
from pathlib import Path
import subprocess
import numpy as np
import pytest
import tilelang
from tilelang import language as T
from tvm import instrument
from tvm.ir.transform import PassContext

ROOT=Path(__file__).resolve().parents[3]

@instrument.pass_instrument
class Capture:
    def __init__(self): self.before=self.after=""
    def run_after_pass(self,mod,info):
        if info.name=="tl.VectorizeLoop": self.before=str(mod)
        if info.name=="tl.FuseLocalRowMap": self.after=str(mod)

def lower(f,fuse=True,vector=False):
    c=Capture()
    with PassContext(config={"tl.hexagon.plan_local_row_reduce":True,
                             "tl.hexagon.fuse_local_row_map":fuse,
                             "tirx.disable_vectorize":not vector},instruments=[c]):
        src=tilelang.engine.lower(f,target="hexagon",enable_host_codegen=False,
                                 enable_device_compile=False).kernel_source
    return src,c

def host(tmp,src,tag):
    header=tmp/"tl_templates/hexagon/common.h"
    header.parent.mkdir(parents=True,exist_ok=True)
    header.write_text('#include <cstddef>\nnamespace tl {inline void hex_require(bool x){if(!x)__builtin_trap();}}\n'
        'inline int tl_hex_worker_id(){return 0;} inline int tl_hex_job_id(){return 0;}\n'
        'inline int tl_hex_num_workers(){return 1;} inline int tl_hex_num_jobs(){return 1;}\n'
        'inline void boundary(){}\n')
    cpp=tmp/(tag+".cpp"); so=tmp/(tag+".so"); cpp.write_text(src)
    subprocess.run(["g++","-std=c++17","-O2","-fPIC","-shared","-I",str(tmp),
                    "-I",str(ROOT/"src"),str(cpp),"-o",str(so)],check=True,capture_output=True)
    lib=ctypes.CDLL(str(so)); lib.kernel_kernel.argtypes=[ctypes.c_void_p]*3
    return lib

def make(n,kind,mapkind,blocked=False,stencil=False,escape=False):
    @T.prim_func
    def kernel(A:T.Tensor((2,n),"float32"), B:T.Tensor((2,),"float32"), C:T.Tensor((2,n),"float32")):
        with T.Kernel(1,threads=1):
            a=T.alloc_local((2,n),"float32")
            b=T.alloc_local((2,),"float32")
            for r in T.serial(2):
                b[r]=B[r]
                for j in T.serial(n):
                    a[r,j]=A[r,j]
            for r in T.serial(2):
                for j in T.serial(n):
                    if stencil:
                        a[r,j]=a[r,(j+1)%n]
                    elif mapkind==0:
                        a[r,j]=T.exp(a[r,j])
                    elif mapkind==1:
                        a[r,j]=T.Select(a[r,j]>0,a[r,j]*2+1,a[r,j]-3)
                    else:
                        a[r,j]=T.cast(T.cast(a[r,j],"float64")*0.25,"float32")
            if blocked:
                T.evaluate(T.call_extern("handle","boundary"))
            if escape:
                T.evaluate(T.call_extern("handle","escape",T.address_of(a[0,0])))
            T.reduce(a,b,kind,dim=1,clear=False)
            for r in T.serial(2):
                B[r]=b[r]
                for j in T.serial(n):
                    C[r,j]=T.cast(T.cast(a[r,j],"float64"),"float32")
    return kernel

@pytest.mark.parametrize("n",[31,32,33,65,128])
@pytest.mark.parametrize("kind",["sum","max"])
@pytest.mark.parametrize("mapkind",[0,1,2])
def test_host_bitwise(tmp_path,n,kind,mapkind):
    f=make(n,kind,mapkind)
    base,_=lower(f,False); fused,c=lower(f)
    assert "row_map_update" in c.after and "row_map_value" in c.after
    assert "row_map_finish" in fused
    if mapkind==0:
        assert base.count("expf(")==fused.count("expf(")
    a=host(tmp_path,base,"base"); b=host(tmp_path,fused,"fused")
    rng=np.random.default_rng(81)
    for exceptional in [None,np.nan,np.inf,-np.inf,-0.]:
        x=rng.uniform(-2,2,(2,n)).astype("float32")
        if exceptional is not None: x[:,n//2]=exceptional
        for seed in [0.,-0.,-17.,np.inf,-np.inf,np.nan]:
            y=np.full(2,seed,dtype="float32"); z=y.copy()
            u=np.zeros_like(x); v=u.copy()
            a.kernel_kernel(x.ctypes.data,y.ctypes.data,u.ctypes.data)
            b.kernel_kernel(x.ctypes.data,z.ctypes.data,v.ctypes.data)
            np.testing.assert_array_equal(y.view("uint32"),z.view("uint32"))
            np.testing.assert_array_equal(u.view("uint32"),v.view("uint32"))

@pytest.mark.parametrize("blocked,stencil",[(True,False),(False,True)])
def test_effect_and_crosslane_rejected(blocked,stencil):
    _,c=lower(make(65,"sum",1,blocked,stencil))
    assert "row_map_update" not in c.after

def test_escaped_alias_rejected():
    _,c=lower(make(65,"sum",1,escape=True))
    assert "row_map_update" not in c.after

@pytest.mark.parametrize("complete",[True,False])
def test_dynamic_partition(tmp_path,complete):
    n=96
    @T.prim_func
    def kernel(A:T.Tensor((2,n),"float32"),B:T.Tensor((2,),"float32"),C:T.Tensor((2,n),"float32")):
        with T.Kernel(1,threads=1):
            a=T.alloc_local((2,n),"float32")
            b=T.alloc_local((2,),"float32")
            for r in T.serial(2):
                for j in T.serial(n):
                    a[r,j]=A[r,j]
            for r in T.serial(2):
                for j in T.serial(T.min(n,T.max(0,T.cast(B[r],"int32")))):
                    a[r,j]=T.exp(a[r,j])
                if complete:
                    for j in T.serial(T.min(n,T.max(0,T.cast(B[r],"int32"))),n):
                        a[r,j]=0.
            T.reduce_sum(a,b,dim=1)
            for r in T.serial(2):
                B[r]=b[r]
                for j in T.serial(n): C[r,j]=a[r,j]
    _,c=lower(kernel)
    # Mutable-memory bounds are deliberately rejected even if ranges appear
    # complementary. Only immutable scalar-domain partition proofs are used.
    assert "row_map_update" not in c.after

@pytest.mark.parametrize("maximum",[False,True])
def test_incremental_leaf_exact(tmp_path,maximum):
    cpp=tmp_path/"leaf.cpp"; so=tmp_path/"leaf.so"
    flag=str(maximum).lower()
    cpp.write_text('#include <tl_templates/hexagon/row_map_reduce.h>\n'
        'extern "C" float run(const float* p,int n,float seed,int step){'
        'alignas(128) float s[64]; '
        f'tl::row_map_init<{flag}>(s);'
        'for(int i=0;i<n;i+=step){int count=n-i<step?n-i:step;'
        f'tl::row_map_update<{flag}>(s,p+i,i,count,n);}}'
        f'return tl::row_map_finish<{flag}>(s,p,n,seed);}}\n'
        'extern "C" float ref(const float* p,int n,float seed){'
        f'return tl::row_reduce_f32<{flag}>(p,n,seed);}}\n')
    subprocess.run(["g++","-std=c++17","-O2","-shared","-fPIC","-I",str(ROOT/"src"),
        str(cpp),"-o",str(so)],check=True,capture_output=True)
    lib=ctypes.CDLL(str(so)); lib.run.restype=lib.ref.restype=ctypes.c_float
    lib.ref.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_float]
    lib.run.argtypes=lib.ref.argtypes+[ctypes.c_int]
    rng=np.random.default_rng(90)
    for n in [0,1,31,32,33,63,65,128,257]:
        for value in [None,np.nan,np.inf,-np.inf,-0.]:
            p=rng.normal(size=n).astype("float32")
            if value is not None and n: p[n//2]=value
            if value==0.: p[:]=-0.
            for seed in [0.,-0.,-17.,np.inf,-np.inf,np.nan]:
                ref=np.float32(lib.ref(p.ctypes.data,n,seed))
                for step in [1,32]:
                    got=np.float32(lib.run(p.ctypes.data,n,seed,step))
                    assert got.view("uint32")==ref.view("uint32")

@pytest.mark.parametrize("kind",["sum","max"])
@pytest.mark.parametrize("half",[False,True])
def test_vector_map_cast_consumer_object(tmp_path,kind,half):
    n=96
    @T.prim_func
    def kernel(A:T.Tensor((2,n),"float32"),B:T.Tensor((2,),"float32"),C:T.Tensor((2,n),"float16")):
        with T.Kernel(1,threads=1):
            a=T.alloc_local((2,n),"float32")
            b=T.alloc_local((2,),"float32")
            for r in T.serial(2):
                b[r]=B[r]
                for j in T.vectorized(n): a[r,j]=A[r,j]
            for r in T.serial(2):
                for j in T.vectorized(n):
                    if half:
                        a[r,j]=T.cast(T.cast(a[r,j]*0.5,"float16"),"float32")
                    else:
                        a[r,j]=T.Select(a[r,j]>0,T.exp(a[r,j]-1),a[r,j]*2+1)
            T.reduce(a,b,kind,dim=1,clear=False)
            for r in T.serial(2):
                B[r]=b[r]
                for j in T.vectorized(n): C[r,j]=T.cast(a[r,j],"float16")
    src,c=lower(kernel,vector=True)
    assert "row_map_update" in c.after
    assert "row_map_value: T.float32x32" in c.after
    assert "float16" in c.after
    cc=Path("/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++")
    if not cc.exists(): pytest.skip("Hexagon compiler unavailable")
    cpp=tmp_path/"map.cpp"; cpp.write_text(src)
    subprocess.run([str(cc),"-mv79","-mhvx","-mhmx","-mhvx-length=128B","-O2",
        "-std=c++17","-I",str(ROOT/"src"),"-c",str(cpp),"-o",str(tmp_path/"map.o")],
        check=True,capture_output=True)

@pytest.mark.parametrize("profile",[True,False])
def test_real_persistent(profile,tmp_path):
    spec=importlib.util.spec_from_file_location("fusion_fa",ROOT/"examples/hexagon/fa/fa_persistent.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    src,c=lower(m.make_fa(instrumentation=profile),vector=True)
    assert ("tl::profile_region" in c.after)==profile
    assert ("row_map_update" in c.after)==(not profile)
    if not profile:
        assert c.before.count("hexagon.local_row_reduce")==2
        assert c.after.count("hexagon.local_row_reduce")==0
        assert c.after.count("hexagon.row_map_update")==3
        assert c.after.count("hexagon.row_map_finish")==2
        assert c.before.count("T.exp(")==c.after.count("T.exp(")
    cc=Path("/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++")
    if not cc.exists(): pytest.skip("Hexagon compiler unavailable")
    cpp=tmp_path/"fa.cpp"; cpp.write_text(src)
    subprocess.run([str(cc),"-mv79","-mhvx","-mhmx","-mhvx-length=128B","-O2",
        "-std=c++17","-I",str(ROOT/"src"),"-c",str(cpp),"-o",str(tmp_path/"fa.o")],
        check=True,capture_output=True)

@pytest.mark.parametrize('kind,mask,expected',[('max',2,True),('sum',2,False),('max',1,False),('sum',1,True)])
@pytest.mark.parametrize('n',[32,64,256])
def test_terminal_select_after_full_map(kind,mask,expected,n):
    @T.prim_func
    def kernel(A:T.Tensor((2,n),'float16'), B:T.Tensor((2,),'float32')):
        with T.Kernel(1,threads=1):
            a=T.alloc_local((2,n),'float32')
            b=T.alloc_local((2,),'float32')
            for r in T.serial(2):
                for j in T.vectorized(n):
                    a[r,j]=T.cast(A[r,j],'float32')*0.5
                for j in T.vectorized(n):
                    a[r,j]=T.Select(j<r+17,a[r,j],-T.infinity('float32'))
            T.reduce(a,b,kind,dim=1,clear=True)
            for r in T.serial(2): B[r]=b[r]
    c=Capture()
    with PassContext(config={'tl.hexagon.plan_local_row_reduce':True,
                            'tl.hexagon.fuse_local_row_map':True,
                            'tl.hexagon.row_map_reducers':mask},instruments=[c]):
        tilelang.engine.lower(kernel,target='hexagon',enable_host_codegen=False,enable_device_compile=False)
    assert ('hexagon.row_map_update' in c.after)==expected
    assert c.before.count('T.Select(')==c.after.count('T.Select(')


@pytest.mark.parametrize("n", [31, 32, 33, 65, 128])
@pytest.mark.parametrize("kind", ["sum", "max"])
def test_terminal_multistage_bitwise(tmp_path, n, kind):
    """Keep both materialized stages and compare the unchanged reduction tree."""
    @T.prim_func
    def kernel(A:T.Tensor((2,n),"float32"), B:T.Tensor((2,),"float32"), C:T.Tensor((2,n),"float32")):
        with T.Kernel(1,threads=1):
            a=T.alloc_local((2,n),"float32")
            b=T.alloc_local((2,),"float32")
            for r in T.serial(2):
                b[r]=B[r]
            for r in T.serial(2):
                for j in T.serial(n):
                    a[r,j]=A[r,j]*0.5
                for j in T.serial(n):
                    a[r,j]=T.Select(j<r+17,a[r,j],-T.infinity("float32"))
            T.reduce(a,b,kind,dim=1,clear=False)
            for r in T.serial(2):
                B[r]=b[r]
                for j in T.serial(n):
                    C[r,j]=a[r,j]
    base,_=lower(kernel,False)
    fused,c=lower(kernel)
    assert "row_map_update" in c.after
    original=host(tmp_path,base,"original")
    candidate=host(tmp_path,fused,"candidate")
    rng=np.random.default_rng(193)
    for exceptional in [None,np.nan,np.inf,-np.inf,-0.]:
        x=rng.uniform(-2,2,(2,n)).astype("float32")
        if exceptional is not None:
            x[:,7]=exceptional  # Unmasked lane, including every tail length.
        for seed in [0.,-0.,-17.,np.inf,-np.inf,np.nan]:
            y=np.full(2,seed,dtype="float32"); z=y.copy()
            u=np.full_like(x,np.nan); v=u.copy()
            original.kernel_kernel(x.ctypes.data,y.ctypes.data,u.ctypes.data)
            candidate.kernel_kernel(x.ctypes.data,z.ctypes.data,v.ctypes.data)
            np.testing.assert_array_equal(y.view("uint32"),z.view("uint32"))
            np.testing.assert_array_equal(u.view("uint32"),v.view("uint32"))
