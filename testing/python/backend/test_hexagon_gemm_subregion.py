import pytest
import tilelang
from tilelang import language as T
from pathlib import Path
import subprocess
from types import SimpleNamespace
from tvm import tirx
from tilelang.hexagon.op.gemm.gemm_hmx import GemmHMX


def lower(f):
    return tilelang.engine.lower(f, target='hexagon', enable_host_codegen=False,
                                enable_device_compile=False).kernel_source


def region_case(n=32, k=64, row=32, col=0, bn=64, bk=64, dynamic=0):
    @T.prim_func
    def f(X:T.Tensor((32,k),'float16'),Y:T.Tensor((bn,bk),'float16'),
          Z:T.Tensor((32,n),'float16'),v:T.int32):
        with T.Kernel(1,threads=1):
            a=T.alloc_shared((32,k),'float16')
            b=T.alloc_shared((bn,bk),'float16')
            c=T.alloc_shared((32,n),'float16')
            T.copy(X,a)
            T.copy(Y,b)
            T.gemm(a,b[row+dynamic*v:row+dynamic*v+n,col:col+k],c,transpose_B=True,clear_accum=True)
            T.copy(c,Z)
    return f


@pytest.mark.parametrize('kw', [dict(row=32,n=64), dict(col=32),
    dict(row=-32), dict(col=-32), dict(row=0,dynamic=32), dict(row=0,dynamic=1)])
def test_invalid_region_bounds(kw):
    with pytest.raises(Exception, match='region bounds|tile boundary'):
        lower(region_case(**kw))


@pytest.mark.parametrize('axis', [0, 1])
def test_single_iteration_negative_region_not_wrapped(axis):
    # Regression: legalization used to wrap -32 to +32 before the strong
    # region proof, because the Python precheck did not bind the loop variable.
    @T.prim_func
    def f(X:T.Tensor((32,32),'float16'),Y:T.Tensor((64,64),'float16'),
          Z:T.Tensor((32,32),'float16')):
        with T.Kernel(1,threads=1):
            a=T.alloc_shared((32,32),'float16')
            b=T.alloc_shared((64,64),'float16')
            c=T.alloc_shared((32,32),'float16')
            T.copy(X,a)
            T.copy(Y,b)
            for i in T.serial(1):
                T.gemm(a,b[(i-1)*32*(1-axis):(i-1)*32*(1-axis)+32,
                           (i-1)*32*axis:(i-1)*32*axis+32],c,
                       transpose_B=True,clear_accum=True)
            T.copy(c,Z)
    with pytest.raises(Exception,match='region bounds'):
        lower(f)


@pytest.mark.parametrize('n,k,row,col,bn,bk', [
    (32,32,32,32,64,64), (64,64,64,64,128,128),
    (64,1056,32,32,96,1088), (32,2560,32,32,64,2592)])
def test_boundary_regions_object(tmp_path,n,k,row,col,bn,bk):
    src = lower(region_case(n,k,row,col,bn,bk))
    source = tmp_path / 'region.cpp'
    source.write_text(src)
    cc = '/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++'
    subprocess.run([cc,'-mv79','-mhvx','-mhvx-length=128B','-mhmx','-O2',
        '-std=c++17','-I',str(Path(__file__).resolve().parents[3]/'src'),
        '-c',str(source),'-o',str(tmp_path/'region.o')],check=True,capture_output=True)
    assert (tmp_path/'region.o').stat().st_size > 0


@pytest.mark.parametrize('pair', [(0,1),(0,2),(1,2)])
@pytest.mark.parametrize('view', [False,True])
def test_alias_data_vars(pair,view):
    buffers = [tirx.decl_buffer((32,64),'float16',name=x,scope='shared') for x in 'abc']
    buffers[pair[1]] = (tirx.decl_buffer((32,64),'float16',name='view',scope='shared',
        data=buffers[pair[0]].data) if view else buffers[pair[0]])
    op = GemmHMX(SimpleNamespace(a=buffers[0],b=buffers[1],c=buffers[2]))
    with pytest.raises(ValueError,match='alias'):
        op._validate()


@pytest.mark.parametrize('kw', [dict(strides=(128,1)),dict(elem_offset=32),
    dict(elem_offset=tirx.Var('offset','int32')),dict(strides=(tirx.Var('stride','int32'),1))])
def test_backing_storage_rejected(kw):
    a=tirx.decl_buffer((32,64),'float16',scope='shared',**kw)
    b=tirx.decl_buffer((32,64),'float16',scope='shared')
    c=tirx.decl_buffer((32,32),'float16',scope='shared')
    with pytest.raises(ValueError,match='offset|strides'):
        GemmHMX(SimpleNamespace(a=a,b=b,c=c))._validate()


def test_same_buffer_real_dsl_rejected():
    @T.prim_func
    def f(Z:T.Tensor((64,32),'float16')):
        with T.Kernel(1,threads=1):
            a=T.alloc_shared((64,64),'float16')
            c=T.alloc_shared((64,32),'float16')
            T.gemm(a,a[32:64,:],c,transpose_B=True,clear_accum=True)
            T.copy(c,Z)
    with pytest.raises(Exception,match='alias'):
        lower(f)

def make(offset=32):
    @T.prim_func
    def f(X:T.Tensor((32,64),'float16'),Y:T.Tensor((64,64),'float16'),Z:T.Tensor((32,32),'float16')):
        with T.Kernel(1,threads=1):
            a=T.alloc_shared((32,64),'float16')
            b=T.alloc_shared((64,64),'float16')
            c=T.alloc_shared((32,32),'float16')
            T.copy(X,a)
            T.copy(Y,b)
            T.gemm(a,b[offset:offset+32,:],c,transpose_B=True,clear_accum=True)
            T.copy(c,Z)
    return f

def test_tile_aligned_b_subregion():
    s=tilelang.engine.lower(make(),target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert 'hmx' in s

def test_unaligned_b_subregion_rejected():
    with pytest.raises(Exception,match='tile boundary'):
        tilelang.engine.lower(make(1),target='hexagon',enable_host_codegen=False,enable_device_compile=False)
