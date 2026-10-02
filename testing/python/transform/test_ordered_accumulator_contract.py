"""Legal inputs and strict rejects; original unknown-alias fixture is unchanged."""
import ctypes
import re
from pathlib import Path
import subprocess

import numpy as np
import pytest
import tilelang
from tilelang import language as T
from tvm import IRModule, tirx
from tvm.ir.transform import PassContext
from tvm.target import Target
from test_ordered_accumulator import make, mutate, Capture, assert_ordered_reference


def run(f, budget=128):
    with Target('hexagon'), PassContext(config={
        'tl.enable_ordered_accumulator_promotion': True,
        'tl.ordered_accumulator_budget_bytes': budget,
    }):
        mod=tirx.transform.BindTarget(Target('hexagon'))(IRModule({'main':f}))
        mod=tilelang.transform.LegalizeVectorizedLoop()(mod)
        before=mod['main']
        return before,tilelang.transform.PromoteOrderedAccumulator()(mod)['main']


def internal(n=64):
    @T.prim_func
    def kernel(out:T.Tensor((2,n),'float32'),a:T.Tensor((7,2),'float32'),b:T.Tensor((7,n),'float32')):
        with T.Kernel(1,threads=1):
            state=T.alloc_local((2,n),'float32')
            left=T.alloc_local((7,2),'float32')
            right=T.alloc_local((7,n),'float32')
            for r in T.serial(2):
                for c in T.vectorized(n):
                    state[r,c]=out[r,c]
            for i in T.serial(7):
                for r in T.serial(2):
                    left[i,r]=a[i,r]
                for c in T.vectorized(n):
                    right[i,c]=b[i,c]
            for r in T.serial(2):
                for i in T.serial(7):
                    for c in T.vectorized(n):
                        state[r,c]=state[r,c]+left[i,r]*right[i,c]
            for r in T.serial(2):
                for c in T.vectorized(n):
                    out[r,c]=state[r,c]
    return kernel


@pytest.mark.parametrize('n',[16,32,64,96,128])
@pytest.mark.parametrize('private',[False,True])
def test_legal(n,private):
    f=internal(n) if private else make(n,7,2).with_attr('tir.noalias',True)
    assert 'ordered_acc' in str(run(f)[1])


def test_original_unknown_alias_is_negative():
    from tvm.ir import assert_structural_equal
    before,after=run(make())
    assert_structural_equal(before,after)


@pytest.mark.parametrize('case',['alias_view','i_dest','neighbor','extern','escape',
    'barrier','volatile','async','zero','negative','budget','serial','predicate',
    'oob','workers','noncompact','unknown_stride'])
def test_strict_reject(case):
    f=make(32).with_attr('tir.noalias',True)
    buffers=list(f.buffer_map.values()); dst=buffers[0]
    red=[]
    tirx.stmt_functor.post_order_visit(f.body,lambda n: red.append(n.loop_var)
        if isinstance(n,tirx.For) and str(n.loop_var)=='reduction' else None)
    def change(n):
        if isinstance(n,tirx.BufferLoad) and case=='alias_view' and n.buffer==buffers[2]:
            return tirx.BufferLoad(tirx.decl_buffer(n.buffer.shape,'float32',data=dst.data),n.indices)
        if isinstance(n,tirx.BufferStore):
            idx=n.indices; val=n.value
            if case=='i_dest': idx=[red[0],idx[1]]
            if case=='neighbor': val=val.a+tirx.BufferLoad(dst,[0,1])
            if case=='extern': val=val.a+tirx.call_extern('float32','opaque')
            if case=='oob': idx=[idx[0],idx[1]+1]
            if case=='barrier': return tirx.SeqStmt([n,tirx.Evaluate(tirx.call_extern('int32','fence'))])
            if case=='predicate': return tirx.BufferStore(n.buffer,val,idx,tirx.const(True))
            return tirx.BufferStore(n.buffer,val,idx)
        if isinstance(n,tirx.For):
            if str(n.loop_var)=='reduction' and case in ('zero','negative'):
                return tirx.For(n.loop_var,0,0 if case=='zero' else -1,n.kind,n.body)
            if n.kind==tirx.ForKind.VECTORIZED and case=='serial':
                return tirx.For(n.loop_var,n.min,n.extent,tirx.ForKind.SERIAL,n.body)
            if n.thread_binding is not None and case=='workers':
                return tirx.For(n.loop_var,n.min,2,n.kind,n.body,n.thread_binding)
        return n
    f=mutate(f,change)
    if case in ('noncompact','unknown_stride'):
        stride=33 if case=='noncompact' else tirx.Var('stride','int32')
        replacement=tirx.decl_buffer(dst.shape,'float32',data=dst.data,strides=[stride,1])
        def replace(n):
            if isinstance(n,tirx.BufferLoad) and n.buffer==dst: return tirx.BufferLoad(replacement,n.indices)
            if isinstance(n,tirx.BufferStore) and n.buffer==dst: return tirx.BufferStore(replacement,n.value,n.indices)
            return n
        f=mutate(f,replace)
    if case in ('volatile','async'):
        f=f.with_body(tirx.AttrStmt(dst.data,'volatile_scope' if case=='volatile' else 'hexagon.async_scope',1,f.body))
    if case=='escape':
        f=f.with_body(tirx.SeqStmt([tirx.Evaluate(tirx.call_extern('int32','retain',dst.data)),f.body]))
    # Do not legalize invalid inputs: rejection must preserve the original IR.
    from tvm.ir import assert_structural_equal
    with PassContext(config={'tl.ordered_accumulator_budget_bytes':4 if case=='budget' else 128}):
        after=tilelang.transform.PromoteOrderedAccumulator()(IRModule({'main':f}))['main']
    assert_structural_equal(f,after)


def test_exact_host_and_hexagon_objects(tmp_path):
    root=Path(tilelang.__file__).resolve().parents[1]
    f=internal(128)
    before,promoted=run(f)
    assert 'ordered_acc' in str(promoted)
    libs=[]
    for name,func in [('base',before),('promoted',promoted)]:
        with Target('c'),PassContext(config={'tirx.disable_vectorize':True}):
            source=tilelang.engine.lower(func.without_attr('target'),target='c',enable_host_codegen=False,enable_device_compile=False).kernel_source
        assert source.strip() and 'kernel_kernel(' in source
        assert re.search(r'int32_t kernel_kernel\(float\* a, float\* b, float\* out\)', source)
        cpp=tmp_path/(name+'.cpp'); cpp.write_text(source)
        so=tmp_path/(name+'.so')
        subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC','-I'+str(root/'src'),str(cpp),'-o',str(so)],check=True)
        libs.append(ctypes.CDLL(str(so)))
    rng=np.random.default_rng(17)
    for special in [None,0.,-0.,np.inf,-np.inf,np.nan,1e20,2**-149]:
        a=rng.normal(size=(7,2)).astype('float32'); b=rng.normal(size=(7,128)).astype('float32')
        seed=rng.normal(size=(2,128)).astype('float32')
        if special is not None:
            a[:]=1; b[:]=0; b[0,:]=special; b[1,:]=1; b[2,:]=-special; seed[:,::2]=-0.
        results=[]
        for lib in libs:
            out=seed.copy(); fn=lib.kernel_kernel; fn.argtypes=[ctypes.c_void_p]*3
            fn.restype=ctypes.c_int32
            assert fn(a.ctypes.data,b.ctypes.data,out.ctypes.data) == 0
            assert_ordered_reference(seed,a,b,out)
            results.append(out.view('uint32'))
        np.testing.assert_array_equal(*results)
    for enabled in [False,True]:
        cap=Capture()
        with PassContext(config={'tl.enable_ordered_accumulator_promotion':enabled},instruments=[cap]):
            source=tilelang.engine.lower(f,target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
        assert source.strip()
        if enabled: assert 'ordered_acc' in cap.after
        stem=tmp_path/('hex_on' if enabled else 'hex_off')
        stem.with_suffix('.cpp').write_text(source); stem.with_suffix('.tir').write_text(cap.after)
        flags=['/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++','-mv79','-mhvx','-mhvx-length=128B','-mhmx','-O2','-fPIC','-std=c++17','-I'+str(root/'src')]
        for mode,suffix in [('-c','.o'),('-S','.s')]:
            subprocess.run(flags+[mode,str(stem.with_suffix('.cpp')),'-o',str(stem.with_suffix(suffix))],check=True)
    print('ARTIFACTS',tmp_path)
