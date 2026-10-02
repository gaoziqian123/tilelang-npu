"""Strict ordered promotion: structural rejects and separately rounded host C."""
import ctypes
import importlib.util
import re
from pathlib import Path
import subprocess

import numpy as np
import pytest
import tilelang
from tilelang import language as T
from tvm import IRModule, instrument, tirx
from tvm.ir.transform import PassContext
from tvm.target import Target


def make(n=64, k=7, rows=3):
    @T.prim_func
    def kernel(dst: T.Tensor((rows, n), 'float32'),
               lhs: T.Tensor((k, rows), 'float32'),
               rhs: T.Tensor((k, n), 'float32')):
        with T.Kernel(1, threads=1):
            for row in T.serial(rows):
                for reduction in T.serial(k):
                    for column in T.vectorized(n):
                        dst[row, column] = dst[row, column] + lhs[reduction, row] * rhs[reduction, column]
    return kernel


def promote(f, budget=128, legalize=True):
    mod = IRModule({'main': f})
    with Target('hexagon'), PassContext(config={'tl.ordered_accumulator_budget_bytes': budget,
                                               'tl.enable_ordered_accumulator_promotion': True}):
        mod = tirx.transform.BindTarget(Target('hexagon'))(mod)
        if legalize:
            mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
        return tilelang.transform.PromoteOrderedAccumulator()(mod)['main']


@pytest.mark.parametrize('n,k,rows', [(32, 1, 1), (64, 7, 3), (96, 5, 2), (128, 64, 4)])
def test_shapes(n, k, rows):
    assert 'ordered_acc' in str(promote(make(n,k,rows).with_attr('tir.noalias', True)))


def mutate(f, callback):
    return f.with_body(tirx.stmt_functor.ir_transform(f.body, None, callback))


@pytest.mark.parametrize('case', ['unknown_alias', 'same_data_view', 'dependent_dest',
    'read_dst', 'neighbor', 'extern', 'barrier', 'volatile', 'zero', 'negative',
    'budget', 'serial', 'predicate', 'oob', 'parallel'])
def test_reject(case):
    f = make(32).with_attr('tir.noalias', True)
    dst = list(f.buffer_map.values())[0]
    def change(node):
        if isinstance(node, tirx.BufferStore):
            if case == 'dependent_dest':
                return tirx.BufferStore(node.buffer, node.value, [0,0])
            if case in ('read_dst','neighbor'):
                val = tirx.BufferLoad(dst, [0, 0 if case=='read_dst' else 1])
                return tirx.BufferStore(node.buffer, node.value.a + val, node.indices)
            if case == 'extern':
                return tirx.BufferStore(node.buffer, node.value.a + tirx.call_extern('float32','opaque'),node.indices)
            if case == 'barrier':
                return tirx.SeqStmt([node,tirx.Evaluate(tirx.call_extern('int32','barrier'))])
            if case == 'predicate':
                return tirx.BufferStore(node.buffer,node.value,node.indices,tirx.const(True))
            if case == 'oob':
                return tirx.BufferStore(node.buffer,node.value,[node.indices[0],node.indices[1]+1])
        if isinstance(node,tirx.BufferLoad) and case=='same_data_view' and node.buffer != dst:
            view=tirx.decl_buffer(node.buffer.shape,'float32',data=dst.data)
            return tirx.BufferLoad(view,node.indices)
        if isinstance(node,tirx.For):
            if str(node.loop_var)=='reduction' and case in ('zero','negative'):
                return tirx.For(node.loop_var,0,0 if case=='zero' else -1,node.kind,node.body)
            if node.kind==tirx.ForKind.VECTORIZED and case in ('serial','parallel'):
                return tirx.For(node.loop_var,node.min,node.extent,
                    tirx.ForKind.SERIAL if case=='serial' else tirx.ForKind.PARALLEL,node.body)
        return node
    f=mutate(f,change)
    if case=='unknown_alias':
        f=f.without_attr('tir.noalias').without_attr('tirx.noalias')
    if case=='volatile':
        f=f.with_body(tirx.AttrStmt(dst.data,'volatile_scope',1,f.body))
    from tvm.ir import assert_structural_equal
    assert_structural_equal(f.with_attr('target', Target('hexagon')),
                            promote(f, 4 if case=='budget' else 128,False))


def assert_ordered_reference(seed, a, b, result):
    reference = seed.copy()
    with np.errstate(invalid='ignore', over='ignore'):
        for i in range(a.shape[0]):
            product = np.multiply(a[i, :, None], b[i, None, :], dtype=np.float32)
            reference = np.add(reference, product, dtype=np.float32)
    np.testing.assert_array_equal(np.isnan(reference), np.isnan(result))
    non_nan = ~np.isnan(reference)
    np.testing.assert_array_equal(reference.view('uint32')[non_nan],
                                  result.view('uint32')[non_nan])


def test_host_exact(tmp_path):
    f=make().with_attr('tir.noalias', True)
    p=promote(f)
    assert 'ordered_acc' in str(p)
    root=Path(tilelang.__file__).resolve().parents[1]
    libs=[]
    for name, func in [('base',f),('promoted',p)]:
        with Target('c'),PassContext(config={'tirx.disable_vectorize':True}):
            src=tilelang.engine.lower(func.without_attr('target'),target='c',enable_host_codegen=False,enable_device_compile=False).kernel_source
        assert src.strip() and 'kernel_kernel(' in src
        assert re.search(r'int32_t kernel_kernel\(float\* dst, float\* lhs, float\* rhs\)', src)
        cpp=tmp_path/(name+'.cpp'); cpp.write_text(src)
        so=tmp_path/(name+'.so')
        subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC',
                        '-I'+str(root/'src'),str(cpp),'-o',str(so)],check=True)
        libs.append(ctypes.CDLL(str(so)))
    rng=np.random.default_rng(4)
    for special in [None,0.,-0.,np.inf,-np.inf,np.nan,1e20,2**-149]:
        a=rng.normal(size=(7,3)).astype('float32')
        b=rng.normal(size=(7,64)).astype('float32')
        seed=rng.normal(size=(3,64)).astype('float32')
        if special is not None:
            a[:]=1; b[:]=0; b[0,:]=special; b[1,:]=1; b[2,:]=-special
            seed[:,::2]=-0.
        outputs=[]
        for lib in libs:
            out=seed.copy(); fn=lib.kernel_kernel; fn.argtypes=[ctypes.c_void_p]*3
            fn.restype=ctypes.c_int32
            assert fn(out.ctypes.data,a.ctypes.data,b.ctypes.data) == 0
            assert_ordered_reference(seed,a,b,out)
            outputs.append(out.view('uint32'))
        np.testing.assert_array_equal(*outputs)


@instrument.pass_instrument
class Capture:
    def __init__(self):
        self.before=self.after=''
        self.legalized=None
        self.pre_legalize=None
    def run_before_pass(self,mod,info):
        if info.name=='tl.PromoteOrderedAccumulator': self.before=str(mod)
        if info.name=='tl.LegalizeVectorizedLoop': self.pre_legalize=mod
    def run_after_pass(self,mod,info):
        if info.name=='tl.PromoteOrderedAccumulator': self.after=str(mod)
        if info.name=='tl.LegalizeVectorizedLoop': self.legalized=mod


def test_hexagon_objects(tmp_path):
    root=Path(tilelang.__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('gdn',root/'examples/hexagon/gdn/gdn.py')
    gdn=importlib.util.module_from_spec(spec); spec.loader.exec_module(gdn)
    for label,f in [('synthetic',make(128).with_attr('tir.noalias', True)),('main_stage',gdn.main_stage())]:
        for enabled in (False,True):
            cap=Capture()
            with PassContext(config={'tl.hexagon.affine_transpose':True,
                                     'tl.enable_ordered_accumulator_promotion':enabled},instruments=[cap]):
                src=tilelang.engine.lower(f,target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
            assert src.strip()
            if enabled:
                if label == 'synthetic':
                    assert 'ordered_acc' in cap.after
                else:
                    # Checked slab + bounded-effect calls now prove exclusivity.
                    assert 'ordered_acc' in cap.after
                    kr = src.split('tl::profile_mark((&(LOG[0])), 13);')[1].split('tl::profile_mark((&(LOG[0])), 14);')[0]
                    update = re.split(r'for \(int32_t i\w* = 0; i\w* < 64;[^\n]*\n',kr)[1].split('\n          }')[0]
                    assert '&S[' not in update
                    assert 'tl::hvx_add' in update and 'tl::hvx_mul' in update
            else:
                default_cap = Capture()
                with PassContext(config={'tl.hexagon.affine_transpose': True},
                                 instruments=[default_cap]):
                    default_src = tilelang.engine.lower(f,target='hexagon',enable_host_codegen=False,
                                                        enable_device_compile=False).kernel_source
                assert default_src.strip()
                # Compare the same input: independent GDN lowering rebuilds
                # layout-map Buffer keys and later scale allocation names.
                from tvm.ir import assert_structural_equal
                with Target('hexagon'), PassContext():
                    default_ir = tilelang.transform.LegalizeVectorizedLoop()(cap.pre_legalize)
                with Target('hexagon'), PassContext(config={'tl.enable_ordered_accumulator_promotion': False}):
                    disabled_ir = tilelang.transform.LegalizeVectorizedLoop()(cap.pre_legalize)
                assert_structural_equal(default_ir, disabled_ir)
                if label == 'synthetic': assert default_src == src
                assert default_cap.before == default_cap.after == ''
            stem=tmp_path/(label+('_on' if enabled else '_off'))
            stem.with_suffix('.cpp').write_text(src)
            stem.with_suffix('.tir').write_text(cap.after)
            flags=['/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++',
                   '-mv79','-mhvx','-mhvx-length=128B','-mhmx','-O2','-fPIC','-std=c++17','-I'+str(root/'src')]
            for mode,suffix in [('-c','.o'),('-S','.s')]:
                subprocess.run(flags+[mode,str(stem.with_suffix('.cpp')),'-o',str(stem.with_suffix(suffix))],check=True)
            assert stem.with_suffix('.o').stat().st_size > 0
            asm = stem.with_suffix('.s').read_text()
            assert ('kernel_kernel:' if label == 'synthetic' else 'k3_kernel:') in asm
            if enabled and label == 'synthetic':
                # This seven-step reduction is unrolled by the real DSP compiler.
                # One final vector store, no vector stack spills/round-trips.
                stores = re.findall(r'^\s*vmemu?\([^\n]*\)\s*=', asm, re.MULTILINE)
                assert len(stores) == 1, stores
                assert not re.search(r'vmemu?\(r(?:29|30)\b', asm)
                assert asm.count('vadd(') == 7
