"""Region boundaries do not erase whole-function escape/alias facts."""
import importlib.util
from pathlib import Path
import pytest
import tilelang
from tvm import IRModule, tirx
from tvm.ir import Op, assert_structural_equal
from test_ordered_accumulator import make
from test_ordered_accumulator_contract import run


@pytest.mark.parametrize('kind', ['bounded_other', 'bounded_dst', 'unknown', 'escape', 'pure', 'inside', 'async_dst', 'async_other'])
def test_boundaries(kind):
    f=make(32).with_attr('tir.noalias', True)
    dst=list(f.buffer_map.values())[0]
    other=tirx.decl_buffer((34,), 'uint64', name='ledger')
    param=tirx.Var('ledger_handle','handle')
    f=tirx.PrimFunc([*f.params,param],f.body,buffer_map={**dict(f.buffer_map),param:other},attrs=f.attrs)
    ptr=tirx.call_intrin('handle','tirx.address_of',tirx.BufferLoad(dst if kind=='bounded_dst' else other,[0,0] if kind=='bounded_dst' else [0]))
    if kind in ('bounded_other','bounded_dst','inside'):
        call=tirx.Call('void',Op.get('tl.hexagon.profile_mark'),[ptr,0])
    elif kind in ('async_dst','async_other'):
        ptr=tirx.call_intrin('handle','tirx.address_of',tirx.BufferLoad(dst if kind=='async_dst' else other,[0,0] if kind=='async_dst' else [0]))
        call=tirx.Call('void',Op.get('tl.hexagon.scatter_release'),[ptr])
    elif kind=='pure': call=tirx.exp(tirx.const(1.,'float32'))
    else: call=tirx.call_extern('int32','unknown',dst.data) if kind=='escape' else tirx.call_extern('int32','unknown')
    stmt=tirx.Evaluate(call)
    if kind=='inside':
        f=f.with_body(tirx.stmt_functor.ir_transform(f.body,None,lambda n:tirx.SeqStmt([stmt,n]) if isinstance(n,tirx.BufferStore) else n))
    else: f=f.with_body(tirx.SeqStmt([stmt,f.body,stmt]))
    before,after=run(f)
    if kind in ('bounded_other','bounded_dst','pure','async_other'):
        assert 'ordered_acc' in str(after)
        assert_structural_equal(before.body.seq[0],after.body.seq[0])
        assert_structural_equal(before.body.seq[-1],after.body.seq[-1])
    else: assert_structural_equal(before,after)


def test_idempotent():
    _,after=run(make(128).with_attr('tir.noalias',True))
    again=tilelang.transform.PromoteOrderedAccumulator()(IRModule({'main':after}))['main']
    assert_structural_equal(after,again)


def test_slab_proof():
    path=Path(tilelang.__file__).resolve().parents[1]/'examples/hexagon/gdn/layout.py'
    spec=importlib.util.spec_from_file_location('layout_contract',path)
    layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
    specs,offsets,size=layout.layout(1024,16,32)
    assert len(layout.validate(specs,offsets,size))==len(specs)
    offsets['RF']=offsets['S']
    with pytest.raises(ValueError,match='overlapping'):
        layout.validate(specs,offsets,size)
