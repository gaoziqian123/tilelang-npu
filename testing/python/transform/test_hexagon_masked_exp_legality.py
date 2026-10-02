"""Negative legality cases must retain lazy evaluation and side effects."""
import pytest
import tilelang
from tilelang import tvm
from tvm import tirx as t


@pytest.mark.parametrize('case', ['side_effect', 'predicate_effect', 'volatile',
                                 'oob', 'float16', 'float64', 'negative_zero',
                                 'condition_load', 'predicated_load'])
def test_masked_exp_no_speculation(case):
    dtype = case if case in ('float16', 'float64') else 'float32'
    a = t.decl_buffer((17 if case == 'oob' else 64,), dtype, name='a')
    b = t.decl_buffer((64,), dtype, name='b')
    i, bound = t.Var('i', 'int32'), t.Var('bound', 'int32')
    condition = i < bound
    value = a[i]
    if case == 'side_effect':
        value = t.call_extern('float32', 'observe', a[i])
    if case == 'predicate_effect':
        condition = t.call_extern('int32', 'predicate', i) != 0
    if case == 'condition_load':
        condition = t.And(i < bound, a[i] > t.const(0, dtype))
    if case == 'predicated_load':
        value = t.BufferLoad(a, [i], predicate=i < bound)
    zero = t.const(-0.0 if case == 'negative_zero' else 0.0, dtype)
    store = t.BufferStore(b, t.if_then_else(condition, t.exp(value), zero), [i])
    body = t.For(i, 0, 64, t.ForKind.VECTORIZED, store)
    if case == 'volatile':
        body = t.AttrStmt(a.data, 'volatile_scope', 1, body)
    f = t.PrimFunc([a.data, b.data, bound], body).with_attr('target', tvm.target.Target('hexagon'))
    with tvm.target.Target('hexagon'):
        result = tilelang.transform.LegalizeVectorizedLoop()(tvm.IRModule({'main': f}))
    lazy, selects = [], []
    def visit(n):
        if isinstance(n, t.Call) and n.op == tvm.ir.Op.get('tirx.if_then_else'):
            lazy.append(n)
        if isinstance(n, t.Select):
            selects.append(n)
    t.stmt_functor.post_order_visit(result['main'].body, visit)
    assert len(lazy) == 1
    assert not selects
