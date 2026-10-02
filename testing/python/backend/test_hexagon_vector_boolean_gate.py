"""Bare predicates must not bypass the physical-vector final IR gate."""
import pytest
import tilelang
from tvm import tirx as tir
from tilelang.hexagon.vector_verify import verify_vector_required


OPS = (tir.LT, tir.LE, tir.GT, tir.GE, tir.EQ, tir.NE, tir.Not, tir.And, tir.Or)


def expr(op, vector):
    boolean = op in (tir.Not, tir.And, tir.Or)
    dtype = ('bool' if boolean else 'int32') + ('x32' if vector else '')
    a, b = tir.Var('a', dtype), tir.Var('b', dtype)
    return op(a) if op is tir.Not else op(a, b)


def verify(body):
    verify_vector_required(tir.PrimFunc([], body).with_attr('tl.vector_required', 1))


@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('vector', [False, True])
def test_bare_predicate_rejected(op, vector):
    with pytest.raises(ValueError, match='generic data expression'):
        verify(tir.Evaluate(expr(op, vector)))


@pytest.mark.parametrize('op', OPS)
def test_scalar_control_preserved(op):
    verify(tir.IfThenElse(expr(op, False), tir.Evaluate(0), None))
