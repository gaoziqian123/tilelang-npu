"""Value masks vectorize; lazy memory guards must not become eager loads."""
import pytest
import tilelang as tl
from tilelang import tvm
from tvm import tirx as t


@pytest.mark.parametrize("dtype", ["float16", "float32", "int16", "int32"])
@pytest.mark.parametrize("extent,offset", [(32, 0), (64, 32), (33, 0), (64, 1)])
@pytest.mark.parametrize("mask", ["outer", "and", "cast"])
def test_eager_value_mask(dtype, extent, offset, mask):
    a = t.decl_buffer((extent + offset,), dtype, name="input")
    b = t.decl_buffer((extent + offset,), dtype, name="output")
    i, row = t.Var("lane", "int32"), t.Var("row", "int32")
    cond = i + 7 <= row * 3 + 11
    if mask == "and":
        cond = t.And(cond, i >= row)
    if mask == "cast":
        cond = t.Cast("int64", i) < t.Cast("int64", row) + 17
    value = t.Select(cond, a[i + offset], t.const(0, dtype))
    body = t.For(i, 0, extent, t.ForKind.VECTORIZED,
                 t.BufferStore(b, value, [i + offset]))
    f = t.PrimFunc([a.data, b.data, row], body)
    with tvm.target.Target("hexagon"):
        result = tl.transform.LegalizeVectorizedLoop()(tvm.IRModule({"main": f}))
    widths = []
    t.stmt_functor.post_order_visit(result["main"].body,
        lambda n: widths.append(int(n.extent)) if isinstance(n, t.For) and n.kind == t.ForKind.VECTORIZED else None)
    # Alignment and odd tails remain conservative; predicate variance alone
    # must not destroy an otherwise legal contiguous plan.
    assert max(widths, default=1) == (1 if extent == 33 or offset == 1 else 32)


@pytest.mark.parametrize("lazy", [True, False])
def test_lazy_guard_stays_scalar(lazy):
    a, b = t.decl_buffer((32,), "float32"), t.decl_buffer((64,), "float32")
    i = t.Var("lane", "int32")
    store = t.BufferStore(b, a[i], [i])
    body = (t.BufferStore(b, t.if_then_else(i < 31, a[i], t.const(0, "float32")), [i])
            if lazy else t.IfThenElse(i < 31, store, None))
    f = t.PrimFunc([a.data, b.data], t.For(i, 0, 64, t.ForKind.VECTORIZED, body))
    with tvm.target.Target("hexagon"):
        result = tl.transform.LegalizeVectorizedLoop()(tvm.IRModule({"main": f}))
    widths = []
    t.stmt_functor.post_order_visit(result["main"].body,
        lambda n: widths.append(int(n.extent)) if isinstance(n, t.For) and n.kind == t.ForKind.VECTORIZED else None)
    assert max(widths, default=1) == 1
