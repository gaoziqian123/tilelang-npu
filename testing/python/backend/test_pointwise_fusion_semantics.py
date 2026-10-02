"""Host semantic checks, independent of target C code spelling."""
import numpy as np
import pytest
import tilelang
from tvm import IRModule, tirx
from tvm.ir import assert_structural_equal
from tilelang.transform.fuse_pointwise_stages import FusePointwiseStages


def case(n, mode="normal"):
    a = tirx.decl_buffer((n,), "float32", name="source")
    b = tirx.decl_buffer((n,), "float32", name="scratch", scope="local")
    i, j = tirx.Var("u", "int32"), tirx.Var("v", "int32")
    value = tirx.Cast("float32", tirx.Cast("float16", a[i]))
    if mode == "producer_stencil":
        value = b[i + 1]
    if mode == "call":
        value = tirx.call_extern("float32", "opaque", a[i])
    p = tirx.For(i, 0, n, tirx.ForKind.SERIAL, tirx.BufferStore(b, value, [i]))
    load = b[j + 1] if mode == "stencil" else b[j]
    if mode == "alias":
        view = tirx.decl_buffer((n,), "float32", data=b.data, scope="local")
        load = view[j]
    if mode == "multiple":
        load = b[j] + b[j + 1]
    value = tirx.Select(j < n - 1, load * tirx.FloatImm("float32", 0.25),
                        tirx.FloatImm("float32", -0.0))
    c = tirx.For(j, 0, n + 1 if mode == "domain" else n, tirx.ForKind.SERIAL,
                 tirx.BufferStore(b, value, [j]))
    seq = [p, c]
    if mode == "escape":
        seq.append(tirx.Evaluate(tirx.call_extern("int32", "escape", b.data)))
    if mode == "bare_handle":
        seq.append(tirx.Bind(tirx.Var("alias", "handle"), b.data))
    if mode == "barrier":
        seq.insert(1, tirx.Evaluate(tirx.call_extern("int32", "barrier")))
    if mode == "old_consumer":
        seq.insert(1, tirx.Evaluate(b[0]))
    body = tirx.SeqStmt(seq)
    if mode in ("async", "profile", "volatile"):
        body = tirx.AttrStmt(b.data, mode, 1, body)
    return tirx.PrimFunc([a.data], body), a, b


def transform(f):
    return FusePointwiseStages()(IRModule({"main": f}))["main"]


def execute(stmt, arrays, env):
    """Small scalar interpreter preserving every explicit cast/FP operation."""
    def expr(e):
        if isinstance(e, tirx.Var):
            return env[e]
        if isinstance(e, (tirx.IntImm, tirx.FloatImm)):
            return np.asarray(e.value, dtype=str(e.dtype))[()]
        if isinstance(e, tirx.BufferLoad):
            return arrays[e.buffer.data][tuple(int(expr(i)) for i in e.indices)]
        if isinstance(e, tirx.Cast):
            return np.asarray(expr(e.value), dtype=str(e.dtype))[()]
        if isinstance(e, tirx.Select):
            return expr(e.true_value if expr(e.condition) else e.false_value)
        ops = {tirx.Add: lambda a, b: a + b, tirx.Sub: lambda a, b: a - b,
               tirx.Mul: lambda a, b: a * b, tirx.LT: lambda a, b: a < b}
        return np.asarray(ops[type(e)](expr(e.a), expr(e.b)), dtype=str(e.dtype))[()]
    if isinstance(stmt, tirx.SeqStmt):
        for s in stmt.seq:
            execute(s, arrays, env)
    elif isinstance(stmt, tirx.For):
        for i in range(int(expr(stmt.min)), int(expr(stmt.min) + expr(stmt.extent))):
            env[stmt.loop_var] = i
            execute(stmt.body, arrays, env)
    elif isinstance(stmt, tirx.BufferStore):
        arrays[stmt.buffer.data][tuple(int(expr(i)) for i in stmt.indices)] = expr(stmt.value)
    else:
        raise AssertionError(type(stmt))


@pytest.mark.parametrize("length", [0, 1, 3, 31, 32, 33, 65, 127, 257])
@pytest.mark.parametrize("dynamic", [False, True])
def test_semantics(length, dynamic):
    n = tirx.Var("runtime_length", "int32") if dynamic else length
    f, a, b = case(n)
    g = transform(f)
    assert isinstance(g.body, tirx.For)
    rng = np.random.default_rng(827)
    x = rng.normal(size=length).astype("float32")
    if length >= 3:
        x[:3] = [np.inf, -np.inf, np.nan]
    outputs = []
    for body in (f.body, g.body):
        arrays = {a.data: x.copy(), b.data: np.full(length, np.nan, dtype="float32")}
        execute(body, arrays, {n: length} if dynamic else {})
        outputs.append(arrays[b.data].view("uint32"))
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("mode", ["producer_stencil", "call", "stencil", "alias",
    "multiple", "domain", "escape", "bare_handle", "barrier", "old_consumer", "async", "profile", "volatile"])
def test_reject_unchanged(mode):
    f, _, _ = case(65, mode)
    assert_structural_equal(f, transform(f))
