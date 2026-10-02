import pytest
import tilelang
from tvm import IRModule, tirx
from tilelang.transform.fuse_pointwise_stages import FusePointwiseStages


@pytest.mark.parametrize("n", [0, 1, 31, 32, 33, 63, 64, 65, 127, 255, 256, 257])
@pytest.mark.parametrize("shift", [0, 1])
def test_private_version(n, shift):
    a = tirx.decl_buffer((n,), "float32", name="renamed_input")
    b = tirx.decl_buffer((n,), "float32", name="renamed_private", scope="local")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    p = tirx.For(i, 0, n, tirx.ForKind.SERIAL,
                 tirx.BufferStore(b, a[i] * tirx.FloatImm("float32", 0.5), [i]))
    c = tirx.For(j, 0, n, tirx.ForKind.SERIAL,
                 tirx.BufferStore(b, tirx.Select(j < 31, b[j + shift],
                                  tirx.FloatImm("float32", -0.0)), [j]))
    f = tirx.PrimFunc([a.data], tirx.SeqStmt([p, c]))
    result = FusePointwiseStages()(IRModule({"main": f}))["main"]
    loops = []
    tirx.stmt_functor.post_order_visit(result.body,
        lambda node: loops.append(node) if isinstance(node, tirx.For) else None)
    assert len(loops) == (2 if shift else 1)
