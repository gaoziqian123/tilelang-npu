import pytest
import tilelang.language as T
from tvm import IRModule, tirx, get_global_func
from tvm.ir import Op
from tvm.script.ir_builder import IRBuilder


def make(body, wrapped=True):
    if wrapped:
        body = tirx.AttrStmt({"engine": "hmx", "first_worker": 0,
                             "local_id": 0, "local_size": 1},
                            "tl.workergroup_local", 1, body)
    return IRModule({"main": tirx.PrimFunc([], body)})


def test_group_barrier_preserves_completion_and_scatter():
    sync = tirx.Call("int32", "tirx.tvm_storage_sync", [tirx.StringImm("shared")])
    body = tirx.SeqStmt([tirx.Evaluate(c) for c in [
        sync, tirx.Call("handle", Op.get("tl.hexagon.hmx_clear_acc"), []),
        tirx.Call("handle", Op.get("tl.hexagon.hmx_store_after"), [0, 0]),
        tirx.Call("handle", Op.get("tl.hexagon.scatter_release"), [0]), sync]])
    mod = get_global_func("tl.hexagon.transform.LowerWorkerSync")()(make(body))
    names = []
    tirx.stmt_functor.post_order_visit(mod["main"].body, lambda n:
        names.append(n.op.name) if isinstance(n, tirx.Call) else None)
    assert names == ["tl.hexagon.workergroup_barrier", "tl.hexagon.hmx_clear_acc",
                     "tl.hexagon.hmx_store_after", "tl.hexagon.scatter_release",
                     "tl.hexagon.workergroup_barrier"]


def test_unowned_full_team_barrier_rejected():
    sync = tirx.Evaluate(tirx.Call("int32", "tirx.tvm_storage_sync", [tirx.StringImm("shared")]))
    owned = make(tirx.Evaluate(0))["main"].body
    mod = make(tirx.SeqStmt([owned, sync]), wrapped=False)
    with pytest.raises(ValueError, match="full-team barrier"):
        get_global_func("tl.hexagon.transform.LowerWorkerSync")()(mod)
