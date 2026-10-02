"""Production CodeGenHexagon callback ABI checks, without host emitter."""
import pytest
import tilelang
from tvm import IRModule, tirx, get_global_func
from tvm.ir import Op, PrimType
from tvm.target import Target


def emit(context=True, full_team=False):
    worker = tirx.Var("worker", "handle")
    call = (tirx.Call("int32", "tirx.tvm_storage_sync", [tirx.StringImm("shared")])
            if full_team else tirx.Call("int32", Op.get("tl.hexagon.workergroup_barrier"), []))
    f = tirx.PrimFunc([worker], tirx.Evaluate(call), ret_type=PrimType("int32"))
    f = f.with_attr("global_symbol", "callback_probe").with_attr("calling_conv", 2)
    if context:
        f = f.with_attr("tl.workergroup_callback_worker", worker)
    return get_global_func("target.build.tilelang_hexagon_without_compile")(
        IRModule({"callback_probe": f}), Target("hexagon")).inspect_source()


def test_production_group_barrier_status():
    source = emit()
    assert "tl_wg_group_barrier((const tl_wg_worker*)worker)" in source
    assert "if (wg_status != 0) return wg_status;" in source
    assert "tl_hex_barrier();" not in source
    assert "workergroup_abi.h" in source


def test_missing_callback_context_rejected():
    with pytest.raises(ValueError, match="explicit ABI worker callback"):
        emit(context=False)


def test_unlowered_team_barrier_rejected():
    with pytest.raises(ValueError, match="full-team storage barrier"):
        emit(full_team=True)
