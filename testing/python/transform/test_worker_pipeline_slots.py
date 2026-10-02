"""Address and role materialization checks, not runtime execution claims."""
import pytest
import tilelang.language as T
from tvm import IRModule, tirx, get_global_func
from tvm.script.ir_builder import IRBuilder


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_real_slot_indices_and_group_branches(depth):
    a = tirx.decl_buffer((32,), "float32", name="a", scope="shared")
    out = tirx.decl_buffer((32,), "float32", name="out")
    with IRBuilder() as builder:
        with T.thread_binding(0, 6, thread="threadIdx.x"):
            with T.serial(0, 3):
                with T.Pipelined(5, num_stages=depth, annotations={"tl.workergroup_schedule": "async"}):
                    with T.pipeline_stage("producer", engine="hvx", workers=4):
                        T.buffer_store(a, 7.0, [0])
                    with T.pipeline_stage("consumer", engine="hvx", workers=2):
                        T.buffer_store(out, a[0], [0])
    mod = IRModule({"main": tirx.PrimFunc([], builder.get())})
    mod = get_global_func("tl.hexagon.transform.PlanWorkerPipeline")(6, 64, 8192)(mod)
    mod = get_global_func("tl.hexagon.transform.MaterializeWorkerSlots")()(mod)
    loads, stores, branches, locals_ = [], [], [], []
    def visit(n):
        if isinstance(n, tirx.BufferLoad): loads.append(n)
        if isinstance(n, tirx.BufferStore): stores.append(n)
        if isinstance(n, tirx.IfThenElse): branches.append(n)
        if isinstance(n, tirx.AttrStmt) and n.attr_key == "tl.workergroup_local": locals_.append(n)
    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    shared_stores = [n for n in stores if n.buffer.name == "a"]
    assert len(shared_stores) == 1 and len(loads) == 1
    assert list(shared_stores[0].buffer.shape) == [depth, 32]
    assert list(loads[0].buffer.strides) == [32, 1]
    assert len(loads[0].indices) == 2
    assert len(branches) == 2
    assert [int(n.node["local_size"]) for n in locals_] == [4, 2]
    assert all("int64" in str(n.node["ordinal"].dtype) for n in locals_)
    # A partially materialized function must never silently enter old codegen.
    with pytest.raises(ValueError, match="materialization"):
        get_global_func("tl.hexagon.transform.RejectUnmaterializedWorkerPipeline")()(mod)
