"""Terminal effects remain inside an unconditional logical stage lifetime."""
import pytest
import tilelang
from tilelang import language as T
from tvm import IRModule, get_global_func, tirx
from tvm.script.ir_builder import IRBuilder


def terminal_plan(trips=3, minimum=2, early=False, shared_write=False):
    src = tirx.decl_buffer((64,), "float16", name="source")
    out = tirx.decl_buffer((64,), "float16", name="output")
    with IRBuilder() as builder:
        with T.sblock("root"):
            slot = T.sblock_alloc_buffer((64,), "float16", scope="shared")
            with T.thread_binding(0, 2, thread="threadIdx.x"):
                with T.Pipelined(minimum, minimum + trips, num_stages=2,
                                 annotations={"tl.workergroup_schedule": "async"}) as iteration:
                    with T.pipeline_stage("load", engine="hvx", workers=1):
                        T.evaluate(T.copy(src, slot))
                        if early:
                            with T.If(iteration == minimum + trips - 1):
                                with T.Then():
                                    T.evaluate(T.copy(slot, out))
                    with T.pipeline_stage("finish", engine="hvx", workers=1):
                        with T.If(iteration == minimum + trips - 1):
                            with T.Then():
                                T.evaluate(T.copy(slot, slot if shared_write else out))
    mod = IRModule({"main": tirx.PrimFunc([], builder.get())})
    return get_global_func("tl.hexagon.transform.PlanWorkerPipeline")(2, 16, 8192)(mod)


@pytest.mark.parametrize("trips", [1, 3, 7])
def test_terminal_condition_keeps_unconditional_edge(trips):
    mod = terminal_plan(trips)
    loops, conditions = [], []
    def visit(node):
        if isinstance(node, tirx.For) and "tl.workergroup_edges" in node.annotations:
            loops.append(node)
        if isinstance(node, tirx.IfThenElse):
            conditions.append(node)
    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert len(conditions) == 1
    assert len(loops[0].annotations["tl.workergroup_edges"]) == 1
    assert int(loops[0].annotations["tl.workergroup_event_count"]) == 4


def test_terminal_producer_rejected():
    with pytest.raises(ValueError, match="final pipeline stage"):
        terminal_plan(early=True)


def test_terminal_cross_role_write_rejected():
    with pytest.raises(ValueError, match="cross-role lifetime"):
        terminal_plan(shared_write=True)
