import pytest
import tilelang.language as T
from tvm import IRModule, get_global_func, tirx
from tvm.script.ir_builder import IRBuilder


def kernel(roles, depth=2, team=6, trips=4):
    with IRBuilder() as builder:
        with T.thread_binding(0, team, thread="threadIdx.x"):
            with T.Pipelined(trips, num_stages=depth, annotations={"tl.workergroup_schedule": "async"}):
                for name, engine, counts in roles:
                    with T.pipeline_stage(name, engine=engine, **counts):
                        T.evaluate(0)
    return IRModule({"main": tirx.PrimFunc([], builder.get())})


def plan(mod, cap=6):
    return get_global_func("tl.hexagon.transform.PlanWorkerPipeline")(cap, 1024, 8388608)(mod)


def annotation(mod, key):
    found = []
    tirx.stmt_functor.post_order_visit(mod["main"].body, lambda n:
        found.append(n.annotations[key]) if isinstance(n, tirx.For) and key in n.annotations else None)
    assert len(found) == 1
    return found[0]


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("trips", [0, 1, 7])
def test_counts_owner_and_depth(depth, trips):
    mod = plan(kernel([("foo", "hvx", {"workers": 3}), ("bar", "hmx", {"workers": 1}),
                       ("baz", "hvx", {"workers": 2})], depth, trips=trips))
    groups = annotation(mod, "tl.workergroup_groups")
    assert [int(g["first_worker"]) for g in groups] == [1, 0, 4]
    assert [int(g["worker_count"]) for g in groups] == [3, 1, 2]
    assert int(annotation(mod, "tl.workergroup_max_ordinal")) == trips


def test_weights_and_capability_not_six():
    mod = plan(kernel([("a", "hvx", {"weight": 2}), ("b", "hmx", {"workers": 1}),
                       ("c", "hvx", {"weight": 1})], team=8), cap=8)
    assert [int(g["worker_count"]) for g in annotation(mod, "tl.workergroup_groups")] == [4, 1, 3]


@pytest.mark.parametrize("roles,team,cap,match", [
    ([("a", "hmx", {"workers": 1}), ("b", "hmx", {"workers": 1})], 2, 6, "multiple HMX"),
    ([("a", "hvx", {"workers": 6})], 6, 5, "capability"),
    ([("a", "hvx", {"workers": 7})], 6, 6, "overflow"),
    ([("a", "hvx", {"workers": 3})], 6, 6, "fit team"),
    ([("a", "hvx", {"workers": 3}), ("a", "hvx", {"workers": 3})], 6, 6, "duplicate"),
])
def test_rejections(roles, team, cap, match):
    with pytest.raises(ValueError, match=match):
        plan(kernel(roles, team=team), cap)


def test_outside_pipeline():
    with IRBuilder() as builder:
        with T.pipeline_stage("outside", engine="hvx", workers=1):
            T.evaluate(0)
    with pytest.raises(ValueError, match="outside"):
        plan(IRModule({"main": tirx.PrimFunc([], builder.get())}))


def test_fail_closed_until_materialized():
    mod = kernel([("a", "hvx", {"workers": 6})])
    with pytest.raises(ValueError, match="materialization"):
        get_global_func("tl.hexagon.transform.RejectUnmaterializedWorkerPipeline")()(mod)
