"""Resident lifetime legality and schedule simulation (no device execution)."""
import importlib.util
from pathlib import Path
import random

import pytest
import tilelang
from tilelang import language as T
from tvm import IRModule, get_global_func, tirx
from tvm.script.ir_builder import IRBuilder


def plan(*, declared=True, varying=False, depth=2, events=8, merge=True):
    src = tirx.decl_buffer((4, 32), "float16", name="source")
    out = tirx.decl_buffer((32,), "float16", name="out")
    with IRBuilder() as builder:
        with T.sblock("root"):
            a = T.sblock_alloc_buffer((32,), "float16", scope="shared")
            b = T.sblock_alloc_buffer((32,), "float16", scope="shared")
            with T.thread_binding(0, 2, thread="threadIdx.x"):
                with T.serial(0, 3):
                    with T.serial(0, 4) as reuse:
                        with T.Pipelined(2, num_stages=depth, annotations={
                                "tl.workergroup_schedule":"async",
                                "tl.workergroup_merge_edges":int(merge)}):
                            with T.pipeline_stage("arbitrary_writer", engine="hvx", workers=1):
                                T.evaluate(T.copy(src[0, :], a))
                                with T.If(reuse == 0):
                                    with T.Then():
                                        T.evaluate(T.copy(src[reuse if varying else 0, :], b,
                                                annotations={"tl.worker_resident_loop":reuse} if declared else {}))
                            with T.pipeline_stage("arbitrary_reader", engine="hvx", workers=1):
                                T.buffer_store(out, a[0] + b[0], [0])
    mod = IRModule({"main":tirx.PrimFunc([], builder.get())})
    return get_global_func("tl.hexagon.transform.PlanWorkerPipeline")(2, events, 8192)(mod)


def test_merged_dependency_not_buffer_name():
    mod = plan()
    loops = []
    tirx.stmt_functor.post_order_visit(mod["main"].body,
        lambda n: loops.append(n) if isinstance(n, tirx.For) and
        "tl.workergroup_edges" in n.annotations else None)
    edges = loops[0].annotations["tl.workergroup_edges"]
    assert len(edges) == 1
    assert [int(x) for x in edges[0]["protected_slots"]] == [0, 1]
    assert int(loops[0].annotations["tl.workergroup_event_count"]) == 4


@pytest.mark.parametrize("kwargs,reason", [
    ({"declared":False}, "explicit resident"),
    ({"varying":True}, "varies across reuse"),
    ({"depth":3}, "full-depth resident"),
    ({"events":3}, "event budget"),
])
def test_resident_rejections(kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        plan(**kwargs)


def test_cross_outer_schedule_model():
    # Random interleavings of independent producer/HMX/collective-HVX roles.
    # Source data is labelled by outer epoch; stale/premature B overwrite is
    # checked independently of event counters. HVX release is collective here.
    for seed in range(100):
        rng = random.Random(seed)
        ready, free, partial_ready, partial_free = ([0, 0] for _ in range(4))
        resident = [None, None]
        cursors = [0, 0, 0]
        total, reuse_count = 24, 4
        while min(cursors) < total:
            choices = []
            for role, q in enumerate(cursors):
                if q == total:
                    continue
                slot, generation = q % 2, q // 2 + 1
                outer, reuse = q // (reuse_count*2), q // 2 % reuse_count
                if role == 0:
                    allowed = free[slot] >= generation-1
                    if reuse == 0 and slot == 0 and outer:
                        allowed &= min(free) >= generation-1
                elif role == 1:
                    allowed = ready[slot] >= generation and partial_free[slot] >= generation-1
                else:
                    allowed = partial_ready[slot] >= generation
                if allowed:
                    choices.append(role)
            assert choices, "schedule deadlock"
            role = rng.choice(choices)
            q = cursors[role]
            slot, generation = q % 2, q // 2 + 1
            outer, reuse = q // (reuse_count*2), q // 2 % reuse_count
            if role == 0:
                if reuse == 0:
                    resident[slot] = outer
                ready[slot] = generation
            elif role == 1:
                assert resident[slot] == outer
                free[slot] = partial_ready[slot] = generation
            else:
                partial_free[slot] = generation
            cursors[role] += 1


def test_formal_async_codegen():
    path = Path(__file__).resolve().parents[3] / "examples/hexagon/gemm/gemm_prepacked_dma.py"
    spec = importlib.util.spec_from_file_location("resident_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = tilelang.engine.lower(module.kernel(M=128, N=128, K=64,
        BM=64, BN=64, BK=32, transform_output=True, schedule="async",
        reuse_b=True, load=1, merge_edges=True), target="hexagon",
        enable_host_codegen=False, enable_device_compile=False)
    source = result.kernel_source
    assert "if (initial % 2u) return TL_WG_INVALID;" in source
    assert "4,3,2,3,8,0,0," in source
    assert "tl_wg_team_barrier" not in source
    assert "tl_wg_group_wait((const tl_wg_worker*)wg_worker, 3," in source
    assert "tl::dma_copy_2d_wait(C," in source
    assert "float" not in source
