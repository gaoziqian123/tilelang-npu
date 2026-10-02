from pathlib import Path
import tilelang
from tilelang import language as T
from tvm import IRModule, tirx, get_global_func
from tvm.target import Target


def test_three_role_production_chain(tmp_path):
    @T.prim_func
    def kernel(A: T.Tensor((32, 64), "float16"),
               B: T.Tensor((64, 64), "float16"),
               C: T.Tensor((32, 64), "float32")):
        with T.Kernel(1, threads=6):
            a = T.alloc_shared((32, 32), "float16")
            b = T.alloc_shared((32, 32), "float16")
            partial = T.alloc_shared((32, 32), "float16")
            c = T.alloc_fragment((32, 32), "float32")
            for bn in T.serial(2):
                T.clear(c)
                for bk in T.Pipelined(2, num_stages=2, annotations={"tl.workergroup_schedule": "async"}):
                    with T.pipeline_stage("feed", engine="hvx", workers=3):
                        T.copy(A[0:32, bk*32:bk*32+32], a)
                        T.copy(B[bn*32:bn*32+32, bk*32:bk*32+32], b)
                    with T.pipeline_stage("dot", engine="hmx", workers=1):
                        T.gemm(a, b, partial, transpose_B=True, clear_accum=True)
                    with T.pipeline_stage("sum", engine="hvx", workers=2):
                        for i, j in T.Parallel(32, 32):
                            c[i, j] += T.cast(partial[i, j], "float32")
                T.copy(c, C[0:32, bn*32:bn*32+32])
    target = Target("hexagon")
    mod = tirx.transform.BindTarget(target)(IRModule({"main": kernel}))
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = get_global_func("tl.hexagon.transform.PlanWorkerPipeline")(6, 64, 8388608)(mod)
    mod = get_global_func("tl.hexagon.transform.InferWorkerOwnership")()(mod)
    scopes = []
    tirx.stmt_functor.post_order_visit(mod["main"].body, lambda n:
        scopes.append(n) if isinstance(n, tirx.AttrStmt) and n.attr_key == "tl.workergroup_local" else None)
    assert [str(s.node["name"]) for s in scopes] == ["sum", "feed", "dot", "sum", "sum"]
    assert [int(s.node["local_size"]) for s in scopes] == [2, 3, 1, 2, 2]
    with target:
        mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
    mod = get_global_func("tl.hexagon.transform.LowerWorkerSync")()(mod)
    mod = get_global_func("tl.hexagon.transform.PlanWorkerPhysicalAllocations")(8388608)(mod)
    calls = []
    tirx.stmt_functor.post_order_visit(mod["main"].body, lambda n:
        calls.append(n.op.name) if isinstance(n, tirx.Call) and hasattr(n.op, "name") else None)
    assert "tl.hexagon.hmx_store_after" in calls
    assert "tl.hexagon.hmx_mma_deep" in calls
    assert "tirx.tvm_storage_sync" not in calls
    assert "tl.hexagon.hmx_acc_read" not in calls
    plan = mod["main"].attrs["tl.workergroup_physical_allocations"]
    assert sum(int(p["depth"]) == 2 for p in plan) == 3
    assert all(int(p["depth"]) == 1 for p in plan if p["buffer"].dtype == "float32")
    allocations = []
    tirx.stmt_functor.post_order_visit(mod["main"].body, lambda n:
        allocations.extend(n.alloc_buffers) if isinstance(n, tirx.SBlock) else None)
    c_alloc = [b for b in allocations if str(b.name) == "c"]
    assert len(c_alloc) == 1 and c_alloc[0].dtype == "float32"
    assert len(c_alloc[0].shape) == 1 and int(c_alloc[0].shape[0]) == 512
    (tmp_path / "three_role_final.tir").write_text(mod.script(show_meta=True))
    callback = get_global_func("tl.hexagon.transform.BuildWorkerCallbacks")()(mod)
    names = []
    tirx.stmt_functor.post_order_visit(callback["main"].body, lambda n:
        names.append(n.op.name) if isinstance(n, tirx.Call) and hasattr(n.op, "name") else None)
    assert names.count("tl.hexagon.workergroup_wait") == 6
    assert names.count("tl.hexagon.workergroup_publish") == 6
    assert names.count("tl.hexagon.workergroup_complete") == 1
    (tmp_path / "three_role_callback.tir").write_text(callback.script(show_meta=True))
    callback = tilelang.transform.LowerOpaqueBlock()(callback)
    callback = tilelang.transform.FlattenBuffer()(callback)
    callback = tirx.transform.Simplify()(callback)
    callback = tirx.transform.LowerIntrin()(callback)
    source = get_global_func("target.build.tilelang_hexagon_without_compile")(callback, target).inspect_source()
    (tmp_path / "generated.c").write_text(source)
    assert "_owner(" in source and "tl_wg_run(" in source
