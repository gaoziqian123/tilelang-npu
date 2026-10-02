"""Exercise production layout/TileOp passes, not the host diagnostic emitter."""
import tilelang
from tilelang import language as T
from tvm import IRModule, tirx
from tvm.target import Target


def test_group_context_shared_hmx_asm_lowering():
    @T.prim_func
    def kernel(A: T.Tensor((32, 32), "float16"),
               B: T.Tensor((32, 32), "float16"),
               C: T.Tensor((32, 32), "float16")):
        with T.Kernel(1, threads=6):
            a = T.alloc_shared((32, 32), "float16")
            b = T.alloc_shared((32, 32), "float16")
            c = T.alloc_shared((32, 32), "float16")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    target = Target("hexagon")
    mod = tirx.transform.BindTarget(target)(IRModule({"main": kernel}))
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    physical = []
    def collect(n):
        if isinstance(n, tirx.AttrStmt) and n.attr_key == "thread_extent" and n.node.thread_tag == "threadIdx.x":
            physical.append(n.node.var)
    tirx.stmt_functor.post_order_visit(mod["main"].body, collect)
    assert len(physical) == 1
    # Isolate the HMX body with the same context the worker materializer emits.
    def wrap(n):
        if isinstance(n, tirx.Evaluate) and isinstance(n.value, tirx.Call) and str(n.value.op.name) == "tl.tileop.gemm":
            return tirx.AttrStmt({"engine": "hmx", "first_worker": 0,
                                 "local_id": physical[0], "local_size": 1},
                                "tl.workergroup_local", 1, n)
        return None
    f = mod["main"]
    mod.update_func(mod.get_global_var("main"), f.with_body(tirx.stmt_functor.ir_transform(f.body, None, wrap)))
    with target:
        mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
    calls = []
    def collect_calls(n):
        if isinstance(n, tirx.Call):
            calls.append(n)
    tirx.stmt_functor.post_order_visit(mod["main"].body, collect_calls)
    names = {c.op.name for c in calls if hasattr(c.op, "name")}
    assert "tl.hexagon.hmx_mma_deep" in names
    assert "tl.hexagon.hmx_store_after" in names
    assert "tl.hexagon.hmx_init_scale" in names
    assert "tl.hexagon.hmx_acc_read" not in names
    assert not any("hexkl" in str(c) or "gemm_accumulator_f16" in str(c) for c in calls)
    assert "tl.workergroup_local" in mod.script()
