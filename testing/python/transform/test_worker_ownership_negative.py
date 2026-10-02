import pytest
import tilelang
from tilelang import language as T
from tvm import IRModule, tirx, get_global_func
from tvm.target import Target


@pytest.mark.parametrize("opaque", [False, True])
def test_outside_conflicting_or_opaque(opaque):
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=2):
            x = T.alloc_shared((32,), "float32")
            y = T.alloc_shared((32,), "float32")
            for k in T.Pipelined(2, num_stages=1):
                with T.pipeline_stage("one", engine="hvx", workers=1):
                    T.clear(x)
                with T.pipeline_stage("two", engine="hvx", workers=1):
                    T.clear(y)
            if T.meta_var(opaque):
                T.evaluate(T.call_extern("int32", "unknown_effect"))
            else:
                T.copy(x, y)
    target = Target("hexagon")
    mod = tirx.transform.BindTarget(target)(IRModule({"main": kernel}))
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = get_global_func("tl.hexagon.transform.PlanWorkerPipeline")(6, 64, 8192)(mod)
    with pytest.raises(ValueError, match="opaque call effects" if opaque else "conflicting ownership"):
        get_global_func("tl.hexagon.transform.InferWorkerOwnership")()(mod)
