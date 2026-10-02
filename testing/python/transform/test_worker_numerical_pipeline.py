"""Shared numerical lowering with and without group context."""
import pytest
import tilelang
from tvm import IRModule, tirx
from tvm.ir.transform import PassContext
from tvm.target import Target
from tilelang.hexagon.pipeline import _NumericalBeforeAllocation, _NumericalAfterFlatten


@pytest.mark.parametrize("cast", [False, True])
@pytest.mark.parametrize("options", [{}, {"tl.disable_fuse_pointwise_stages": False},
                                      {"tl.enable_ordered_accumulator_promotion": True}])
def test_same_body_group_numeric(cast, options):
    a = tirx.decl_buffer((128,), "float16" if cast else "float32", name="a")
    b = tirx.decl_buffer((128,), "float32", name="b")
    i = tirx.Var("i", "int32")
    value = tirx.Cast("float32", a[i]) if cast else a[i]
    body = tirx.For(i, 0, 128, tirx.ForKind.VECTORIZED, tirx.BufferStore(b, value + 1, [i]))
    spec = {"engine": "hvx", "first_worker": 0, "local_id": 0, "local_size": 1}
    wrapped = tirx.AttrStmt(spec, "tl.workergroup_local", 1, body)
    def lower(s):
        mod = IRModule({"main": tirx.PrimFunc([a.data, b.data], s,
                                            buffer_map={a.data: a, b.data: b})})
        with Target("hexagon"), PassContext(config=options) as ctx:
            mod = _NumericalBeforeAllocation(mod, ctx)
            mod = tilelang.transform.FlattenBuffer()(mod)
            mod = _NumericalAfterFlatten(mod, ctx)
        return mod["main"].body
    ordinary, worker = lower(body), lower(wrapped)
    assert isinstance(worker, tirx.AttrStmt)
    from tvm.ir import assert_structural_equal
    assert_structural_equal(ordinary, worker.body)


def test_normal_helper_pass_order(monkeypatch):
    from tilelang.hexagon import pipeline
    from tilelang.hexagon import vector_lowering, vector_verify
    order = []
    names = ["DecoupleTypeCast", "LegalizeVectorizedLoop", "PromoteOrderedAccumulator",
             "LegalizeSafeMemoryAccess", "LowerAccessPtr", "Simplify", "HoistNonRestrictParams",
             "PlanLocalRowReduce", "VectorizeLoop", "FuseLocalRowMap"]
    for name in names:
        monkeypatch.setattr(tilelang.transform, name,
                            lambda *args, _name=name, **kwargs: lambda mod: (order.append(_name), mod)[1])
    monkeypatch.setattr(pipeline, "FusePointwiseStages",
                        lambda: lambda mod: (order.append("FusePointwiseStages"), mod)[1])
    monkeypatch.setattr(vector_lowering, "VectorLowering",
                        lambda: lambda mod: (order.append("VectorLowering"), mod)[1])
    monkeypatch.setattr(vector_verify, "VerifyVectorRequired",
                        lambda: lambda mod: (order.append("VerifyVectorRequired"), mod)[1])
    with PassContext(config={"tl.disable_fuse_pointwise_stages": False,
                             "tl.enable_ordered_accumulator_promotion": True,
                             "tl.hexagon.plan_local_row_reduce": True,
                             "tl.hexagon.fuse_local_row_map": True}) as ctx:
        pipeline._NumericalBeforeAllocation(None, ctx)
        pipeline._NumericalAfterFlatten(None, ctx)
    assert order == ["DecoupleTypeCast", "FusePointwiseStages", *names[1:-1],
                     "VectorLowering", "VerifyVectorRequired", names[-1]]
