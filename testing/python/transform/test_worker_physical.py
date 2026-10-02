"""Post-layout physical allocation planning must not mutate logical ranks."""
import pytest
import tilelang
from tvm import IRModule, tirx, get_global_func


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_physical_stride_and_rank(depth):
    b = tirx.decl_buffer((32, 32), "float16", name="physical", scope="shared",
                        strides=[64, 1], data_alignment=128)
    k = tirx.Var("k", "int32")
    loop = tirx.For(k, 0, 4, tirx.ForKind.SERIAL, tirx.Evaluate(0), annotations={
        "tl.workergroup_slots": [{"buffer": b, "depth": depth}]})
    body = tirx.SeqStmt([tirx.AllocBuffer(b), loop])
    mod = IRModule({"main": tirx.PrimFunc([], body)})
    result = get_global_func("tl.hexagon.transform.PlanWorkerPhysicalAllocations")(65536)(mod)
    f = result["main"]
    plan = f.attrs["tl.workergroup_physical_allocations"]
    assert len(plan) == 1
    assert list(plan[0]["buffer"].shape) == [32, 32]
    # (31*64+31+1)*2 = 4032 bytes, rounded to 4096, NOT 2048.
    assert int(plan[0]["bytes_per_slot"]) == 4096
    assert int(plan[0]["depth"]) == depth
    assert int(f.attrs["tl.workergroup_vtcm_bytes"]) == 4096 * depth
    assert f.body.same_as(body)
    with pytest.raises(ValueError, match="budget exceeded"):
        get_global_func("tl.hexagon.transform.PlanWorkerPhysicalAllocations")(4096 * depth - 1)(mod)
