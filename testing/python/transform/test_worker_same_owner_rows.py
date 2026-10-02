import pytest
import tilelang
from tilelang import language as T
from tvm import IRModule, tirx, get_global_func
from tvm.target import Target


def plan(shift=0):
    @T.prim_func
    def kernel(O:T.Tensor((32,), 'float32')):
        with T.Kernel(1,threads=2):
            state=T.alloc_shared((32,), 'float32')
            T.clear(state)
            for k in T.Pipelined(3,num_stages=2,annotations={'tl.workergroup_schedule':'async'}):
                with T.pipeline_stage('first',engine='hvx',workers=2,physical_owner='rows'):
                    for r in T.Parallel(32):
                        state[r]=state[r]+T.float32(1)
                with T.pipeline_stage('second',engine='hvx',workers=2,physical_owner='rows'):
                    for r in T.Parallel(32-shift):
                        O[r]=state[r+shift]
    target=Target('hexagon')
    mod=tirx.transform.BindTarget(target)(IRModule({'main':kernel}))
    mod=tilelang.transform.MaterializeKernelLaunch()(mod)
    return get_global_func('tl.hexagon.transform.PlanWorkerPipeline')(2,64,8388608)(mod)


def test_same_rows():
    plan()


def test_changed_rows_reject():
    with pytest.raises(ValueError,match='row partition'):
        plan(1)
