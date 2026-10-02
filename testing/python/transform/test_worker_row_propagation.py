import tilelang
from tilelang import language as T
from tvm import IRModule, tirx, get_global_func
from tvm.target import Target


def test_matrix_producer_inherits_state_partition():
    @T.prim_func
    def kernel(O:T.Tensor((128,), 'float32')):
        with T.Kernel(1, threads=2):
            matrix=T.alloc_shared((128,32), 'float32')
            state=T.alloc_shared((128,), 'float32')
            T.clear(state)
            for k in T.Pipelined(2,num_stages=2,annotations={'tl.workergroup_schedule':'async'}):
                with T.pipeline_stage('map',engine='hvx',workers=2,physical_owner='rows'):
                    for r in T.Parallel(128):
                        for c in T.vectorized(32):
                            matrix[r,c]=T.float32(2)
                    for r in T.Parallel(128):
                        state[r]=matrix[r,0]+state[r]
                with T.pipeline_stage('output',engine='hvx',workers=2,physical_owner='rows'):
                    for r in T.Parallel(128):
                        O[r]=state[r]
    mod=tirx.transform.BindTarget(Target('hexagon'))(IRModule({'main':kernel}))
    mod=tilelang.transform.MaterializeKernelLaunch()(mod)
    mod=get_global_func('tl.hexagon.transform.PlanWorkerPipeline')(2,64,65536)(mod)
    mod=get_global_func('tl.hexagon.transform.InferWorkerOwnership')()(mod)
    layouts=[]; columns=[]
    def visit(n):
        if isinstance(n,tirx.For) and n.kind==tirx.ForKind.PARALLEL:
            layouts.append(n.annotations['parallel_loop_layout'])
        if isinstance(n,tirx.For) and n.kind==tirx.ForKind.VECTORIZED:
            columns.append(int(n.extent))
    tirx.stmt_functor.post_order_visit(mod['main'].body,visit)
    assert len(layouts)==3
    from tvm.ir import assert_structural_equal
    for layout in layouts[1:]:
        assert_structural_equal(layouts[0],layout)
    assert columns==[32]
