"""Fail-closed ownership/lifetime gate for closed group reductions."""
from tvm import tirx as tir


def VerifyGroupCollectives():
    @tir.transform.prim_func_pass(opt_level=0)
    def verify(func, mod, ctx):
        stack = []
        stage = []
        allowed_loads = set()
        scratch_owners = {}
        allocations = set()
        def allocated(n):
            if isinstance(n, tir.AllocBuffer):
                allocations.add(n.buffer.data)
            if isinstance(n, tir.SBlock):
                allocations.update(b.data for b in n.alloc_buffers)
        tir.stmt_functor.post_order_visit(func.body, allocated)
        def before(n):
            stack.append(n)
            if isinstance(n, tir.AttrStmt) and n.attr_key == 'tl.pipeline_stage':
                stage.append((n.node, len(stack)))
            if not isinstance(n, tir.Call) or str(getattr(n.op, 'name', '')) != 'tl.hexagon.group_reduce':
                return
            if not stage:
                raise ValueError('group_reduce requires active stage')
            spec, depth = stage[-1]
            if any(isinstance(p, (tir.For, tir.IfThenElse, tir.Select)) for p in stack[depth:-1]):
                raise ValueError('group_reduce requires unconditional stage-level invocation')
            if len(n.args) != 7 or str(n.dtype) != 'float32':
                raise ValueError('group_reduce invalid ABI')
            op, workers, owner, value, seed, ptr, size = n.args
            if str(spec['engine']) != 'hvx' or str(spec.get('physical_owner', spec['name'])) != owner.value:
                raise ValueError('group_reduce wrong group owner')
            if int(workers) not in (1,2,4,8,16,32,64) or int(spec['workers']) != int(workers):
                raise ValueError('group_reduce wrong worker count')
            if op.value not in ('sum','max') or str(value.dtype) != 'float32' or str(seed.dtype) != 'float32':
                raise ValueError('group_reduce requires FP32 sum/max')
            if not isinstance(ptr, tir.Call) or str(getattr(ptr.op,'name','')) != 'tirx.address_of':
                raise ValueError('group_reduce requires owned scratch address')
            load = ptr.args[0]
            if not isinstance(load, tir.BufferLoad):
                raise ValueError('group_reduce scratch descriptor missing')
            b = load.buffer
            if b.data not in allocations or b.scope() not in ('shared','shared.dyn') or str(b.dtype) != 'float32' or len(b.shape)!=1 or int(load.indices[0])!=0 or int(size)!=int(b.shape[0]) or int(size)<int(workers):
                raise ValueError('group_reduce needs exclusive allocated shared FP32 scratch')
            if b.data in scratch_owners and scratch_owners[b.data] != owner.value:
                raise ValueError('group_reduce scratch aliases another group')
            scratch_owners[b.data] = owner.value
            allowed_loads.add(load)
        def after(n):
            if isinstance(n, tir.AttrStmt) and n.attr_key == 'tl.pipeline_stage':
                stage.pop()
            stack.pop()
        tir.stmt_functor.ir_transform(func.body, before, after, None)
        def audit(n):
            if isinstance(n, tir.BufferLoad) and n.buffer.data in scratch_owners and n not in allowed_loads:
                raise ValueError('group_reduce scratch partial escaped')
            if isinstance(n, tir.BufferStore) and n.buffer.data in scratch_owners:
                raise ValueError('group_reduce scratch externally modified')
            if isinstance(n, tir.Var) and n in scratch_owners:
                raise ValueError('group_reduce scratch pointer escaped')
        tir.stmt_functor.post_order_visit(func.body, audit)
        return func
    return verify
