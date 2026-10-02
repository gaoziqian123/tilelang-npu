"""Single-outstanding native DMA lexical lifetime verifier.

Conservative: while outstanding, only pure integer/register calculations and
the matching wait are allowed. No unknown calls, memory, scope exit or publish.
"""
from tvm import tirx as tir


def VerifyNativeDMA():
    @tir.transform.prim_func_pass(opt_level=0)
    def verify(func, mod, ctx):
        active = None
        def walk(stmt):
            nonlocal active
            if isinstance(stmt, tir.SeqStmt):
                for child in stmt.seq: walk(child)
                return
            if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
                c=stmt.value
                name=str(getattr(c.op,'name',''))
                if name=='tl.hexagon.dma_submit':
                    if active is not None: raise ValueError('DMA duplicate outstanding ticket')
                    if len(c.args)!=7 or not isinstance(c.args[0],tir.IntImm):
                        raise ValueError('DMA ticket must be static')
                    active=int(c.args[0]); return
                if name=='tl.hexagon.dma_wait':
                    if len(c.args)!=1 or not isinstance(c.args[0],tir.IntImm) or active!=int(c.args[0]):
                        raise ValueError('DMA stale/mismatched wait ticket')
                    active=None; return
            if active is not None:
                if isinstance(stmt,(tir.Bind,tir.Evaluate)):
                    value=stmt.value
                    safe=True
                    def inspect(n):
                        nonlocal safe
                        if isinstance(n,tir.BufferLoad): safe=False
                        if isinstance(n,tir.Call):
                            name=str(getattr(n.op,'name',''))
                            if name=='tl.hexagon.vector_leaf': return
                            if name=='tl.hexagon.vector_io' and len(n.args) and isinstance(n.args[0],tir.StringImm) and n.args[0].value in ('bitcast','splat16','splat32'):
                                return
                            safe=False
                    tir.stmt_functor.post_order_visit(value,inspect)
                    if safe: return
                raise ValueError('DMA requires wait before memory, publish, control or scope exit')
            # Lifetimes cannot cross a control/scope boundary.
            if isinstance(stmt,tir.For): walk(stmt.body)
            elif isinstance(stmt,tir.AttrStmt): walk(stmt.body)
            elif isinstance(stmt,tir.IfThenElse):
                walk(stmt.then_case)
                if active is not None: raise ValueError('DMA escapes branch')
                if stmt.else_case is not None: walk(stmt.else_case)
            elif isinstance(stmt,tir.SBlockRealize): walk(stmt.block.body)
            if active is not None: raise ValueError('DMA ticket escapes scope')
        walk(func.body)
        if active is not None: raise ValueError('DMA missing wait')
        return func
    return verify
