"""Conservative final-IR gate for explicitly vector-required functions."""
from tvm import tirx as tir


def verify_vector_required(func):
    if not func.attrs or not func.attrs.get("tl.vector_required", False):
        return
    addresses = set()
    control = set()
    # Explicit final extraction is allowed once, outside loops, as an Evaluate.
    # No extraction feeding casts/arithmetic or multiple lane extraction API.
    extracts=[]
    terminal_extracts=set()
    loop_extracts=set()
    def is_extract(n):
        return (isinstance(n,tir.Call) and str(getattr(n.op,"name",""))=="tl.hexagon.vector_io"
                and len(n.args)>0 and isinstance(n.args[0],tir.StringImm)
                and n.args[0].value in ("extract16","extract32"))
    effects={"tl.hexagon."+n for n in (
        "hmx_mma_deep","hmx_mma_f16","hmx_clear_acc","hmx_init_scale","hmx_store_after",
        "workergroup_barrier","workergroup_wait","workergroup_publish","workergroup_complete",
        "workergroup_stage_enter","workergroup_stage_exit","workergroup_team_barrier",
        "dma_copy_2d_wait","dma_submit","dma_wait","require","profile_mark")}
    def mark(expr):
        tir.stmt_functor.post_order_visit(expr, lambda n: control.add(n))
    def address(node):
        if is_extract(node): extracts.append(node)
        if isinstance(node,tir.Evaluate) and is_extract(node.value): terminal_extracts.add(node.value)
        if isinstance(node,tir.For):
            tir.stmt_functor.post_order_visit(node.body,lambda n: loop_extracts.add(n) if is_extract(n) else None)
        if isinstance(node, tir.Call) and str(getattr(node.op, "name", "")) == "tirx.address_of":
            if len(node.args) == 1 and isinstance(node.args[0], tir.BufferLoad):
                addresses.add(node.args[0])
                for index in node.args[0].indices:
                    mark(index)
        if isinstance(node, tir.For):
            mark(node.min)
            mark(node.extent)
        if isinstance(node, tir.IfThenElse):
            mark(node.condition)
    tir.stmt_functor.post_order_visit(func.body, address)
    if len(extracts)>1 or any(e not in terminal_extracts or e in loop_extracts for e in extracts):
        raise ValueError("vector_required: extract may only be one explicit terminal result outside loops")
    def check(node):
        if isinstance(node, tir.BufferLoad) and node not in addresses:
            raise ValueError("vector_required: use proven vector load, scalar/generic load forbidden")
        if isinstance(node, tir.BufferStore):
            raise ValueError("vector_required: use proven vector store")
        if isinstance(node, tir.Call):
            name = str(getattr(node.op, "name", ""))
            if name not in ("tl.hexagon.vector_leaf", "tl.hexagon.vector_io", "tirx.address_of") and name not in effects:
                raise ValueError("vector_required: unverified call " + name)
        if isinstance(node, (tir.Add, tir.Sub, tir.Mul, tir.Div, tir.Min, tir.Max, tir.Cast,
                             tir.Select, tir.Broadcast, tir.Shuffle,
                             tir.LT, tir.LE, tir.GT, tir.GE, tir.EQ, tir.NE,
                             tir.Not, tir.And, tir.Or)):
            dtype = str(node.dtype)
            if "x" in dtype or dtype.startswith("float") or node not in control:
                # Constant casts for splat bits are harmless, not data fallback.
                if isinstance(node, tir.Cast) and isinstance(node.value, tir.IntImm):
                    return
                raise ValueError("vector_required: generic data expression needs physical lowering")
    tir.stmt_functor.post_order_visit(func.body, check)


def VerifyVectorRequired():
    @tir.transform.prim_func_pass(opt_level=0)
    def verify(func, mod, ctx):
        verify_vector_required(func)
        return func
    return verify
