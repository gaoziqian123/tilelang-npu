"""TileLang Hexagon host-side TIR pass chain.

The Hexagon C emitter is intentionally dumb: it lowers a small set of already
canonical Hexagon TIR intrinsics.  Analysis, peephole fusion, storage planning,
and rule verification live here so they can run before code generation and fail
with rule-numbered diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tvm import tirx
from tvm.tirx import (
    AttrStmt,
    Buffer,
    BufferLoad,
    BufferStore,
    Call,
    Evaluate,
    For,
    IfThenElse,
    IntImm,
    PrimFunc,
    PyStmtExprMutator,
    SBlock,
    SBlockRealize,
    SeqStmt,
)
from tvm.tirx.stmt_functor import post_order_visit, substitute
from tvm.tirx.transform import prim_func_pass

from .emitter import HexagonEmitError, VTCM_BUDGET
from .language.layout import detect_layout_mode


def _bool_arg(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    iv = _i64(x)
    return bool(iv) if iv is not None else bool(x)


def _i64(x: Any) -> int | None:
    if isinstance(x, IntImm):
        return int(x.value)
    if isinstance(x, int):
        return x
    try:
        if hasattr(x, "value"):
            return int(x.value)
    except Exception:
        pass
    return None


def _align(x: int, a: int = 128) -> int:
    return (x + a - 1) & ~(a - 1)


def _dtype_bytes(dtype: str) -> int:
    if dtype in ("float16", "bfloat16", "uint16", "int16"):
        return 2
    if dtype in ("float32", "uint32", "int32"):
        return 4
    if dtype in ("uint8", "int8", "bool"):
        return 1
    raise HexagonEmitError(f"R3: 不支持计算 dtype={dtype} 的 VTCM 占用")


def _shape_numel(shape: Any) -> int | None:
    n = 1
    for dim in shape:
        iv = _i64(dim)
        if iv is None:
            return None
        n *= iv
    return n


def _buffer_scope(buf: Buffer | None) -> str:
    if buf is None:
        return "unknown"
    try:
        return str(buf.scope())
    except Exception:
        return "global"


def _buffer_name(buf: Buffer | None) -> str:
    if buf is None:
        return "unknown"
    return getattr(getattr(buf, "data", None), "name", None) or getattr(buf, "name", "buf")


def _mark_write(writes: set[str], buf: Buffer | None) -> None:
    if buf is None:
        return
    writes.add(_buffer_name(buf))
    writes.add(_var_name(getattr(buf, "data", "")))


def _is_vtcm(scope: str) -> bool:
    return scope.startswith("vtcm") or scope.startswith("shared")


def _is_wscratch(scope: str) -> bool:
    return scope.startswith("wscratch")


def _is_acc(scope: str) -> bool:
    return scope == "hmx.acc" or "fragment" in scope


def _scope_layout_suffix(scope: str) -> str | None:
    suffix = scope.rsplit(".", 1)[-1]
    return suffix if suffix in ("ah", "wh") else None


def _ann_str(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "value"):
        return str(obj.value)
    return str(obj)


def _call_op_name(call: Call) -> str:
    return getattr(call.op, "name", str(call.op))


def _loc(op: Any) -> str:
    span = getattr(op, "span", None)
    return f" at {span}" if span else ""


def _var_name(v: Any) -> str:
    return getattr(v, "name", None) or getattr(v, "name_hint", None) or str(v)


def _region_buffer(expr: Any) -> Buffer | None:
    if isinstance(expr, Call) and _call_op_name(expr) == "tl.region":
        return getattr(expr.args[0], "buffer", None)
    return getattr(expr, "buffer", None)


def _region_extents(expr: Any) -> list[int | None]:
    if isinstance(expr, Call) and _call_op_name(expr) == "tl.region":
        return [_i64(a) for a in expr.args[2:]]
    region = getattr(expr, "region", None)
    if region is not None:
        return [_i64(r.extent) for r in region]
    return []


def _is_extern(call: Call, name: str | None = None) -> bool:
    if not _call_op_name(call).endswith("call_pure_extern") or not call.args:
        return False
    callee = _extern_callee(call)
    return callee == name if name is not None else bool(callee and callee.startswith("hexagon."))


def _extern_callee(call: Call) -> str | None:
    if not _call_op_name(call).endswith("call_pure_extern"):
        return None
    for arg in call.args[:3]:
        s = _ann_str(arg)
        if s and s.startswith("hexagon."):
            return s
    return None


def _extern_arg_offset(call: Call) -> int:
    """Return index immediately after the hexagon callee name."""

    for i, arg in enumerate(call.args[:3]):
        s = _ann_str(arg)
        if s and s.startswith("hexagon."):
            return i + 1
    return 0


def _is_vector_for(op: Any) -> bool:
    return isinstance(op, For) and ("vectorized" in str(getattr(op, "kind", "")).lower() or str(getattr(op, "kind", "")) == "2")


def _is_mul_expr(e: Any) -> bool:
    return type(e).__name__ == "Mul"


def _same_buffer_data(a: Buffer | None, b: Buffer | None) -> bool:
    if a is None or b is None:
        return False
    return _var_name(getattr(a, "data", "")) == _var_name(getattr(b, "data", ""))


def _arg_matches_buffer_data(arg: Any, buf: Buffer | None) -> bool:
    if buf is None:
        return False
    abuf = getattr(arg, "buffer", None)
    if abuf is not None:
        return _same_buffer_data(abuf, buf)
    return _var_name(arg) == _var_name(getattr(buf, "data", ""))


@dataclass
class _ProdReduce:
    loop: For
    store: BufferStore
    reduce_call: Call
    x: Any
    y: Any
    dst: Any


def _match_prod_reduce(loop: Any, red_stmt: Any) -> _ProdReduce | None:
    if not _is_vector_for(loop):
        return None
    store = loop.body
    if isinstance(store, SeqStmt) and len(store.seq) == 1:
        store = store.seq[0]
    if not isinstance(store, BufferStore) or not _is_mul_expr(store.value):
        return None
    if not isinstance(store.value.a, BufferLoad) or not isinstance(store.value.b, BufferLoad):
        return None
    if not (isinstance(red_stmt, Evaluate) and isinstance(red_stmt.value, Call)):
        return None
    call = red_stmt.value
    if not _is_extern(call, "hexagon.reduce_sum128") or len(call.args) < 3:
        return None
    src = call.args[1]
    if not _arg_matches_buffer_data(src, store.buffer):
        return None
    zero = IntImm(str(getattr(loop.loop_var, "dtype", "int32")), 0)
    return _ProdReduce(
        loop=loop,
        store=store,
        reduce_call=call,
        x=substitute(store.value.a, {loop.loop_var: zero}),
        y=substitute(store.value.b, {loop.loop_var: zero}),
        dst=call.args[2],
    )


def _vector_mul_stores(loop: Any) -> list[BufferStore]:
    if not _is_vector_for(loop):
        return []
    body = loop.body
    seq = list(body.seq) if isinstance(body, SeqStmt) else [body]
    out: list[BufferStore] = []
    for stmt in seq:
        if isinstance(stmt, BufferStore) and _is_mul_expr(stmt.value):
            out.append(stmt)
        else:
            return []
    return out


def _match_store_reduce(loop: Any, store: BufferStore, red_stmt: Any) -> _ProdReduce | None:
    if not isinstance(store.value.a, BufferLoad) or not isinstance(store.value.b, BufferLoad):
        return None
    if not (isinstance(red_stmt, Evaluate) and isinstance(red_stmt.value, Call)):
        return None
    call = red_stmt.value
    if not _is_extern(call, "hexagon.reduce_sum128") or len(call.args) < 3:
        return None
    src = call.args[1]
    if not _arg_matches_buffer_data(src, store.buffer):
        return None
    zero = IntImm(str(getattr(loop.loop_var, "dtype", "int32")), 0)
    return _ProdReduce(
        loop=loop,
        store=store,
        reduce_call=call,
        x=substitute(store.value.a, {loop.loop_var: zero}),
        y=substitute(store.value.b, {loop.loop_var: zero}),
        dst=call.args[2],
    )


def _make_eval(callee: str, args: list[Any], span: Any = None) -> Evaluate:
    return Evaluate(tirx.call_pure_extern("handle", callee, *args, span=span))


@tirx.functor.mutator
class _ProductReduceFusionMutator(PyStmtExprMutator):
    def visit_seq_stmt_(self, op: SeqStmt):
        visited = [self.visit_stmt(s) for s in op.seq]
        out: list[Any] = []
        i = 0
        while i < len(visited):
            if i + 2 < len(visited):
                stores = _vector_mul_stores(visited[i])
                if len(stores) == 2:
                    p0 = _match_store_reduce(visited[i], stores[0], visited[i + 1])
                    p1 = _match_store_reduce(visited[i], stores[1], visited[i + 2])
                    if p0 is not None and p1 is not None:
                        out.append(_make_eval("hexagon.reduce_prod2_128", [p0.x, p0.y, p0.dst, p1.x, p1.y, p1.dst], getattr(p0.reduce_call, "span", None)))
                        i += 3
                        continue
            if i + 3 < len(visited):
                p0 = _match_prod_reduce(visited[i], visited[i + 1])
                p1 = _match_prod_reduce(visited[i + 2], visited[i + 3])
                if p0 is not None and p1 is not None:
                    out.append(_make_eval("hexagon.reduce_prod2_128", [p0.x, p0.y, p0.dst, p1.x, p1.y, p1.dst], getattr(p0.reduce_call, "span", None)))
                    i += 4
                    continue
            if i + 1 < len(visited):
                p = _match_prod_reduce(visited[i], visited[i + 1])
                if p is not None:
                    out.append(_make_eval("hexagon.reduce_prod128", [p.x, p.y, p.dst], getattr(p.reduce_call, "span", None)))
                    i += 2
                    continue
            out.append(visited[i])
            i += 1
        if out == list(op.seq):
            return op
        if len(out) == 1:
            return out[0]
        return SeqStmt(out)


def HexagonProductReduceFusion():
    """Fuse vectorized Mul scratch + reduce_sum128 into product-reduce intrinsics."""

    def pass_fn(func: PrimFunc, mod, ctx):
        mut = _ProductReduceFusionMutator()
        return func.with_body(mut.visit_stmt(func.body), span=func.span)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.hexagon.ProductReduceFusion")


def HexagonWriteSet():
    """Attach ``hexagon.write_set``: global buffers written by stores/extern outs."""

    def pass_fn(func: PrimFunc, mod, ctx):
        writes: set[str] = set()
        data_to_buf: dict[str, Buffer] = {
            _var_name(getattr(buf, "data", "")): buf
            for _, buf in func.buffer_map.items()
            if _buffer_scope(buf) == "global"
        }

        def visit(node: Any) -> None:
            if isinstance(node, BufferStore) and _buffer_scope(node.buffer) == "global":
                _mark_write(writes, node.buffer)
            if isinstance(node, Call) and _is_extern(node):
                callee = _extern_callee(node)
                off = _extern_arg_offset(node)
                outs = []
                if callee == "hexagon.reduce_sum128" and len(node.args) >= off + 2:
                    outs = [node.args[off + 1]]
                elif callee in ("hexagon.reduce_max32", "hexagon.reduce_max128") and len(node.args) >= off + 2:
                    outs = [node.args[off + 1]]
                elif callee == "hexagon.reduce_prod128" and len(node.args) >= off + 3:
                    outs = [node.args[off + 2]]
                elif callee == "hexagon.reduce_prod2_128" and len(node.args) >= off + 6:
                    outs = [node.args[off + 2], node.args[off + 5]]
                elif callee in ("hexagon.copy_acc_rm", "hexagon.copy_ah_rm", "hexagon.copy_ddr") and len(node.args) >= off + 3:
                    outs = [node.args[off + 1]]
                for arg in outs:
                    buf = getattr(arg, "buffer", None) or data_to_buf.get(_var_name(arg))
                    if buf is not None and _buffer_scope(buf) == "global":
                        _mark_write(writes, buf)
            if isinstance(node, Call) and _call_op_name(node) == "tl.tileop.reduce" and len(node.args) >= 2:
                buf = _region_buffer(node.args[1])
                if buf is not None and _buffer_scope(buf) == "global":
                    _mark_write(writes, buf)

        post_order_visit(func.body, visit)
        return func.with_attr("hexagon.write_set", sorted(writes))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.hexagon.WriteSet")


def _collect_alloc_buffers(func: PrimFunc) -> list[Buffer]:
    out: list[Buffer] = []
    seen: set[str] = set()

    def add(buf: Buffer) -> None:
        key = _var_name(getattr(buf, "data", ""))
        if key not in seen:
            seen.add(key)
            out.append(buf)

    def visit(node: Any) -> None:
        if isinstance(node, SBlockRealize):
            for buf in node.block.alloc_buffers:
                add(buf)
        elif isinstance(node, SBlock):
            for buf in node.alloc_buffers:
                add(buf)

    post_order_visit(func.body, visit)
    return out


def HexagonStoragePlan():
    """Attach ``hexagon.vtcm_offsets`` and enforce R3 static VTCM budget."""

    def pass_fn(func: PrimFunc, mod, ctx):
        offsets: dict[str, IntImm] = {}
        details: list[str] = []
        off = 0
        for buf in _collect_alloc_buffers(func):
            scope = _buffer_scope(buf)
            if not _is_vtcm(scope):
                if not _is_acc(scope):
                    continue
            numel = _shape_numel(buf.shape)
            if numel is None:
                raise HexagonEmitError(f"R3: VTCM buffer {_buffer_name(buf)} shape 必须静态")
            # HMX acc itself is not VTCM, but acc_read materializes fp16 AH tiles
            # into VTCM before the final AH->RM writeback.  Plan that output
            # scratch here so the emitter can mechanically use StoragePlan
            # offsets instead of hard-coded GM_OUT maps.
            nbytes = _align(numel * (2 if _is_acc(scope) else _dtype_bytes(str(buf.dtype))))
            off = _align(off)
            offsets[_buffer_name(buf)] = IntImm("int32", off)
            details.append(f"{_buffer_name(buf)}:{nbytes}")
            off += nbytes
        if off > VTCM_BUDGET:
            raise HexagonEmitError(f"R3: VTCM 静态预算超限 {off} > 8MB; 明细: {', '.join(details)}")
        return func.with_attr("hexagon.vtcm_offsets", offsets)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.hexagon.StoragePlan")


_GDN_WSCRATCH_SLOTS: tuple[tuple[str, tuple[int, ...], str], ...] = (
    # Logical declaration order used by the public GDN TileLang example.  The C
    # ABI remains hrt_gdn_slot_t; this table only maps arbitrary user buffer
    # names to existing ABI field names and verifies shape/dtype compatibility.
    ("S", (128, 128), "float32"),
    ("kf", (32, 128), "float32"),
    ("qf", (32, 128), "float32"),
    ("vf", (32, 128), "float32"),
    ("w", (32, 128), "float32"),
    ("o", (32, 128), "float32"),
    ("A", (32, 32), "float32"),
    ("P", (32, 32), "float32"),
    ("kkprod", (128,), "float32"),
    ("qkprod", (128,), "float32"),
    ("eG", (32,), "float32"),
    ("eGinv", (32,), "float32"),
    ("beta", (32,), "float32"),
    ("eGC", (1,), "float32"),
)


def HexagonWScratchPlan():
    """Map per-worker DDR scratch declarations to hrt_gdn_slot_t ABI fields."""

    def pass_fn(func: PrimFunc, mod, ctx):
        bufs = [buf for buf in _collect_alloc_buffers(func) if _is_wscratch(_buffer_scope(buf))]
        if not bufs:
            return func
        if len(bufs) < len(_GDN_WSCRATCH_SLOTS):
            missing = ", ".join(
                f"{field}: shape={shape}, dtype={dtype}"
                for field, shape, dtype in _GDN_WSCRATCH_SLOTS[len(bufs):]
            )
            raise HexagonEmitError(
                f"wscratch: 声明数量 {len(bufs)} 少于 GDN slot 字段数量 {len(_GDN_WSCRATCH_SLOTS)}; 缺少 {missing}"
            )
        if len(bufs) > len(_GDN_WSCRATCH_SLOTS):
            raise HexagonEmitError(
                f"wscratch: 声明数量 {len(bufs)} 超过 GDN slot 字段数量 {len(_GDN_WSCRATCH_SLOTS)}"
            )
        pairs: list[str] = []
        for i, buf in enumerate(bufs):
            field, want_shape, want_dtype = _GDN_WSCRATCH_SLOTS[i]
            got_shape = tuple(_i64(d) for d in buf.shape)
            got_dtype = str(buf.dtype)
            if got_shape != want_shape or got_dtype != want_dtype:
                want = f"{field}: shape={want_shape}, dtype={want_dtype}"
                got = f"{_buffer_name(buf)}: shape={got_shape}, dtype={got_dtype}"
                raise HexagonEmitError(f"wscratch: 第 {i} 个声明无法映射到 GDN slot; 需要 {want}, 实际 {got}")
            pairs.append(f"{_buffer_name(buf)}:{field}")
        return func.with_attr("hexagon.wscratch_slots", tirx.StringImm(",".join(pairs)))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.hexagon.WScratchPlan")


def HexagonProfileConfig(enabled: bool = False):
    """Thread pass-config profiling opt-in into the per-kernel attribute."""

    def pass_fn(func: PrimFunc, mod, ctx):
        del mod, ctx
        if not enabled:
            return func
        attrs = func.attrs or {}
        if _bool_arg(attrs.get("tl.hexagon_prof", False)):
            return func
        return func.with_attr("tl.hexagon_prof", True)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.hexagon.ProfileConfig")


class _Verifier:
    def __init__(self) -> None:
        self.in_vector = 0
        self.in_pool = 0
        self.has_gdn_leaf = False
        self.buffers: dict[str, Buffer] = {}
        self.annotated_layouts: dict[str, Any] = {}

    def verify(self, func: PrimFunc) -> None:
        for _, buf in func.buffer_map.items():
            self.buffers[_buffer_name(buf)] = buf

        # Layout annotations created by ``alloc_shared(layout=...)`` are
        # attached to the SBlock that is current at the call site.  In nested
        # T.serial/T.parallel structures that can be an inner SBlock, while the
        # corresponding allocation still appears in an outer block's
        # ``alloc_buffers``.  Gather all alloc buffers and all layout maps before
        # checking allocations so suffix-vs-annotation validation is not
        # sensitive to SBlock nesting order.
        for buf in _collect_alloc_buffers(func):
            self.buffers[_buffer_name(buf)] = buf

        def collect_layouts(node: Any) -> None:
            if isinstance(node, SBlock):
                self._collect_layout_annotations(node)

        post_order_visit(func.body, collect_layouts)

        def scan(node: Any) -> None:
            if isinstance(node, Call) and _is_extern(node):
                callee = _extern_callee(node)
                if callee and callee.startswith("hexagon.") and callee.split(".", 1)[1] in {
                    "load_state128", "store_state128", "load_h2f_rows128", "scan_exp32", "dot128x2_store",
                    "state_x2_matvec128", "affine_rows128", "forward_solve32", "output_rows128",
                    "state_decay_rows128", "state_update32",
                }:
                    self.has_gdn_leaf = True

        post_order_visit(func.body, scan)
        self._visit_stmt(func.body)

    def _visit_stmt(self, op: Any) -> None:
        if isinstance(op, SBlockRealize):
            self._visit_stmt(op.block)
            return
        if isinstance(op, SBlock):
            self._collect_layout_annotations(op)
            for buf in op.alloc_buffers:
                self.buffers[_buffer_name(buf)] = buf
                self._check_alloc(buf)
            self._visit_stmt(op.body)
            return
        if isinstance(op, SeqStmt):
            for s in op.seq:
                self._visit_stmt(s)
            return
        if isinstance(op, For):
            is_vec = _is_vector_for(op)
            extent = _i64(op.extent)
            if is_vec and (extent is None or extent % 32):
                raise HexagonEmitError(f"R2: T.vectorized extent 必须是 32 个 fp32/64 个 fp16 lane 的倍数, 实际 {extent}{_loc(op)}")
            ttag = str(getattr(getattr(op, "thread_binding", None), "thread_tag", ""))
            var = _var_name(op.loop_var)
            if "threadIdx.x" in ttag or var == "tx":
                if extent is None:
                    raise HexagonEmitError(f"R10: threadIdx.x extent 必须是静态整数{_loc(op)}")
                if extent > 6:
                    raise HexagonEmitError(f"R10: Hexagon worker 数必须 ≤6, 实际 {extent}{_loc(op)}")
            is_pool = ("blockIdx.x" in ttag or var == "bx") and self._contains_gdn_leaf(op.body)
            self.in_vector += 1 if is_vec else 0
            self.in_pool += 1 if is_pool else 0
            self._visit_stmt(op.body)
            self.in_pool -= 1 if is_pool else 0
            self.in_vector -= 1 if is_vec else 0
            return
        if isinstance(op, AttrStmt):
            if str(op.attr_key) == "thread_extent":
                extent = _i64(op.value)
                node = op.node
                name = str(getattr(node, "thread_tag", "")) or str(node)
                vname = _var_name(getattr(node, "var", node))
                if "threadIdx.x" in name or vname == "tx":
                    if extent is None:
                        raise HexagonEmitError("R10: threadIdx.x extent 必须是静态整数")
                    if extent > 6:
                        raise HexagonEmitError(f"R10: Hexagon worker 数必须 ≤6, 实际 {extent}")
                is_pool = ("blockIdx.x" in name or vname == "bx") and self._contains_gdn_leaf(op.body)
            else:
                is_pool = False
            self.in_pool += 1 if is_pool else 0
            self._visit_stmt(op.body)
            self.in_pool -= 1 if is_pool else 0
            return
        if isinstance(op, IfThenElse):
            self._check_expr(op.condition)
            self._visit_stmt(op.then_case)
            if op.else_case is not None:
                self._visit_stmt(op.else_case)
            return
        if isinstance(op, BufferStore):
            if _is_vtcm(_buffer_scope(op.buffer)) and not self.in_vector and not self.has_gdn_leaf:
                raise HexagonEmitError(f"R2: 禁止对 VTCM buffer {_buffer_name(op.buffer)} 做标量 load/store{_loc(op)}")
            self._check_expr(op.value)
            for idx in op.indices:
                self._check_expr(idx)
            return
        if isinstance(op, Evaluate):
            self._check_expr(op.value)

    def _check_alloc(self, buf: Buffer) -> None:
        suffix = _scope_layout_suffix(_buffer_scope(buf))
        if suffix is not None:
            layout = self.annotated_layouts.get(_var_name(getattr(buf, "data", "")))
            if layout is None:
                layout = self.annotated_layouts.get(_buffer_name(buf))
            if layout is not None:
                try:
                    annotated = detect_layout_mode(layout, buf)
                except Exception as e:
                    raise HexagonEmitError(f"layout: 无法识别 buffer {_buffer_name(buf)} 的 annotation layout") from e
                # Strictly reject only explicit AH/WH-vs-suffix conflicts.
                # Missing annotations are allowed because the scope suffix is the
                # transition-period fallback channel.  Identity annotations
                # (``rm`` from fill_default_layout, historical ``none``, or any
                # other non-AH/WH identity/custom mode) are advisory and do not
                # contradict a .ah/.wh storage suffix; they can be introduced by
                # generic LayoutInference/default-fill around nested SBlocks.
                if annotated in ("ah", "wh") and annotated != suffix:
                    raise HexagonEmitError(
                        f"layout: buffer {_buffer_name(buf)} scope 后缀 .{suffix} 与 annotation {annotated} 不一致"
                    )
        if _is_acc(_buffer_scope(buf)):
            for dim in buf.shape:
                iv = _i64(dim)
                if iv is not None and iv % 32:
                    raise HexagonEmitError(f"R5: hmx.acc buffer {_buffer_name(buf)} 维度 {iv} 不是 32 的倍数")

    def _collect_layout_annotations(self, op: SBlock) -> None:
        layout_map = op.annotations.get("layout_map") if hasattr(op, "annotations") else None
        if not layout_map:
            return
        for var, layout in layout_map.items():
            self.annotated_layouts[_var_name(var)] = layout
            buf = getattr(var, "buffer", None) or self._buffer_from_data_arg(var)
            if buf is not None:
                self.annotated_layouts[_buffer_name(buf)] = layout

    def _check_expr(self, e: Any) -> None:
        if isinstance(e, BufferLoad):
            if _is_vtcm(_buffer_scope(e.buffer)) and not self.in_vector and not self.has_gdn_leaf:
                raise HexagonEmitError(f"R2: 禁止对 VTCM buffer {_buffer_name(e.buffer)} 做标量 load/store{_loc(e)}")
            for idx in e.indices:
                self._check_expr(idx)
            return
        if isinstance(e, Call):
            if _is_extern(e):
                callee = _extern_callee(e)
                off = _extern_arg_offset(e)
                if callee in ("hexagon.reduce_sum128", "hexagon.reduce_prod128"):
                    dst = e.args[off + 1] if callee == "hexagon.reduce_sum128" else e.args[off + 2]
                    dbuf = getattr(dst, "buffer", None)
                    if _is_vtcm(_buffer_scope(dbuf)):
                        raise HexagonEmitError(f"R2: 归约结果禁止标量写 VTCM buffer {_buffer_name(dbuf)}{_loc(e)}")
                if callee == "hexagon.reduce_prod2_128":
                    for dst in (e.args[off + 2], e.args[off + 5]):
                        dbuf = getattr(dst, "buffer", None)
                        if _is_vtcm(_buffer_scope(dbuf)):
                            raise HexagonEmitError(f"R2: 归约结果禁止标量写 VTCM buffer {_buffer_name(dbuf)}{_loc(e)}")
                if callee in ("hexagon.copy_rm_ah", "hexagon.copy_f32_ah", "hexagon.copy_f32_wh", "hexagon.copy_ah_rm", "hexagon.copy_acc_rm", "hexagon.copy_ddr"):
                    self._check_copy_intrin(e, off)
                if callee == "hexagon.gemm_hmx":
                    self._check_r4(e)
                    self._check_gemm_intrin(e, off)
                return
            else:
                opname = _call_op_name(e)
                if opname == "tl.tileop.copy":
                    self._check_copy(e)
                    return
                elif opname == "tl.tileop.gemm":
                    self._check_r4(e)
                    self._check_gemm(e)
                    return
            for a in e.args:
                self._check_expr(a)
            return
        cls = type(e).__name__
        if cls in ("Add", "Sub", "Mul", "Div", "FloorDiv", "Mod", "LT", "LE", "GT", "GE", "EQ", "NE", "Min", "Max"):
            self._check_expr(e.a); self._check_expr(e.b)
        elif cls in ("Cast", "Not", "Neg"):
            self._check_expr(e.value)

    def _check_r4(self, op: Any) -> None:
        if self.in_pool:
            raise HexagonEmitError(f"R4: HMX/T.gemm(hexagon.gemm_hmx) 不得出现在 pool worker 段内{_loc(op)}")

    def _contains_gdn_leaf(self, op: Any) -> bool:
        found = False

        def scan(node: Any) -> None:
            nonlocal found
            if found:
                return
            if isinstance(node, Call) and _is_extern(node):
                callee = _extern_callee(node)
                if callee and callee.startswith("hexagon.") and callee.split(".", 1)[1] in {
                    "load_state128", "store_state128", "load_h2f_rows128", "scan_exp32", "dot128x2_store",
                    "state_x2_matvec128", "affine_rows128", "forward_solve32", "output_rows128",
                    "state_decay_rows128", "state_update32",
                }:
                    found = True

        post_order_visit(op, scan)
        return found

    def _check_copy(self, call: Call) -> None:
        if len(call.args) < 2:
            return
        src = _region_buffer(call.args[0])
        dst = _region_buffer(call.args[1])
        if _is_vtcm(_buffer_scope(src)) or _is_vtcm(_buffer_scope(dst)):
            bytes_hint = 2
            for v in [x for x in _region_extents(call.args[1]) if x]:
                bytes_hint *= v
            if bytes_hint % 128:
                raise HexagonEmitError(f"R1: VTCM copy 大小必须可按 128B 向量化, 实际 {bytes_hint} bytes")

    def _check_gemm(self, call: Call) -> None:
        if len(call.args) < 10:
            raise HexagonEmitError("R5: tl.tileop.gemm 参数数量不足")
        trans_a = bool(_i64(call.args[3])) if _i64(call.args[3]) is not None else bool(call.args[3])
        trans_b = bool(_i64(call.args[4])) if _i64(call.args[4]) is not None else bool(call.args[4])
        m, n, k = (_i64(call.args[5]), _i64(call.args[6]), _i64(call.args[7]))
        if trans_a or not trans_b:
            raise HexagonEmitError("R5: v2 GEMM 只支持 NT 形式(A 非转置, B 已 WH/等价转置)")
        if None in (m, n, k):
            raise HexagonEmitError("R5: T.gemm 的 M/N/K 必须是静态整数")
        assert m is not None and n is not None and k is not None
        if m % 32 or n % 32 or k % 32:
            raise HexagonEmitError(f"R5: T.gemm tile 维度必须 32 整除, 实际 M={m},N={n},K={k}")

    def _check_copy_intrin(self, call: Call, off: int) -> None:
        if len(call.args) < off + 3:
            return
        src = self._buffer_from_data_arg(call.args[off])
        dst = self._buffer_from_data_arg(call.args[off + 1])
        if not (_is_vtcm(_buffer_scope(src)) or _is_vtcm(_buffer_scope(dst))):
            return
        src_nd = _i64(call.args[off + 2]) if len(call.args) > off + 2 else 0
        src_nd = int(src_nd or 0)
        dst_meta = off + 3 + 2 * src_nd
        dst_nd = _i64(call.args[dst_meta]) if len(call.args) > dst_meta else 0
        dst_nd = int(dst_nd or 0)
        bytes_hint = 2
        nd_for_bytes = dst_nd
        ext_base = dst_meta + 2
        # BufferLoad point-view copies into AH can arrive with an empty dst
        # region after LowerTileOp canonicalization.  For the same intrinsic the
        # source range still carries the logical full tile, so use it as the R1
        # byte-size proof instead of treating the copy as a scalar 1-element VTCM
        # access.  The emitter still validates the static trailing 2-D shape.
        if nd_for_bytes == 0:
            nd_for_bytes = src_nd
            ext_base = off + 4
        for i in range(nd_for_bytes):
            idx = ext_base + 2 * i
            if idx < len(call.args):
                v = _i64(call.args[idx])
                if v:
                    bytes_hint *= v
        if bytes_hint == 2 and dst is not None and _is_vtcm(_buffer_scope(dst)) and len(dst.shape) >= 2:
            bytes_hint = 2
            for dim in dst.shape[-2:]:
                v = _i64(dim)
                if v:
                    bytes_hint *= v
        if bytes_hint % 128:
            raise HexagonEmitError(f"R1: VTCM copy 大小必须可按 128B 向量化, 实际 {bytes_hint} bytes")

    def _check_gemm_intrin(self, call: Call, off: int) -> None:
        if len(call.args) < off + 6:
            raise HexagonEmitError("R5: hexagon.gemm_hmx 参数数量不足")
        # Trailing args: (M, N, K) legacy, or (M, N, K, transB) with the flag.
        nargs = len(call.args) - off
        if nargs >= 11:
            m, n, k = (_i64(call.args[off + nargs - 4]), _i64(call.args[off + nargs - 3]), _i64(call.args[off + nargs - 2]))
        else:
            m, n, k = (_i64(call.args[-3]), _i64(call.args[-2]), _i64(call.args[-1]))
        if None in (m, n, k):
            raise HexagonEmitError("R5: hexagon.gemm_hmx 的 M/N/K 必须是静态整数")
        assert m is not None and n is not None and k is not None
        if m % 32 or n % 32 or k % 32:
            raise HexagonEmitError(f"R5: hexagon.gemm_hmx tile 维度必须 32 整除, 实际 M={m},N={n},K={k}")

    def _buffer_from_data_arg(self, arg: Any) -> Buffer | None:
        if isinstance(arg, BufferLoad):
            return arg.buffer
        if isinstance(arg, Call) and _call_op_name(arg) == "tl.region":
            load = arg.args[0] if arg.args else None
            return getattr(load, "buffer", None)
        buf = getattr(arg, "buffer", None)
        if buf is not None:
            return buf
        name = _var_name(arg)
        for buf in self.buffers.values():
            if _var_name(getattr(buf, "data", "")) == name or _buffer_name(buf) == name:
                return buf
        return None


def HexagonVerify():
    """Verify Hexagon TIR rules that are visible before emission."""

    def pass_fn(func: PrimFunc, mod, ctx):
        _Verifier().verify(func)
        return func

    return prim_func_pass(pass_fn, opt_level=0, name="tl.hexagon.Verify")


# Public names used by the Hexagon pipeline.  Keep the Hexagon* aliases for
# compatibility with existing local scripts that imported the first draft.
ProductReduceFusion = HexagonProductReduceFusion
WriteSet = HexagonWriteSet
StoragePlan = HexagonStoragePlan
WScratchPlan = HexagonWScratchPlan


__all__ = [
    "HexagonProductReduceFusion",
    "HexagonWriteSet",
    "HexagonStoragePlan",
    "HexagonWScratchPlan",
    "HexagonVerify",
    "ProductReduceFusion",
    "WriteSet",
    "StoragePlan",
    "WScratchPlan",
]
