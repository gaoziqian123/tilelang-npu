"""Hexagon intrinsic C emitter for high-level TileLang TIR.

Emitter v2 lowers the pre-LowerTileOp TileLang TIR statement-by-statement into
DSP-side C.  The generated C still targets the proven GEMM_NT FastRPC entry, but
the loop/control-flow skeleton is derived from the user's TIR instead of being a
single whole-kernel text template.  Low-level HVX/HMX recipes are emitted as
calls into ``hexagon_rt.h``; generated kernels must not inline those helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import struct
from pathlib import Path
from typing import Any

from tvm import IRModule, arith, ir, tirx
from tvm.target import Target
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
    PrimExpr,
    PrimFunc,
    PyStmtExprVisitor,
    SBlock,
    SBlockRealize,
    SeqStmt,
)

from tilelang.hexagon.language.layout import make_layout


VTCM_BUDGET = 8 * 1024 * 1024
GM_TILE = 2048
GM_WA_MAX = 5632 * 1024


class HexagonEmitError(ValueError):
    """User-facing verifier/lowering error with Hexagon rule number."""


@dataclass
class _Alloc:
    name: str
    scope: str
    bytes: int
    offset: int | None = None


@dataclass
class _HVXVal:
    names: list[str]
    dtype: str


@dataclass
class _ScalarVar:
    name: str
    dtype: str


@dataclass
class _GlobalArg:
    buf: Buffer
    cname: str
    offset_name: str | None = None


@dataclass(frozen=True)
class _TileLayoutAddr:
    mode: str
    row: str
    col: str
    tile_index: str
    byte_offset: str


@dataclass
class _Ctx:
    block_var: str | None = None
    row_var: str | None = None
    vec_var: str | None = None
    vec_extent: int | None = None
    worker_var: str | None = None
    outer_vars: tuple[str, ...] = ()
    pipeline_async_first: bool = False


@dataclass
class _PoolWorker:
    name: str
    ctx_type: str
    extent: int
    outer_vars: tuple[str, ...]
    body: list[str]


def _i64(x: Any) -> int | None:
    if isinstance(x, IntImm):
        return int(x.value)
    if isinstance(x, int):
        return x
    try:
        if hasattr(x, "value"):
            return int(x.value)
    except Exception:  # pragma: no cover - defensive for non-IntImm PrimExpr
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


def _is_vtcm(scope: str) -> bool:
    return scope.startswith("vtcm") or scope.startswith("shared")


def _is_wscratch(scope: str) -> bool:
    return scope.startswith("wscratch")


def _is_acc(scope: str) -> bool:
    return scope == "hmx.acc" or "fragment" in scope


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


def _region_buffer(expr: Any) -> Buffer | None:
    """Extract the Buffer from a tl.region(...) Call or BufferRegion-like node."""

    if isinstance(expr, Call) and _call_op_name(expr) == "tl.region":
        load = expr.args[0]
        return getattr(load, "buffer", None)
    return getattr(expr, "buffer", None)


def _region_extents(expr: Any) -> list[int | None]:
    if isinstance(expr, Call) and _call_op_name(expr) == "tl.region":
        return [_i64(a) for a in expr.args[2:]]
    region = getattr(expr, "region", None)
    if region is not None:
        return [_i64(r.extent) for r in region]
    return []


def _region_load(expr: Any) -> BufferLoad | None:
    if isinstance(expr, Call) and _call_op_name(expr) == "tl.region":
        load = expr.args[0]
        return load if isinstance(load, BufferLoad) else None
    return None


def _bool_arg(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    iv = _i64(x)
    return bool(iv) if iv is not None else bool(x)


def _var_name(v: Any) -> str:
    return getattr(v, "name", None) or getattr(v, "name_hint", None) or str(v)


def _loc(op: Any) -> str:
    span = getattr(op, "span", None)
    return f" at {span}" if span else ""


def hexagon_fold_view_base_rc(mins: list[Any], buf: Buffer | None) -> tuple[Any, Any]:
    """Fold leading-dim point-view mins into trailing logical row/col.

    AH/WH layout maps are defined over the trailing logical 2-D plane, with any
    leading dimensions expanded as batches of full planes.  Both the copy
    stagers and ``hexagon.gemm_hmx`` lowering use this routine before the emitter
    evaluates the layout index map, so producer/consumer bases cannot diverge.
    """

    zero = IntImm("int32", 0)
    if len(mins) < 2:
        return zero, zero
    row, col = mins[-2], mins[-1]
    if buf is not None and len(mins) > 2 and len(buf.shape) >= len(mins):
        lead = None
        for i in range(len(mins) - 2):
            rows_per: Any = 1
            for d in buf.shape[i + 1:-1]:
                rows_per = rows_per * d
            term = mins[i] * rows_per
            lead = term if lead is None else lead + term
        if lead is not None:
            row = row + lead
    return row, col


@tirx.functor.visitor
class HexagonEmitter(PyStmtExprVisitor):
    """Statement-level TileLang Hexagon emitter (v2)."""

    def __init__(self, mod: IRModule | PrimFunc, target: Target | None = None):
        super().__init__()
        self.mod = mod
        self.target = target
        self.allocs: list[_Alloc] = []
        self.buffers: dict[str, Buffer] = {}
        self.has_copy = False
        self.has_zip16 = False
        self.has_unperm = False
        self.has_gemm = False
        self.gdn_mode: str | None = None
        self.has_vector_kernel = False
        self.vector_uses_pool = False
        # DDR 直读的 dcfetch 预取距离(元素数;fp16 下 4096 elem = 8192B,
        # 设计文档 §3:dcfetch 距离 2048 起有效,8192 平顶)
        self.dcfetch_elems = 4096
        self.nworkers: int | None = None
        self.seen_block_launch = False
        self.gdn_leaf_pool = False
        self.gemm_mnk: tuple[int, int, int] | None = None
        self.func_name: str | None = None
        self.body_lines: list[str] = []
        self.block_N_hint: int | None = None
        self.block_M_hint: int | None = None
        self.scalar_vars: dict[str, str] = {}
        self.has_scalar_kernel = False
        self.global_args: list[_GlobalArg] = []
        self.global_ptrs: dict[str, str] = {}
        self._written_globals: set[str] = set()
        self._vtcm_offsets: dict[str, int] = {}
        self._wscratch_slots: dict[str, str] = {}
        self._gemm_b_buffers: set[str] = set()
        self._pre_has_gemm = False
        self._pre_has_gemm_view = False
        self._pre_has_vector = False
        self._pre_has_parallel = False
        self._pool_workers: list[_PoolWorker] = []
        self._shape_params: list[str] = []
        self.hexagon_prof = False
        self._prof_next_slot = 5
        self._pending_async_pool_slot: int | None = None
        self._pipeline_async_first_available = False

    def emit(self, output_path: str | os.PathLike[str] | None = None) -> str:
        funcs = [self.mod] if isinstance(self.mod, PrimFunc) else [f for _, f in self.mod.functions.items()]
        for func in funcs:
            if isinstance(func, PrimFunc):
                self._record_params(func)
                self._pre_has_gemm = self._stmt_contains_gemm_intrin(func.body)
                self._pre_has_vector = self._contains_vector_for(func.body)
                self._pre_has_parallel = self._contains_parallel_for(func.body)
                self._shape_params = self._derive_shape_params(func)
                self.body_lines = self._lower_stmt(func.body, _Ctx(), indent=1)
        self._finalize_allocs()
        source = self._render_c()
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return source

    # ---- legacy visitor hooks used by negative/scalar checks ---------------

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        if _is_vtcm(_buffer_scope(op.buffer)):
            raise HexagonEmitError(f"R2: 禁止对 VTCM buffer {_buffer_name(op.buffer)} 做标量 load/store{_loc(op)}")

    def visit_buffer_store_(self, op: BufferStore) -> None:
        if _is_vtcm(_buffer_scope(op.buffer)):
            raise HexagonEmitError(f"R2: 禁止对 VTCM buffer {_buffer_name(op.buffer)} 做标量 load/store{_loc(op)}")
        self._check_expr(op.value)

    def visit_call_(self, op: Call) -> None:
        ann = getattr(op, "annotations", {}) or {}
        opname = _call_op_name(op)
        if opname == "tirx.call_pure_extern" and len(op.args) >= 1:
            callee = _ann_str(op.args[0])
            if callee in ("hexagon.gdn_prefill", "hexagon.silu_fp16", "hexagon.exp_fp16", "hexagon.exp_fp32", "hexagon.h2f", "hexagon.f2h", "hexagon.reduce_sum128", "hexagon.reduce_max32", "hexagon.reduce_max128", "hexagon.reduce_prod128", "hexagon.reduce_prod2_128") or (callee and callee.startswith("hexagon.") and callee.split(".", 1)[1] in self._gdn_leaf_names()):
                return
            if callee and callee.startswith("hexagon."):
                raise HexagonEmitError(f"白名单: 未支持的 Hexagon 原语 {callee}{_loc(op)}")
        hex_ann = _ann_str(ann.get("hexagon_intrin"))
        if hex_ann or opname.startswith("tl.hexagon") or opname.startswith("T.hexagon") or opname.startswith("tirx."):
            raise HexagonEmitError(f"白名单: 未支持的 Hexagon 原语 {opname}{_loc(op)}")
        raise HexagonEmitError(f"白名单: 不支持的 Call 表达式 {opname}{_loc(op)}")

    def _check_expr(self, expr: Any) -> None:
        """Strict expression verifier that preserves HexagonEmitError type."""

        if isinstance(expr, BufferLoad):
            self.visit_buffer_load_(expr)
            for idx in expr.indices:
                self._check_expr(idx)
            return
        if isinstance(expr, Call):
            opname = _call_op_name(expr)
            if opname in ("tir.exp", "tirx.exp", "exp") or str(opname).endswith(".exp"):
                if len(expr.args) != 1:
                    raise HexagonEmitError(f"白名单: exp 参数数量错误{_loc(expr)}")
                self._check_expr(expr.args[0])
                return
            self.visit_call_(expr)
            return
        if isinstance(expr, (IntImm,)) or isinstance(expr, (int, float, bool)):
            return
        if hasattr(expr, "name") or hasattr(expr, "name_hint"):
            return
        cls = type(expr).__name__
        if cls in ("Add", "Sub", "Mul", "Div", "FloorDiv", "Mod", "LT", "LE", "GT", "GE", "EQ", "NE", "Min", "Max"):
            self._check_expr(expr.a)
            self._check_expr(expr.b)
            return
        if cls in ("Cast", "Not", "Neg"):
            self._check_expr(expr.value)
            return
        raise HexagonEmitError(f"白名单: 不支持表达式 {cls}{_loc(expr)}")

    # ---- statement lowering -------------------------------------------------

    def _lower_stmt(self, op: Any, ctx: _Ctx, indent: int) -> list[str]:
        if isinstance(op, SBlockRealize):
            return self._lower_sblock(op.block, ctx, indent)
        if isinstance(op, SBlock):
            return self._lower_sblock(op, ctx, indent)
        if isinstance(op, SeqStmt):
            out: list[str] = []
            seq = list(op.seq)
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and self._is_gemm_intrin_eval(seq[i]) and self._is_acc_copy_intrin_eval(seq[i + 1]):
                    out += self._emit_gemm_intrin_with_copy(self._eval_call(seq[i]), self._eval_call(seq[i + 1]), ctx, indent)
                    i += 2
                    continue
                if i + 1 < len(seq) and isinstance(seq[i], IfThenElse) and isinstance(seq[i + 1], IfThenElse):
                    a, b = seq[i], seq[i + 1]
                    if (a.else_case is None and b.else_case is None
                            and self._expr_c(a.condition) == self._expr_c(b.condition)
                            and self._is_gemm_intrin_eval(a.then_case)
                            and self._is_acc_copy_intrin_eval(b.then_case)):
                        out.append(self._ind(indent, f"if ({self._expr_c(a.condition)}) {{"))
                        out += self._emit_gemm_intrin_with_copy(self._eval_call(a.then_case), self._eval_call(b.then_case), ctx, indent + 1)
                        out.append(self._ind(indent, "}"))
                        i += 2
                        continue
                if self._pending_async_pool_slot is not None and self._contains_pool_phase(seq[i]):
                    out += self._emit_profiled_pool_join(indent, self._pending_async_pool_slot)
                    self._pending_async_pool_slot = None
                out += self._lower_stmt(seq[i], ctx, indent)
                i += 1
            return out
        if isinstance(op, For):
            return self._lower_for(op, ctx, indent)
        if isinstance(op, IfThenElse):
            return self._lower_if(op, ctx, indent)
        if isinstance(op, AttrStmt):
            return self._lower_attr(op, ctx, indent)
        if isinstance(op, Evaluate):
            return self._lower_evaluate(op, ctx, indent)
        if type(op).__name__ == "LetStmt":
            name = _var_name(op.var)
            dtype = self._expr_dtype_hint(op.value) or str(getattr(op.var, "dtype", "")) or "float32"
            val = self._expr_scalar(op.value, dtype)
            self.scalar_vars[name] = dtype
            return [self._ind(indent, f"{self._scalar_ctype(dtype)} {name} = {val};")] + self._lower_stmt(op.body, ctx, indent)
        if type(op).__name__ == "Bind":
            name = _var_name(op.var)
            dtype = self._expr_dtype_hint(op.value) or str(getattr(op.var, "dtype", "")) or "float32"
            val = self._expr_scalar(op.value, dtype)
            self.scalar_vars[name] = dtype
            return [self._ind(indent, f"{self._scalar_ctype(dtype)} {name} = {val};")]
        if isinstance(op, BufferStore):
            if ctx.vec_var is not None:
                return self._emit_vector_store(op, ctx, indent)
            return self._emit_scalar_store(op, ctx, indent)
        cls = type(op).__name__
        if cls == "While":
            raise HexagonEmitError(f"白名单: 暂不支持 While 节点{_loc(op)}")
        raise HexagonEmitError(f"白名单: 不支持的 TIR 节点 {cls}{_loc(op)}")

    def _lower_sblock(self, op: SBlock, ctx: _Ctx, indent: int) -> list[str]:
        decls: list[str] = []
        for buf in op.alloc_buffers:
            self._record_alloc(buf)
            if _buffer_scope(buf) == "local.var":
                name = _buffer_name(buf)
                dtype = str(buf.dtype)
                self.scalar_vars[name] = dtype
                decls.append(self._ind(indent, f"{self._scalar_ctype(dtype)} {name};"))
        return decls + self._lower_stmt(op.body, ctx, indent)

    def _lower_for(self, op: For, ctx: _Ctx, indent: int) -> list[str]:
        anns = {str(k): str(v) for k, v in (getattr(op, "annotations", {}) or {}).items()}
        kind = str(getattr(op, "kind", ""))
        if self._is_unconsumed_pipeline_for(kind, anns):
            raise HexagonEmitError(f"Hexagon software pipeline annotations were not consumed before emit: For { _var_name(op.loop_var) } anns={sorted(anns)}{_loc(op)}")
        if self._is_injected_pipeline_for(kind, anns):
            return self._lower_injected_pipeline_for(op, ctx, indent)

        tb = getattr(op, "thread_binding", None)
        ttag = getattr(tb, "thread_tag", None) if tb is not None else None
        tname = str(ttag if ttag is not None else (getattr(tb, "name", "") if tb is not None else ""))
        var = _var_name(op.loop_var)
        extent = _i64(op.extent)
        is_vec = "vectorized" in kind.lower() or str(getattr(op, "kind", "")) == "2"
        is_parallel = "parallel" in kind.lower() or str(getattr(op, "kind", "")) == "1"
        if is_vec:
            if extent is None or extent % 32:
                raise HexagonEmitError(f"R2: T.vectorized extent 必须是 32 个 fp32/64 个 fp16 lane 的倍数, 实际 {extent}{_loc(op)}")
            body = self._lower_stmt(op.body, _Ctx(block_var=ctx.block_var, row_var=ctx.row_var, vec_var=var, vec_extent=extent, worker_var=ctx.worker_var), indent + 1)
            if not body:
                return []
            step = 32 if extent == 32 else 64
            return [self._ind(indent, f"// T.vectorized({extent}) -> 128B HVX vector loop"),
                    self._ind(indent, f"for (int {var} = 0; {var} < {extent}; {var} += {step}) {{")] + body + [self._ind(indent, "}")]
        if is_parallel:
            if extent is None:
                raise HexagonEmitError(f"白名单: T.Parallel extent 必须是静态整数: {var}{_loc(op)}")
            if self._pre_has_gemm:
                wid = len(self._pool_workers)
                wname = f"{self.func_name or 'tl'}_pool{wid}_worker"
                ctx_type = f"{self.func_name or 'tl'}_pool{wid}_ctx_t"
                worker_ctx = _Ctx(block_var=ctx.block_var, row_var=ctx.row_var, worker_var="job", outer_vars=ctx.outer_vars)
                body = [self._ind(1, f"const int {var} = job;")]
                body += self._lower_stmt(op.body, worker_ctx, 1)
                self._pool_workers.append(_PoolWorker(wname, ctx_type, extent, ctx.outer_vars, body))
                init_vals = ", ".join([arg.cname for arg in self.global_args] + self._shape_params + list(ctx.outer_vars) + ["abl", "prof"])
                async_pool = self._consume_pipeline_async_first(ctx)
                return [
                    self._ind(indent, f"// T.Parallel({extent}) -> Hexagon worker-pool phase ({'async start' if async_pool else 'sync join'})"),
                    self._ind(indent, f"{ctx_type} ctx{wid} = {{ {init_vals} }};"),
                    *(self._emit_profiled_pool_start(indent, f"attnops_pool_start_ctx({wname}, &ctx{wid}, {extent});")
                      if async_pool else
                      self._emit_profiled_pool_call(indent, f"attnops_pool_run_ctx({wname}, &ctx{wid}, {extent});")),
                ]
            # Hexagon elementwise kernels lower T.Parallel as an outer strip-mined
            # loop around inner T.vectorized stores.  The HVX work is still the
            # 128B vector body; T.Kernel(threads<=6) supplies the worker-pool
            # contract at the entry level.
            body = self._lower_stmt(op.body, _Ctx(block_var=ctx.block_var, row_var=ctx.row_var, worker_var=ctx.worker_var, outer_vars=ctx.outer_vars), indent + 1)
            return [self._ind(indent, f"// T.Parallel({extent}) strip over 128B HVX vectors"),
                    self._ind(indent, f"for (int {var} = 0; {var} < {extent}; {var}++) {{")] + body + [self._ind(indent, "}")]
        if "blockIdx.x" in tname or var == "bx":
            if extent is None:
                raise HexagonEmitError(f"白名单: blockIdx.x extent 必须静态{_loc(op)}")
            self.seen_block_launch = True
            is_vector_block = self._contains_vector_for(op.body)
            if is_vector_block and not self._contains_gdn_leaf(op.body) and not self._pre_has_gemm:
                self.has_vector_kernel = True
                self.vector_uses_pool = True
                wv = ctx.worker_var or "job"
                body = self._lower_stmt(op.body, _Ctx(block_var=var, row_var=ctx.row_var, worker_var=wv), indent + 1)
                return [self._ind(indent, f"// TIR launch_thread(blockIdx.x, extent={extent}) -> worker-pooled panels"),
                        self._ind(indent, f"const int {var} = {wv};")] + body
            body = self._lower_stmt(op.body, _Ctx(block_var=var, row_var=ctx.row_var, worker_var=ctx.worker_var), indent + 1)
            if self.gdn_mode == "leaf":
                self.gdn_leaf_pool = True
                wv = ctx.worker_var or "job"
                return [self._ind(indent, f"// TIR launch_thread(blockIdx.x, extent={extent}) -> worker-pooled heads"),
                        self._ind(indent, f"const int {var} = {wv};")] + body
            if self.gdn_mode == "escape":
                bound = "1"
            elif self._generic_gemm_recipe_mode():
                bound = str(extent)
            elif (self.has_vector_kernel and not self._pre_has_gemm) or not self._pre_has_gemm:
                bound = "(total + PANEL - 1) / PANEL"
            elif self._legacy_gemm_mode():
                bound = "N / NP"
            else:
                bound = "(N + NP - 1) / NP"
                body = [
                    self._ind(indent + 1, f"int n_rem_{var} = N - {var} * NP;"),
                    self._ind(indent + 1, f"int nct = (n_rem_{var} < NP ? n_rem_{var} : NP) / 32;"),
                    self._ind(indent + 1, "(void)nct;  // 部分配置下 tail 逻辑不引用,防 -Werror"),
                ] + body
            return [self._ind(indent, f"// TIR launch_thread(blockIdx.x, extent={extent})"),
                    self._ind(indent, f"for (int {var} = 0; {var} < {bound}; {var}++) {{")] + body + [self._ind(indent, "}")]
        if "threadIdx" in tname or var in ("tx", "ty", "tz"):
            if extent is None:
                raise HexagonEmitError(f"R10: threadIdx extent 必须是静态整数{_loc(op)}")
            if "threadIdx.x" in tname or var == "tx":
                if extent > 6:
                    raise HexagonEmitError(f"R10: Hexagon worker 数必须 ≤6, 实际 {extent}{_loc(op)}")
                self.nworkers = extent
            body = self._lower_stmt(op.body, ctx, indent)
            return [self._ind(indent, f"// TIR {tname} extent={extent} is lowered into hrt_* worker-pool recipes")] + body
            
        minv = _i64(op.min)
        if minv not in (0, None):
            raise HexagonEmitError(f"白名单: 暂只支持 min=0 的 For, {var} min={minv}{_loc(op)}")
        if extent is None:
            # Serial loops may have scalar-expression bounds, e.g. triangular
            # GDN loops `for j in T.serial(i + 1)`.  Vectorized/parallel/thread
            # loops were handled above and still require static extents.
            bound = self._expr_scalar(op.extent)
            outer = ctx.outer_vars + ((var,) if var != (ctx.row_var or "") else ())
            body = self._lower_stmt(op.body, _Ctx(block_var=ctx.block_var, row_var=ctx.row_var, worker_var=ctx.worker_var, outer_vars=outer), indent + 1)
            return [self._ind(indent, f"for (int {var} = 0; {var} < {bound}; {var}++) {{")] + body + [self._ind(indent, "}")]
        # For the GEMM row-block loop, keep runtime M/block_M rather than the
        # anchor shape's static trip count so one generated skel covers §5 shapes.
        is_row_loop = self._looks_like_row_loop(op)
        if is_row_loop and not self._legacy_gemm_mode():
            bound = "(M + GM_BLOCK_M - 1) / GM_BLOCK_M"
        else:
            bound = "M / 32" if is_row_loop else str(extent)
        row = var if is_row_loop else ctx.row_var
        outer = ctx.outer_vars + ((var,) if not is_row_loop else ())
        body = self._lower_stmt(op.body, _Ctx(block_var=ctx.block_var, row_var=row, worker_var=ctx.worker_var, outer_vars=outer), indent + 1)
        if is_row_loop and not self._legacy_gemm_mode():
            body = [
                self._ind(indent + 1, f"int m_rem_{var} = M - {var} * GM_BLOCK_M;"),
                self._ind(indent + 1, f"int mbt = (m_rem_{var} < GM_BLOCK_M ? m_rem_{var} : GM_BLOCK_M) / 32;"),
                self._ind(indent + 1, "(void)mbt;  // 同上,防 -Werror"),
            ] + body
        return [self._ind(indent, f"for (int {var} = 0; {var} < {bound}; {var}++) {{")] + body + [self._ind(indent, "}")]

    def _is_unconsumed_pipeline_for(self, kind: str, anns: dict[str, str]) -> bool:
        if "Pipelined" in kind:
            return True
        bad = {"num_stages", "software_pipeline_stage", "software_pipeline_order"}
        if any(k in bad for k in anns):
            return True
        for k, v in anns.items():
            kl = k.lower()
            vl = v.lower()
            if ("pipeline" in kl or "pipelined" in kl or "pipeline" in vl or "pipelined" in vl) and k != "tl_pipelined_num_stages":
                return True
        return False

    def _is_injected_pipeline_for(self, kind: str, anns: dict[str, str]) -> bool:
        return "tl_pipelined_num_stages" in anns and not self._is_unconsumed_pipeline_for(kind, anns)

    def _contains_pool_phase(self, op: Any) -> bool:
        return self._pre_has_gemm and self._contains_parallel_for(op)

    def _lower_injected_pipeline_for(self, op: For, ctx: _Ctx, indent: int) -> list[str]:
        var = _var_name(op.loop_var)
        minv = _i64(op.min)
        if minv not in (0, None):
            raise HexagonEmitError(f"白名单: 暂只支持 min=0 的 pipelined For, {var} min={minv}{_loc(op)}")
        extent = _i64(op.extent)
        bound = str(extent) if extent is not None else self._expr_scalar(op.extent)
        outer = ctx.outer_vars + ((var,) if var != (ctx.row_var or "") else ())
        loop_ctx = _Ctx(block_var=ctx.block_var, row_var=ctx.row_var, worker_var=ctx.worker_var, outer_vars=outer)
        old_pending = self._pending_async_pool_slot
        old_async = self._pipeline_async_first_available
        self._pending_async_pool_slot = None
        self._pipeline_async_first_available = True
        body = self._lower_pipeline_body(op.body, loop_ctx, indent + 1)
        if self._pending_async_pool_slot is not None:
            body += self._emit_profiled_pool_join(indent + 1, self._pending_async_pool_slot)
            self._pending_async_pool_slot = None
        self._pending_async_pool_slot = old_pending
        self._pipeline_async_first_available = old_async
        return [
            self._ind(indent, f"// T.Pipelined steady loop: first worker-pool phase is async; join before next pool phase or loop back-edge"),
            self._ind(indent, f"for (int {var} = 0; {var} < {bound}; {var}++) {{"),
        ] + body + [self._ind(indent, "}")]

    def _lower_pipeline_body(self, op: Any, ctx: _Ctx, indent: int) -> list[str]:
        seq = list(op.seq) if isinstance(op, SeqStmt) else [op]
        out: list[str] = []
        i = 0
        while i < len(seq):
            stmt = seq[i]
            if self._pending_async_pool_slot is not None and self._contains_pool_phase(stmt):
                out += self._emit_profiled_pool_join(indent, self._pending_async_pool_slot)
                self._pending_async_pool_slot = None
            if (i + 1 < len(seq)
                    and self._is_gemm_intrin_eval(stmt)
                    and self._is_acc_copy_intrin_eval(seq[i + 1])):
                out += self._emit_gemm_intrin_with_copy(self._eval_call(stmt), self._eval_call(seq[i + 1]), ctx, indent)
                i += 2
                continue
            out += self._lower_stmt(stmt, _Ctx(block_var=ctx.block_var, row_var=ctx.row_var, worker_var=ctx.worker_var,
                                              outer_vars=ctx.outer_vars), indent)
            i += 1
        return out

    def _consume_pipeline_async_first(self, ctx: _Ctx) -> bool:
        if ctx.pipeline_async_first or self._pipeline_async_first_available:
            self._pipeline_async_first_available = False
            return True
        return False

    def _lower_if(self, op: IfThenElse, ctx: _Ctx, indent: int) -> list[str]:
        cond = self._expr_c(op.condition)
        out = [self._ind(indent, f"if ({cond}) {{")]
        out += self._lower_stmt(op.then_case, ctx, indent + 1)
        if op.else_case is not None:
            out += [self._ind(indent, "} else {")]
            out += self._lower_stmt(op.else_case, ctx, indent + 1)
        out += [self._ind(indent, "}")]
        return out

    def _lower_attr(self, op: AttrStmt, ctx: _Ctx, indent: int) -> list[str]:
        key = str(op.attr_key)
        if key == "thread_extent":
            extent = _i64(op.value)
            node = op.node
            tag = str(getattr(node, "thread_tag", ""))
            vname = _var_name(getattr(node, "var", node))
            name = tag or str(node)
            if "blockIdx.x" in name or vname == "bx":
                if extent is None:
                    raise HexagonEmitError("白名单: blockIdx.x extent 必须静态")
                self.seen_block_launch = True
                is_vector_block = self._contains_vector_for(op.body)
                is_gdn_block = self._contains_gdn_leaf(op.body)
                if is_gdn_block:
                    self.gdn_mode = "leaf"
                if is_vector_block and not is_gdn_block and not self._pre_has_gemm:
                    self.has_vector_kernel = True
                    self.vector_uses_pool = True
                    wv = ctx.worker_var or "job"
                    body = self._lower_stmt(op.body, _Ctx(block_var=vname, row_var=ctx.row_var, worker_var=wv), indent + 1)
                    return [self._ind(indent, f"// TIR launch_thread(blockIdx.x, extent={extent}) -> worker-pooled panels"),
                            self._ind(indent, f"const int {vname} = {wv};")] + body
                body = self._lower_stmt(op.body, _Ctx(block_var=vname, row_var=ctx.row_var, worker_var=ctx.worker_var), indent + 1)
                if self.gdn_mode == "leaf":
                    self.gdn_leaf_pool = True
                    wv = ctx.worker_var or "job"
                    return [self._ind(indent, f"// TIR launch_thread(blockIdx.x, extent={extent}) -> worker-pooled heads"),
                            self._ind(indent, f"const int {vname} = {wv};")] + body
                if self.gdn_mode == "escape":
                    bound = "1"
                elif self._generic_gemm_recipe_mode():
                    bound = str(extent)
                elif (self.has_vector_kernel and not self._pre_has_gemm) or not self._pre_has_gemm:
                    bound = "(total + PANEL - 1) / PANEL"
                else:
                    bound = "N / NP"
                return [self._ind(indent, f"// TIR launch_thread(blockIdx.x, extent={extent})"),
                        self._ind(indent, f"for (int {vname} = 0; {vname} < {bound}; {vname}++) {{")] + body + [self._ind(indent, "}")]
            if "threadIdx.x" in name or vname == "tx":
                if extent is None:
                    raise HexagonEmitError("R10: threadIdx.x extent 必须是静态整数")
                if extent > 6:
                    raise HexagonEmitError(f"R10: Hexagon worker 数必须 ≤6, 实际 {extent}")
                self.nworkers = extent
            if "threadIdx" in name or vname in ("tx", "ty", "tz"):
                body = self._lower_stmt(op.body, ctx, indent)
                prefix = [self._ind(indent, f"// TIR {name} extent={extent} is lowered into hrt_* worker-pool recipes")]
                if any(re.search(r"\b%s\b" % re.escape(vname), line) for line in body):
                    prefix.append(self._ind(indent, f"const int {vname} = 0;"))
                return prefix + body
            return self._lower_stmt(op.body, ctx, indent)
        if key == "tl.assume":
            return self._lower_stmt(op.body, ctx, indent)
        raise HexagonEmitError(f"白名单: 不支持 AttrStmt attr_key={key}{_loc(op)}")

    def _lower_evaluate(self, op: Evaluate, ctx: _Ctx, indent: int) -> list[str]:
        if not isinstance(op.value, Call):
            raise HexagonEmitError(f"白名单: 不支持非 Call 的 Evaluate: {type(op.value).__name__}{_loc(op)}")
        call = op.value
        opname = _call_op_name(call)
        if call.op.same_as(ir.Op.get("tl.tileop.copy")) or opname == "tl.tileop.copy":
            raise HexagonEmitError(f"白名单: tl.tileop.copy 必须先经 LowerTileOp 改写为 hexagon.copy_*{_loc(op)}")
        if call.op.same_as(ir.Op.get("tl.tileop.reduce")) or opname == "tl.tileop.reduce":
            return self._emit_reduce(call, ctx, indent)
        if call.op.same_as(ir.Op.get("tl.tileop.gemm")) or opname == "tl.tileop.gemm":
            raise HexagonEmitError(f"白名单: tl.tileop.gemm 必须先经 LowerTileOp 改写为 hexagon.gemm_hmx{_loc(op)}")
        ann = getattr(call, "annotations", {}) or {}
        hex_ann = _ann_str(ann.get("hexagon_intrin"))
        if opname in ("tir.call_pure_extern", "tirx.call_pure_extern") and len(call.args) >= 1 and _ann_str(call.args[0]) == "hexagon.gdn_prefill":
            return self._emit_gdn_prefill(call, ctx, indent)
        if opname in ("tir.call_pure_extern", "tirx.call_pure_extern") and len(call.args) >= 1:
            callee = _ann_str(call.args[0])
            if callee == "hexagon.reduce_sum128":
                return self._emit_reduce_sum128(call, ctx, indent)
            if callee in ("hexagon.reduce_max32", "hexagon.reduce_max128"):
                return self._emit_reduce_max(call, ctx, indent)
            if callee == "hexagon.reduce_prod128":
                return self._emit_reduce_prod128(call, ctx, indent)
            if callee == "hexagon.reduce_prod2_128":
                return self._emit_reduce_prod2_128(call, ctx, indent)
            if callee == "hexagon.gemm_hmx":
                raise HexagonEmitError(f"白名单: hexagon.gemm_hmx 必须后接 hexagon.copy_acc_rm 以执行延迟 acc_read{_loc(op)}")
            if callee in ("hexagon.copy_rm_ah", "hexagon.copy_f32_ah", "hexagon.copy_f32_wh", "hexagon.copy_ah_rm", "hexagon.copy_acc_rm", "hexagon.copy_ddr"):
                return self._emit_copy_intrin(call, ctx, indent)
            if callee and callee.startswith("hexagon.") and callee.split(".", 1)[1] in self._gdn_leaf_names():
                return self._emit_gdn_leaf(callee.split(".", 1)[1], call, ctx, indent)
        if hex_ann or opname.startswith("tl.hexagon") or opname.startswith("T.hexagon") or opname.startswith("tir."):
            raise HexagonEmitError(f"白名单: 未支持的 Hexagon 原语 {opname}{_loc(op)}")
        raise HexagonEmitError(f"白名单: 不支持的 Evaluate Call {opname}{_loc(op)}")

    # ---- recipe emitters ----------------------------------------------------

    def _emit_copy_intrin(self, call: Call, ctx: _Ctx, indent: int) -> list[str]:
        callee = _ann_str(call.args[0]) if call.args else None
        if callee == "hexagon.copy_rm_ah":
            dst = self._buffer_from_data_arg(call.args[2]) if len(call.args) > 2 else None
            if self._generic_gemm_recipe_mode():
                self.has_copy = True
                src = self._buffer_from_data_arg(call.args[1]) if len(call.args) > 1 else None
                dst = self._buffer_from_data_arg(call.args[2]) if len(call.args) > 2 else None
                if src is None or dst is None:
                    raise HexagonEmitError(f"copy_rm_ah: 找不到源/目标 buffer{_loc(call)}")
                src_ptr = self._buffer_ptr_c(src)
                dst_off = self._vtcm_offset_c(dst)
                dst_scope = _buffer_scope(dst)
                snd = int(_i64(call.args[3]) or 0) if len(call.args) > 3 else 0
                src_base = self._copy_region_linear_base(call, 4, snd, src)
                dst_meta = 4 + 2 * snd
                dnd = int(_i64(call.args[dst_meta]) or 0) if len(call.args) > dst_meta else 0
                dst_exts = [_i64(call.args[dst_meta + 2 + 2 * i]) if len(call.args) > dst_meta + 2 + 2 * i else None for i in range(dnd)]
                if len(dst_exts) < 2 or dst_exts[-2] is None or dst_exts[-1] is None:
                    raise HexagonEmitError(f"copy_rm_ah: 目标 AH/WH region 必须是静态二维尾维{_loc(call)}")
                rows, cols = int(dst_exts[-2]), int(dst_exts[-1])
                if rows % 32 or cols % 32:
                    raise HexagonEmitError(f"R5: copy_rm_ah 目标 tile 维度必须 32 整除, 实际 rows={rows},cols={cols}")
                dst_mins = [call.args[dst_meta + 1 + 2 * i] for i in range(dnd) if len(call.args) > dst_meta + 1 + 2 * i]
                drow, dcol = self._flatten_view_rc(dst_mins, dst)
                if "wh" in dst_scope:
                    # WH staged sub-panels are addressed within the copied
                    # logical [rows, cols] region.  The generic tile-layout
                    # fast path is tuned for HMX operand addressing and can use
                    # the parent WH map's compact column count; for sliced
                    # staging we need the row-slice offset in the destination
                    # panel itself (row_tile * kt + col_tile).
                    kt_c = str(cols // 32)
                    drow_c = self._expr_c(drow)
                    dcol_c = self._expr_c(dcol)
                    dst_view = f"(((size_t)({drow_c}) / 32) * {kt_c} + ((size_t)({dcol_c}) / 32)) * HRT_TILE_BYTES"
                else:
                    dst_view = self._tile_layout_addr(dst, drow, dcol).byte_offset
                if dst_view != "0":
                    dst_off = f"{dst_off} + {dst_view}"
                kt = cols // 32
                mt = rows // 32
                if "wh" in dst_scope:
                    src_ld_i = _i64(src.shape[-1]) if src is not None and len(src.shape) >= 1 else None
                    src_stride_c = str(int(src_ld_i)) if src_ld_i is not None else str(cols)
                    # hrt_stage_f16_rm_to_wh_nt derives the destination
                    # row-block stride from the copied region's column count,
                    # which is only correct when the region spans the buffer's
                    # full width (or has a single row block).  A sliced
                    # multi-row-block region (e.g. a [128,256] WH operand
                    # staged in 64-column slices) needs the buffer's own
                    # col-tile count as the dst row-block stride.
                    # WH buffers may carry the packed physical shape
                    # [row_blocks, col_tiles, 16, 64] after layout rewrite;
                    # the logical col-tile count is then shape[1].
                    if len(dst.shape) >= 4 and _i64(dst.shape[1]) is not None:
                        buf_kt = int(_i64(dst.shape[1]))
                    else:
                        buf_cols_i = _i64(dst.shape[-1]) if len(dst.shape) >= 1 else None
                        buf_kt = int(buf_cols_i) // 32 if buf_cols_i is not None else kt
                    if mt > 1 and buf_kt != kt:
                        body = [self._ind(indent, "if (!(abl & 2)) {"),
                                self._ind(indent + 1, "hrt_mask_init();"),
                                self._ind(indent + 1, f"hrt_stage_f16_rm_to_wh_nt_s({src_ptr} + (size_t)({src_base}), V + {dst_off}, {cols}, {rows}, {src_stride_c}, {buf_kt});"),
                                self._ind(indent, "}")]
                    else:
                        body = [self._ind(indent, "if (!(abl & 2)) {"),
                                self._ind(indent + 1, "hrt_mask_init();"),
                                self._ind(indent + 1, f"hrt_stage_f16_rm_to_wh_nt({src_ptr} + (size_t)({src_base}), V + {dst_off}, {cols}, {rows}, {src_stride_c});"),
                                self._ind(indent, "}")]
                else:
                    src_ld_i = _i64(src.shape[-1]) if src is not None and len(src.shape) >= 1 else None
                    if src_ld_i is not None and int(src_ld_i) != cols:
                        body = [self._ind(indent, "if (!(abl & 2)) {"),
                                self._ind(indent + 1, "hrt_mask_init();"),
                                self._ind(indent + 1, f"for (int tl_gm_r = 0; tl_gm_r < {mt}; tl_gm_r++)"),
                                self._ind(indent + 2, f"hrt_stage_act_hvx_direct_strided((const uint8_t *)({src_ptr} + (size_t)({src_base}) + (size_t)tl_gm_r * 32 * {src_ld_i}),"),
                                self._ind(indent + 3, f"V + {dst_off} + (size_t)tl_gm_r * {kt} * HRT_TILE_BYTES, {src_ld_i}, {cols}, {kt}, HRT_TILE_BYTES, 32);"),
                                self._ind(indent, "}")]
                    else:
                        body = [self._ind(indent, "if (!(abl & 2)) {"),
                                self._ind(indent + 1, "hrt_mask_init();"),
                                self._ind(indent + 1, f"for (int tl_gm_r = 0; tl_gm_r < {mt}; tl_gm_r++)"),
                                self._ind(indent + 2, f"hrt_stage_act_hvx_direct((const uint8_t *)({src_ptr} + (size_t)({src_base}) + (size_t)tl_gm_r * 32 * {cols}),"),
                                self._ind(indent + 3, f"V + {dst_off} + (size_t)tl_gm_r * {kt} * HRT_TILE_BYTES, {cols}, {kt}, HRT_TILE_BYTES, 32);"),
                                self._ind(indent, "}")]
                return body if ctx.worker_var is not None else self._prof_wrap(indent, 1, body)
            if self._is_gemm_b_operand(dst):
                self.has_copy = True
                w_base = "(size_t)bx * nct * kt" if self._legacy_gemm_mode() else "(size_t)bx * (NP / 32) * kt"
                return [self._ind(indent, "t0 = HAP_perf_get_qtimer_count();"),
                        self._ind(indent, "if (!(abl & 1))"),
                        self._ind(indent + 1, f"hrt_copy_pooled(V + GM_WA, (const uint8_t *)w + {w_base} * HRT_TILE_BYTES,"),
                        self._ind(indent + 2, "(size_t)nct * kt * HRT_TILE_BYTES);"),
                        self._ind(indent, "prof[0] += (int32_t)(HAP_perf_get_qtimer_count() - t0);")]
            self.has_copy = True
            rv = ctx.row_var or "0"
            src = self._buffer_from_data_arg(call.args[1]) if len(call.args) > 1 else None
            src_ptr = self._buffer_ptr_c(src)
            if self._legacy_gemm_mode():
                return [self._ind(indent, "t0 = HAP_perf_get_qtimer_count();"),
                        self._ind(indent, "if (!(abl & 2)) {"),
                        self._ind(indent + 1, f"hrt_copy_128_dcfetch(V + GM_LIN, (const uint8_t *)({src_ptr} + (size_t){rv} * 32 * K), (size_t)32 * K * 2);"),
                        self._ind(indent + 1, "hrt_mask_init();"),
                        self._ind(indent + 1, "hrt_stage_act_hvx(V + GM_LIN, V + GM_ACT, K, kt, HRT_TILE_BYTES);"),
                        self._ind(indent, "}"),
                        self._ind(indent, "prof[1] += (int32_t)(HAP_perf_get_qtimer_count() - t0);")]
            bm = self.block_M_hint or 32
            if bm == 32:
                stage = [self._ind(indent + 1, f"hrt_stage_act_hvx_direct((const uint8_t *)({src_ptr} + (size_t){rv} * GM_BLOCK_M * K), V + GM_ACT, K, kt, HRT_TILE_BYTES, 32);")]
            else:
                mbt_expr = "mbt" if ctx.row_var is not None else str(bm // 32)
                stage = [self._ind(indent + 1, f"for (int rb = 0; rb < {mbt_expr}; rb++)"),
                         self._ind(indent + 2, f"hrt_stage_act_hvx_direct((const uint8_t *)({src_ptr} + ((size_t){rv} * GM_BLOCK_M + rb * 32) * K),"),
                         self._ind(indent + 3, "V + GM_ACT + (size_t)rb * kt * HRT_TILE_BYTES, K, kt, HRT_TILE_BYTES, 32);")]
            body = [self._ind(indent, "if (!(abl & 2)) {"),
                    self._ind(indent + 1, "hrt_mask_init();"),
                    *stage,
                    self._ind(indent, "}")]
            return (body if ctx.worker_var is not None else self._prof_wrap(indent, 1, body)) if self._generic_gemm_recipe_mode() else [
                    self._ind(indent, "t0 = HAP_perf_get_qtimer_count();"),
                    *body,
                    self._ind(indent, "prof[1] += (int32_t)(HAP_perf_get_qtimer_count() - t0);")]
        if callee in ("hexagon.copy_f32_ah", "hexagon.copy_f32_wh"):
            self.has_copy = True
            self.has_scalar_kernel = True
            src = self._buffer_from_data_arg(call.args[1]) if len(call.args) > 1 else None
            dst = self._buffer_from_data_arg(call.args[2]) if len(call.args) > 2 else None
            if dst is None:
                raise HexagonEmitError(f"copy_f32_ah: 找不到目标 VTCM buffer{_loc(call)}")
            src_ptr = self._buffer_ptr_c(src)
            snd = int(_i64(call.args[3]) or 0) if len(call.args) > 3 else 0
            src_mins = [self._expr_c(call.args[4 + 2 * i]) for i in range(snd) if len(call.args) > 4 + 2 * i]
            src_exts = [_i64(call.args[5 + 2 * i]) if len(call.args) > 5 + 2 * i else None for i in range(snd)]
            dst_meta = 4 + 2 * snd
            dnd = int(_i64(call.args[dst_meta]) or 0) if len(call.args) > dst_meta else 0
            dst_exts = [_i64(call.args[dst_meta + 2 + 2 * i]) if len(call.args) > dst_meta + 2 + 2 * i else None for i in range(dnd)]
            if len(dst_exts) < 2 or dst_exts[-2] is None or dst_exts[-1] is None:
                raise HexagonEmitError(f"copy_f32_ah: 目标 AH region 必须是静态二维尾维{_loc(call)}")
            rows, cols = int(dst_exts[-2]), int(dst_exts[-1])
            if len(src_exts) < 2 or src_exts[-1] is None:
                raise HexagonEmitError(f"copy_f32_ah: 源 fp32 region 必须是静态二维尾维{_loc(call)}")
            # 点视图的 region extent 可能被对侧广播成逻辑 tile 形状(如
            # w^T staging 的 (128,32)),而行距永远是背板 buffer 的行长。
            src_ld_i = _i64(src.shape[-1]) if src is not None and len(src.shape) >= 1 else None
            ld = int(src_ld_i) if src_ld_i is not None else int(src_exts[-1])
            trans_arg_idx = dst_meta + 1 + 2 * dnd
            if len(call.args) > trans_arg_idx:
                trans = self._expr_c(call.args[trans_arg_idx])
            else:
                trans_i = 1 if rows != cols and src_exts[-2] == cols and src_exts[-1] == rows else 0
                trans = str(trans_i)
            src_base = self._copy_region_linear_base(call, 4, snd, src)
            bname = _buffer_name(dst)
            alloc = next((a for a in self.allocs if a.name == bname), None)
            if alloc is None or alloc.offset is None:
                raise HexagonEmitError(f"R3: 找不到 VTCM buffer {bname} 的静态偏移")
            dst_mins = [call.args[dst_meta + 1 + 2 * i] for i in range(dnd) if len(call.args) > dst_meta + 1 + 2 * i]
            dst_view = self._hmx_tile_base_offset_from_region_c(dst_mins, dst, rows, cols)
            dst_c = f"V + {alloc.offset}" if dst_view == "0" else f"V + {alloc.offset} + {dst_view}"
            if callee == "hexagon.copy_f32_wh":
                # WH: dst 逻辑形状 [N, K](gemm B 操作数约定);trans=1 表示
                # 源按 [N,K] rm 读、产出 W=src^T 的 WH tiles(默认 trans=0:
                # 源按 [K,N] rm 读)。不做方形自动推断,需要转置必须显式标注。
                if len(call.args) > trans_arg_idx:
                    pass
                else:
                    trans = "0"
                helper = "hrt_tlgdn_stage_f32_to_wh"
                dim_args = f"{cols}, {rows}"
            else:
                helper = "hrt_tlgdn_stage_f32_to_ah"
                dim_args = f"{rows}, {cols}"
            body = [
                self._ind(indent, "if (!(abl & 2)) {"),
                self._ind(indent + 1, f"{helper}(({src_ptr} + {src_base}), {dst_c}, {dim_args}, {ld}, {trans});"),
                self._ind(indent, "}"),
            ]
            return body if ctx.worker_var is not None else self._prof_wrap(indent, 1, body)
        if callee == "hexagon.copy_ddr":
            self.has_copy = True
            src = self._buffer_from_data_arg(call.args[1]) if len(call.args) > 1 else None
            dst = self._buffer_from_data_arg(call.args[2]) if len(call.args) > 2 else None
            ss, ds = _buffer_scope(src), _buffer_scope(dst)
            src_ptr = self._buffer_ptr_c(src)
            dst_ptr = self._buffer_ptr_c(dst)
            # C++ Hexagon copy lowering serializes: callee, src_data, dst_data,
            # src_ndim, (src_min, src_extent)*, dst_ndim, (dst_min, dst_extent)*.
            snd = _i64(call.args[3]) if len(call.args) > 3 else 0
            snd = int(snd or 0)
            src_base = self._expr_c(call.args[4]) if len(call.args) > 4 and snd else "0"
            dst_meta = 4 + 2 * snd
            dnd = _i64(call.args[dst_meta]) if len(call.args) > dst_meta else 0
            dnd = int(dnd or 0)
            dst_base = self._expr_c(call.args[dst_meta + 1]) if len(call.args) > dst_meta + 1 and dnd else "0"
            elems = self._expr_c(call.args[dst_meta + 2]) if len(call.args) > dst_meta + 2 and dnd else "1"
            global_base = src_base if ss == "global" else dst_base
            lines = [
                self._ind(indent, "{"),
                self._ind(indent + 1, f"size_t copy_base = (size_t)({global_base});"),
                self._ind(indent + 1, f"size_t copy_elems = (size_t){elems};"),
                self._ind(indent + 1, "if (copy_base >= total) copy_elems = 0;"),
                self._ind(indent + 1, "else if (copy_base + copy_elems > total) copy_elems = total - copy_base;"),
                self._ind(indent + 1, "size_t copy_bytes = copy_elems * sizeof(f16);"),
                self._ind(indent + 1, "if (copy_bytes) {"),
            ]
            if ss == "global" and _is_vtcm(ds):
                lines.append(self._ind(indent + 2, f"hrt_copy_128_dcfetch((uint8_t *)({dst_ptr} + (size_t)({dst_base})), (const uint8_t *)({src_ptr} + (size_t)({src_base})), copy_bytes);"))
            elif _is_vtcm(ss) and ds == "global":
                lines += [
                    self._ind(indent + 2, "for (size_t copy_i = 0; copy_i < copy_bytes; copy_i += 128) {"),
                    self._ind(indent + 3, f"*(HVX_Vector *)((uint8_t *)({dst_ptr} + (size_t)({dst_base})) + copy_i) = *(const HVX_Vector *)((const uint8_t *)({src_ptr} + (size_t)({src_base})) + copy_i);"),
                    self._ind(indent + 2, "}"),
                ]
            else:
                lines.append(self._ind(indent + 2, f"memcpy((uint8_t *)({dst_ptr} + (size_t)({dst_base})), (const uint8_t *)({src_ptr} + (size_t)({src_base})), copy_bytes);"))
            lines += [self._ind(indent + 1, "}"), self._ind(indent, "}")]
            return lines
        if callee in ("hexagon.copy_ah_rm", "hexagon.copy_acc_rm"):
            raise HexagonEmitError("白名单: acc/ah -> rm copy 必须由 hexagon.gemm_hmx + hexagon.copy_acc_rm 融合处理")
        raise HexagonEmitError(f"白名单: 未支持的 Hexagon copy intrinsic {callee}{_loc(call)}")

    def _emit_gdn_prefill(self, call: Call, ctx: _Ctx, indent: int) -> list[str]:
        self.gdn_mode = "escape"
        if len(call.args) < 12:
            raise HexagonEmitError("GDN: hexagon.gdn_prefill 参数数量不足")
        T = _i64(call.args[9])
        hk = _i64(call.args[10])
        hv = _i64(call.args[11])
        if (T, hk, hv) != (1024, 16, 32):
            raise HexagonEmitError(f"GDN: 当前只验证 Qwen3.5 anchor T=1024,Hk=16,Hv=32, 实际 T={T},Hk={hk},Hv={hv}")
        return [
            self._ind(indent, "uint64_t tt = HAP_perf_get_qtimer_count();"),
            self._ind(indent, f"int e = hrt_gdn_prefill_hvx(slab, slabLen, {T}, {hk}, {hv});"),
            self._ind(indent, "if (e) return e;"),
            self._ind(indent, "prof[0] = (int32_t)(HAP_perf_get_qtimer_count() - tt);")
        ]

    @staticmethod
    def _gdn_leaf_names() -> set[str]:
        return {
            "load_state128", "store_state128", "load_h2f_rows128", "scan_exp32", "dot128x2_store",
            "state_x2_matvec128", "affine_rows128", "forward_solve32", "output_rows128",
            "state_decay_rows128", "state_update32",
        }

    def _gdn_ptr_arg(self, arg: Any) -> str:
        name = _var_name(arg)
        buf = self.buffers.get(name) or self._buffer_from_data_arg(arg)
        if buf is not None and _is_wscratch(_buffer_scope(buf)):
            return self._buffer_ptr_c(buf)
        if buf is not None and _is_vtcm(_buffer_scope(buf)):
            raise HexagonEmitError(f"GDN: scratch buffer {name} 必须用 T.alloc_wscratch, 不能用 VTCM alloc_shared")
        if buf is not None and _buffer_scope(buf) == "global":
            return self._buffer_ptr_c(buf)
        return name

    def _wscratch_field(self, buf: Buffer | None) -> str:
        bname = _buffer_name(buf)
        field = self._wscratch_slots.get(bname)
        if field is None:
            raise HexagonEmitError(f"wscratch: buffer {bname} 未映射到 runtime slot; 请用 T.alloc_wscratch 且检查声明顺序/shape")
        return field

    def _buffer_elem_ctype(self, buf: Buffer | None) -> str:
        dtype = str(getattr(buf, "dtype", "float16"))
        return "f32" if dtype == "float32" else "f16"

    def _vtcm_offset_c(self, buf: Buffer | None) -> str:
        bname = _buffer_name(buf)
        alloc = next((a for a in self.allocs if a.name == bname), None)
        if alloc is None or alloc.offset is None:
            raise HexagonEmitError(f"R3: 找不到 VTCM buffer {bname} 的静态偏移")
        return str(alloc.offset)

    def _copy_region_linear_base(self, call: Call, start: int, ndim: int, buf: Buffer | None) -> str:
        if ndim <= 0:
            return "0"
        terms: list[str] = []
        for i in range(ndim):
            mi = start + 2 * i
            if len(call.args) <= mi:
                break
            min_c = self._expr_c(call.args[mi])
            suffix = buf.shape[i + 1:] if buf is not None and len(buf.shape) > i + 1 else ()
            stride = self._shape_numel_c(suffix) if suffix else "1"
            terms.append(f"(size_t)({min_c})" if stride == "1" else f"(size_t)({min_c}) * ({stride})")
        return " + ".join(terms) if terms else "0"

    def _gdn_arg(self, call: Call, idx: int) -> str:
        off = 1 if len(call.args) and (_ann_str(call.args[0]) or "").startswith("hexagon.") else 0
        if len(call.args) <= idx + off:
            raise HexagonEmitError("GDN leaf: 参数数量不足")
        return self._expr_c(call.args[idx + off])

    def _emit_gdn_leaf(self, name: str, call: Call, ctx: _Ctx, indent: int) -> list[str]:
        self.gdn_mode = "leaf"
        bv = ctx.block_var or "hv"
        hv_expr = bv
        hk_expr = f"({bv} % hk_heads)"
        t0_expr = "(c * 32)"
        off = 1 if len(call.args) and (_ann_str(call.args[0]) or "").startswith("hexagon.") else 0
        p = lambda n: self._gdn_ptr_arg(call.args[n + off])
        if name == "load_state128":
            lines = [f"hrt_tlgdn_load_state({p(0)} + (size_t){self._gdn_arg(call, 2)} * 128 * 128, {p(1)});"]
        elif name == "store_state128":
            lines = [f"hrt_tlgdn_store_state({p(0)}, {p(1)} + (size_t){self._gdn_arg(call, 2)} * 128 * 128);"]
        elif name == "load_h2f_rows128":
            lines = [f"hrt_tlgdn_load_h2f_rows({p(0)}, {p(1)}, {p(2)}, {p(3)}, {p(4)}, {p(5)}, {self._gdn_arg(call, 6)}, {self._gdn_arg(call, 7)}, {self._gdn_arg(call, 8)}, {self._gdn_arg(call, 9)});"]
        elif name == "scan_exp32":
            lines = [f"hrt_tlgdn_scan_exp32({p(0)}, {p(1)}, {p(2)}, {p(3)}, {p(4)}, {p(5)}, {self._gdn_arg(call, 6)}, {self._gdn_arg(call, 7)}, {self._gdn_arg(call, 8)});"]
        elif name == "dot128x2_store":
            lines = [f"hrt_tlgdn_dot128x2_store({p(0)}, {p(1)}, {p(2)}, {p(3)}, {p(4)}, {p(5)}, {p(6)}, {self._gdn_arg(call, 7)}, {self._gdn_arg(call, 8)});"]
        elif name == "state_x2_matvec128":
            lines = [f"hrt_tlgdn_state_x2_matvec128({p(0)}, {p(1)}, {p(2)}, {p(3)}, {p(4)}, {self._gdn_arg(call, 5)});"]
        elif name == "affine_rows128":
            lines = [f"hrt_tlgdn_affine_row128({p(0)}, {p(1)}, {p(2)}, {p(3)}, {self._gdn_arg(call, 4)});"]
        elif name == "forward_solve32":
            lines = [f"hrt_tlgdn_forward_solve32({p(0)}, {p(1)}, {self._gdn_arg(call, 2)});"]
        elif name == "output_rows128":
            lines = [f"hrt_tlgdn_output_row128({p(0)}, {p(1)}, {p(2)}, {p(3)}, {p(4)}, {self._gdn_arg(call, 5)}, {self._gdn_arg(call, 6)}, {self._gdn_arg(call, 7)}, {self._gdn_arg(call, 8)});"]
        elif name == "state_decay_rows128":
            lines = [f"hrt_tlgdn_state_decay_rows128({p(0)}, {p(1)}, {p(2)});"]
        elif name == "state_update32":
            lines = [f"hrt_tlgdn_state_update32({p(0)}, {p(1)}, {p(2)}, {p(3)});"]
        else:
            raise HexagonEmitError(f"白名单: 未支持的 GDN leaf {name}")
        return [self._ind(indent, s.replace("(&slot->eGC)", "&slot->eGC")) for s in lines]

    def _emit_vector_store(self, op: BufferStore, ctx: _Ctx, indent: int) -> list[str]:
        self.has_vector_kernel = True
        if ctx.vec_var is None or ctx.vec_extent is None:
            raise HexagonEmitError(f"R2: vector store 缺少 T.vectorized 上下文{_loc(op)}")
        buf = op.buffer
        scope = _buffer_scope(buf)
        if scope != "global" and not _is_vtcm(scope) and not _is_wscratch(scope):
            raise HexagonEmitError(f"R2: 当前向量表达式 store 只支持 global/vtcm/wscratch 输出, 实际 {scope}{_loc(op)}")
        if str(buf.dtype) not in ("float16", "float32"):
            raise HexagonEmitError(f"白名单: 当前向量 store 只支持 fp16/fp32, 实际 {buf.dtype}{_loc(op)}")
        base = self._vector_index_c(op.indices, ctx, buf)
        expr_lines, val = self._expr_hvx(op.value, ctx, indent + 1)
        store_name = val.names[0]
        self._check_vector_alignment(op.indices, ctx, op, buf, base)
        ptr = self._buffer_ptr_c(buf)
        # 设计文档 §3:DDR 直读必须带 dcfetch(单线程无预取只有 7-14GB/s)。
        # 对表达式里出现的每个 global 读 buffer,在当前向量索引前方
        # dcfetch_elems 个元素处发预取(每 128B 一次,dcfetch 越界是安全 hint,
        # 手写 kernel 惯例,见 attnops_gemm.c 的 gm_copy)。
        load_ptrs: list[str] = []

        def _collect_loads(e: Any) -> None:
            if isinstance(e, BufferLoad) and _buffer_scope(e.buffer) == "global":
                p = self._buffer_ptr_c(e.buffer)
                if p not in load_ptrs:
                    load_ptrs.append(p)

        try:
            tirx.stmt_functor.post_order_visit(op.value, _collect_loads)
        except Exception:
            pass
        idx_name = self._new_scalar_tmp("hvx_idx")
        lines = [self._ind(indent, f"size_t {idx_name} = (size_t)({base});")]
        for p in load_ptrs:
            lines.append(
                self._ind(indent, f"Q6_dcfetch_A((void *)({p} + {idx_name} + {self.dcfetch_elems}));")
            )
        if scope == "global":
            limit = self._shape_numel_c(buf.shape) if buf is not None and len(buf.shape) > 1 else "total"
            guard = "total" if limit == "total" else f"(size_t)({limit})"
            store_lanes = 64
            if str(buf.dtype) == "float32" and not (len(val.names) == 2 and val.names[0] != val.names[1]):
                store_lanes = 32
            lines += [
                self._ind(indent, f"if ({idx_name} + {store_lanes} <= {guard}) {{"),
                *expr_lines,
                self._ind(indent + 1, f"*(HVX_Vector *)({ptr} + {idx_name}) = {store_name};"),
            self._ind(indent, "}"),
            ]
        else:
            lines += [*expr_lines, self._ind(indent, f"*(HVX_Vector *)({ptr} + {idx_name}) = {store_name};")]
        if str(buf.dtype) == "float32" and val.dtype == "float32" and len(val.names) == 2 and val.names[0] != val.names[1]:
            # One HVX register carries 32 fp32 lanes.  64-lane fp32 elementwise
            # expressions are a lo/hi register pair; store both halves.  A
            # 32-lane context reuses a single register ([name, name]) and must
            # NOT emit the +32 store, which would clobber the next row.
            if scope == "global":
                lines.insert(-1, self._ind(indent + 1, f"*(HVX_Vector *)({ptr} + {idx_name} + 32) = {val.names[1]};"))
            else:
                lines.append(self._ind(indent, f"*(HVX_Vector *)({ptr} + {idx_name} + 32) = {val.names[1]};"))
        return lines

    def _contains_parallel_for(self, op: Any) -> bool:
        if isinstance(op, For):
            _k = str(getattr(op, "kind", "")).lower()
            if "parallel" in _k or _k == "1":
                return True
            return self._contains_parallel_for(op.body)
        if isinstance(op, AttrStmt):
            return self._contains_parallel_for(op.body)
        if isinstance(op, (SBlockRealize, SBlock)):
            return self._contains_parallel_for(op.block if isinstance(op, SBlockRealize) else op.body)
        if isinstance(op, SeqStmt):
            return any(self._contains_parallel_for(x) for x in op.seq)
        if isinstance(op, IfThenElse):
            return self._contains_parallel_for(op.then_case) or (op.else_case is not None and self._contains_parallel_for(op.else_case))
        return False

    def _contains_vector_for(self, op: Any) -> bool:
        if isinstance(op, For):
            kind = str(getattr(op, "kind", ""))
            if "vectorized" in kind.lower() or str(getattr(op, "kind", "")) == "2":
                return True
            return self._contains_vector_for(op.body)
        if isinstance(op, AttrStmt):
            return self._contains_vector_for(op.body)
        if isinstance(op, (SBlockRealize, SBlock)):
            return self._contains_vector_for(op.block if isinstance(op, SBlockRealize) else op.body)
        if isinstance(op, SeqStmt):
            return any(self._contains_vector_for(x) for x in op.seq)
        if isinstance(op, IfThenElse):
            return self._contains_vector_for(op.then_case) or (op.else_case is not None and self._contains_vector_for(op.else_case))
        return False

    def _contains_gdn_leaf(self, op: Any) -> bool:
        if isinstance(op, Evaluate) and isinstance(op.value, Call):
            call = op.value
            if _call_op_name(call) in ("tir.call_pure_extern", "tirx.call_pure_extern") and len(call.args) >= 1:
                callee = _ann_str(call.args[0])
                if callee and callee.startswith("hexagon.") and callee.split(".", 1)[1] in self._gdn_leaf_names():
                    return True
        if isinstance(op, For):
            return self._contains_gdn_leaf(op.body)
        if isinstance(op, AttrStmt):
            return self._contains_gdn_leaf(op.body)
        if isinstance(op, (SBlockRealize, SBlock)):
            return self._contains_gdn_leaf(op.block if isinstance(op, SBlockRealize) else op.body)
        if isinstance(op, SeqStmt):
            return any(self._contains_gdn_leaf(x) for x in op.seq)
        if isinstance(op, IfThenElse):
            return self._contains_gdn_leaf(op.then_case) or (op.else_case is not None and self._contains_gdn_leaf(op.else_case))
        return False

    def _buffer_ptr_c(self, buf: Buffer | None) -> str:
        bname = _buffer_name(buf)
        if _buffer_scope(buf) == "global" and bname in self.global_ptrs:
            return self.global_ptrs[bname]
        if _is_wscratch(_buffer_scope(buf)):
            field = self._wscratch_field(buf)
            if field == "eGC":
                return "(&slot->eGC)"
            return f"slot->{field}"
        if _is_vtcm(_buffer_scope(buf)):
            alloc = next((a for a in self.allocs if a.name == bname), None)
            if alloc is None or alloc.offset is None:
                raise HexagonEmitError(f"R3: 找不到 VTCM buffer {bname} 的静态偏移")
            return f"(({self._buffer_elem_ctype(buf)} *)(V + {alloc.offset}))"
        return bname

    def _assign_vtcm_offsets(self) -> int:
        if self._vtcm_offsets:
            max_end = 0
            for alloc in self.allocs:
                if alloc.name in self._vtcm_offsets:
                    alloc.offset = self._vtcm_offsets[alloc.name]
                    max_end = max(max_end, alloc.offset + alloc.bytes)
            return max_end
        if any(a.bytes for a in self.allocs):
            # StoragePlan owns static VTCM placement.  This fallback is only for
            # direct emitter unit tests that bypass the pipeline; production
            # lowering always supplies hexagon.vtcm_offsets on the PrimFunc.
            pass
        off = 0
        for alloc in self.allocs:
            if alloc.bytes == 0:
                continue
            off = _align(off)
            alloc.offset = off
            off += alloc.bytes
        return off

    def _linear_index_expr(self, indices: Any, buf: Buffer | None = None) -> Any:
        """Return the flattened element index as a TIR PrimExpr, before C rendering."""

        if len(indices) == 1:
            return indices[0]
        terms: list[Any] = []
        for i, idx in enumerate(indices):
            suffix_shape = buf.shape[i + 1:] if buf is not None and len(buf.shape) > i + 1 else ()
            stride: Any = 1
            for dim in suffix_shape:
                iv = _i64(dim)
                stride = stride * iv if iv is not None and isinstance(stride, int) else stride * dim
            terms.append(idx if stride == 1 else idx * stride)
        if not terms:
            return IntImm("int32", 0)
        expr = terms[0]
        for term in terms[1:]:
            expr = expr + term
        return expr

    def _expr_contains_var(self, expr: Any, name: str | None) -> bool:
        if not name:
            return False
        found = False

        def visit(node: Any) -> None:
            nonlocal found
            if found:
                return
            if (hasattr(node, "name") or hasattr(node, "name_hint")) and _var_name(node) == name:
                found = True

        try:
            tirx.stmt_functor.post_order_visit(expr, visit)
        except Exception:
            if (hasattr(expr, "name") or hasattr(expr, "name_hint")) and _var_name(expr) == name:
                found = True
        return found

    def _find_var_by_name(self, expr: Any, name: str | None) -> Any | None:
        if not name:
            return None
        found = None

        def visit(node: Any) -> None:
            nonlocal found
            if found is None and (hasattr(node, "name") or hasattr(node, "name_hint")) and _var_name(node) == name:
                found = node

        try:
            tirx.stmt_functor.post_order_visit(expr, visit)
        except Exception:
            if (hasattr(expr, "name") or hasattr(expr, "name_hint")) and _var_name(expr) == name:
                found = expr
        return found

    def _buffer_base_byte_offset(self, buf: Buffer | None) -> int:
        """Known static byte offset of a buffer base; unknown/aligned bases use 0.

        Global/rpcmem pointers and wscratch slots follow the existing ABI
        assumption that their base pointers are HVX-aligned.  VTCM buffers have
        explicit static placement, so include their byte offset in the proof.
        """

        if buf is None or not _is_vtcm(_buffer_scope(buf)):
            return 0
        bname = _buffer_name(buf)
        alloc = next((a for a in self.allocs if a.name == bname), None)
        if alloc is None:
            return 0
        if alloc.offset is None:
            self._assign_vtcm_offsets()
        return int(alloc.offset or 0)

    def _is_aligned_byte_expr(self, byte_expr: Any, alignment: int, ctx: _Ctx | None = None) -> bool:
        # T.vectorized loops advance the induction variable by one full HVX
        # vector at C level (32 fp32 lanes or 64 fp16 lanes).  Encode that loop
        # fact in the expression given to the analyzer by substituting
        # vec_var := vec_step * fresh_symbol; otherwise modular_set only sees an
        # unconstrained scalar loop variable and cannot prove row*stride+vec.
        if ctx is not None and ctx.vec_var is not None and ctx.vec_extent is not None:
            v = self._find_var_by_name(byte_expr, ctx.vec_var)
            if v is not None:
                step = 32 if ctx.vec_extent == 32 else 64
                fresh = tirx.Var(f"{ctx.vec_var}_hvx_step", str(getattr(v, "dtype", "int32") or "int32"))
                try:
                    byte_expr = tirx.stmt_functor.substitute(byte_expr, {v: fresh * step})
                except Exception:
                    pass
        analyzer = arith.Analyzer()
        try:
            rem = analyzer.rewrite_simplify(byte_expr % alignment)
        except AttributeError:
            rem = analyzer.simplify(byte_expr % alignment)
        except Exception:
            rem = None
        if _i64(rem) == 0:
            return True
        try:
            if analyzer.can_prove((byte_expr % alignment) == 0):
                return True
        except Exception:
            pass
        try:
            mod = analyzer.modular_set(byte_expr)
            coeff = int(getattr(mod, "coeff"))
            base = int(getattr(mod, "base"))
            return coeff % alignment == 0 and base % alignment == 0
        except Exception:
            return False

    def _check_vector_alignment(self, indices: Any, ctx: _Ctx, op: Any, buf: Buffer | None = None, base: str | None = None) -> None:
        # R1: the HVX pointer must denote a 128B boundary.  Prove that property
        # on the TIR element-index expression before rendering C: byte_offset =
        # flattened_index * sizeof(dtype) + known_static_buffer_base.  If TVM's
        # arithmetic analyzer cannot prove divisibility by 128, keep the old
        # conservative behavior (reject this vector load/store).
        index_expr = self._linear_index_expr(indices, buf)
        rendered = base if base is not None else self._expr_c(index_expr)
        if not self._expr_contains_var(index_expr, ctx.vec_var):
            raise HexagonEmitError(f"R1: HVX 向量 load/store 地址必须包含 T.vectorized 变量并按 128B 对齐: {rendered}{_loc(op)}")
        elem_bytes = _dtype_bytes(str(buf.dtype)) if buf is not None else 1
        vec_step = 32 if ctx.vec_extent == 32 else 64
        required_alignment = min(128, elem_bytes * vec_step)
        byte_expr = index_expr * elem_bytes + self._buffer_base_byte_offset(buf)
        if not self._is_aligned_byte_expr(byte_expr, required_alignment, ctx):
            raise HexagonEmitError(f"R1: HVX 向量 load/store 地址不可证 {required_alignment}B 对齐: {rendered}{_loc(op)}")

    def _vector_index_c(self, indices: Any, ctx: _Ctx, buf: Buffer | None = None) -> str:
        if len(indices) == 1:
            return self._expr_c(indices[0])
        terms: list[str] = []
        for i, idx in enumerate(indices):
            idx_c = self._expr_c(idx)
            suffix_shape = buf.shape[i + 1:] if buf is not None and len(buf.shape) > i + 1 else ()
            stride = self._shape_numel_c(suffix_shape) if suffix_shape else "1"
            terms.append(f"({idx_c})" if stride == "1" else f"({idx_c}) * {stride}")
        return " + ".join(terms) if terms else "0"

    def _shape_numel_c(self, shape: Any) -> str:
        terms: list[str] = []
        for dim in shape:
            iv = _i64(dim)
            terms.append(str(iv) if iv is not None else f"({self._expr_c(dim)})")
        return " * ".join(terms) if terms else "1"

    def _region_base_c(self, expr: Any, ctx: _Ctx) -> str:
        load = _region_load(expr)
        if load is None:
            raise HexagonEmitError(f"白名单: 1D copy 必须来自 tl.region(BufferLoad){_loc(expr)}")
        return self._vector_index_c(load.indices, ctx, load.buffer)

    def _expr_hvx(self, e: Any, ctx: _Ctx, indent: int = 0) -> tuple[list[str], _HVXVal]:
        """Lower a pure elementwise TIR expression tree into SSA-like HVX C.

        Dtype rule (R7): fp16 is the default elementwise domain.  If any operand
        is fp32 (buffer, cast, or constant under a fp32 parent), the current
        binary/unary subtree is promoted to fp32 and represented as two
        HVX_Vector values covering the 64 fp16 lanes (lo/hi 32-lane fp32).
        Unsupported nodes fail with HexagonEmitError instead of falling back to
        scalar C.
        """
        lines: list[str] = []
        val = self._lower_hvx_expr(e, ctx, lines, None, indent)
        return lines, val

    def _new_scalar_tmp(self, prefix: str = "sc") -> str:
        n = getattr(self, "_scalar_tmp", 0)
        self._scalar_tmp = n + 1
        return f"{prefix}{n}"

    def _scalar_ctype(self, dtype: str) -> str:
        return "f32" if dtype == "float32" else ("f16" if dtype == "float16" else "int")

    def _expr_scalar(self, e: Any, want: str | None = None) -> str:
        imm = self._imm_value(e)
        if imm is not None:
            if (want or self._expr_dtype_hint(e)) == "float32":
                s = f"{float(imm):.9g}"
                if "." not in s and "e" not in s and "E" not in s:
                    s += ".0"
                return f"{s}f"
            return str(imm)
        if isinstance(e, BufferLoad):
            if _is_vtcm(_buffer_scope(e.buffer)):
                raise HexagonEmitError(f"R2: 禁止对 VTCM buffer {_buffer_name(e.buffer)} 做标量 load/store{_loc(e)}")
            if _buffer_scope(e.buffer) == "local.var":
                return _buffer_name(e.buffer)
            ptr = self._buffer_ptr_c(e.buffer)
            idx = self._vector_index_c(e.indices, _Ctx(), e.buffer)
            return f"{ptr}[(size_t)({idx})]"
        if hasattr(e, "name") or hasattr(e, "name_hint"):
            return _var_name(e)
        if isinstance(e, Call):
            opname = _call_op_name(e)
            if opname in ("tir.call_pure_extern", "tirx.call_pure_extern") and len(e.args) >= 1:
                callee = _ann_str(e.args[0])
                if callee == "hexagon.exp_fp32":
                    return f"expf({self._expr_scalar(e.args[1], 'float32')})"
            if opname in ("tir.exp", "tirx.exp", "exp") or str(opname).endswith(".exp"):
                return f"expf({self._expr_scalar(e.args[0], 'float32')})"
            raise HexagonEmitError(f"白名单: 标量表达式暂不支持 Call {_call_op_name(e)}{_loc(e)}")
        cls = type(e).__name__
        cls_key = cls.lower()
        if cls_key in ("add", "sub", "mul", "div", "floordiv", "floormod", "mod", "lt", "le", "gt", "ge", "eq", "ne"):
            op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "floordiv": "/", "floormod": "%", "mod": "%",
                  "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}[cls_key]
            return f"({self._expr_scalar(e.a, want)} {op} {self._expr_scalar(e.b, want)})"
        if cls in ("Min", "Max"):
            fn = "fminf" if cls == "Min" else "fmaxf"
            return f"{fn}({self._expr_scalar(e.a, 'float32')}, {self._expr_scalar(e.b, 'float32')})"
        if cls == "Cast":
            dst = str(getattr(e, "dtype", ""))
            src_hint = self._expr_dtype_hint(e.value)
            if dst == "float32" and src_hint == "float16":
                return f"hrt_f16_to_f32({self._expr_scalar(e.value, 'float16')})"
            if dst == "float16" and src_hint == "float32":
                return f"hrt_f32_to_f16({self._expr_scalar(e.value, 'float32')})"
            if dst == "float16" and src_hint == "float16":
                if isinstance(e.value, BufferLoad):
                    return self._expr_scalar(e.value, 'float16')
                return f"hrt_f32_to_f16({self._expr_scalar(e.value, 'float32')})"
            if dst == "float16" and src_hint is None:
                return f"hrt_f32_to_f16({self._expr_scalar(e.value, 'float32')})"
            return f"({self._scalar_ctype(dst)})({self._expr_scalar(e.value, dst)})"
        if cls == "Neg":
            return f"(-{self._expr_scalar(e.value, want)})"
        raise HexagonEmitError(f"白名单: 标量表达式暂不支持 {cls}{_loc(e)}")

    def _emit_scalar_store(self, op: BufferStore, ctx: _Ctx, indent: int) -> list[str]:
        self.has_scalar_kernel = True
        if _is_vtcm(_buffer_scope(op.buffer)):
            raise HexagonEmitError(f"R2: 禁止对 VTCM buffer {_buffer_name(op.buffer)} 做标量 load/store{_loc(op)}")
        if _buffer_scope(op.buffer) == "local.var":
            val = self._expr_scalar(op.value, str(op.buffer.dtype))
            name = _buffer_name(op.buffer)
            if name not in self.scalar_vars:
                self.scalar_vars[name] = str(op.buffer.dtype)
                return [self._ind(indent, f"{self._scalar_ctype(str(op.buffer.dtype))} {name} = {val};")]
            return [self._ind(indent, f"{name} = {val};")]
        ptr = self._buffer_ptr_c(op.buffer)
        idx = self._vector_index_c(op.indices, ctx, op.buffer)
        dst_dtype = str(op.buffer.dtype)
        src_hint = self._expr_dtype_hint(op.value)
        if dst_dtype == "float16" and src_hint == "float16" and isinstance(op.value, BufferLoad):
            val = self._expr_scalar(op.value, dst_dtype)
        elif dst_dtype == "float16" and src_hint == "float16" and type(op.value).__name__ == "Cast":
            val = self._expr_scalar(op.value, dst_dtype)
        elif dst_dtype == "float16":
            val = f"hrt_f32_to_f16({self._expr_scalar(op.value, 'float32')})"
        else:
            val = self._expr_scalar(op.value, dst_dtype)
        return [self._ind(indent, f"{ptr}[(size_t)({idx})] = {val};")]

    def _emit_reduce(self, call: Call, ctx: _Ctx, indent: int) -> list[str]:
        self.has_scalar_kernel = True
        if len(call.args) < 6:
            raise HexagonEmitError("T.reduce_sum: 参数数量不足")
        src = _region_buffer(call.args[0])
        dst = _region_buffer(call.args[1])
        rtype = _ann_str(call.args[2])
        dim = _i64(call.args[3])
        if rtype == "max":
            return self._emit_reduce_max(call, ctx, indent)
        if rtype != "sum" or dim not in (0, -1):
            raise HexagonEmitError(f"T.reduce_*: 当前 Hexagon 只支持 dim=0 reduce_sum/reduce_max, 实际 type={rtype}, dim={dim}")
        if src is None or dst is None:
            raise HexagonEmitError("T.reduce_sum: region buffer 提取失败")
        if _is_vtcm(_buffer_scope(dst)):
            raise HexagonEmitError(f"R2: 归约结果禁止标量写 VTCM buffer {_buffer_name(dst)}{_loc(call)}")
        n = _shape_numel(src.shape)
        if n != 128:
            raise HexagonEmitError(f"R7: 当前 HVX reduce_sum 支持 128 维向量, 实际 {n}")
        sp = self._buffer_ptr_c(src)
        dp = self._buffer_ptr_c(dst)
        dtype = str(src.dtype)
        if dtype == "float32":
            rhs = f"hrt_reduce_sum_f32_128({sp})"
        elif dtype == "float16":
            rhs = f"hrt_reduce_sum_f16_128({sp})"
        else:
            raise HexagonEmitError(f"R7: reduce_sum 只支持 fp16/fp32, 实际 {dtype}")
        if _buffer_scope(dst) == "local.var":
            return [self._ind(indent, f"{dp} = {rhs};")]
        return [self._ind(indent, f"{dp}[0] = {rhs};")]

    def _emit_reduce_sum128(self, call: Call, ctx: _Ctx, indent: int) -> list[str]:
        self.has_scalar_kernel = True
        off = 1 if len(call.args) and (_ann_str(call.args[0]) or "").startswith("hexagon.") else 0
        if len(call.args) < off + 2:
            raise HexagonEmitError("T.reduce_sum: reduce_sum128 参数数量不足")
        src = call.args[off]
        dst = call.args[off + 1]
        sbuf = getattr(src, "buffer", None) or self._buffer_from_data_arg(src)
        dbuf = getattr(dst, "buffer", None) or self._buffer_from_data_arg(dst)
        if _is_vtcm(_buffer_scope(dbuf)):
            raise HexagonEmitError(f"R2: 归约结果禁止标量写 VTCM buffer {_buffer_name(dbuf)}{_loc(call)}")
        ext = _region_extents(src)
        n = ext[-1] if ext else ((_i64(sbuf.shape[-1]) if sbuf is not None and len(sbuf.shape) > 1 else _shape_numel(sbuf.shape)) if sbuf is not None else None)
        if n != 128:
            raise HexagonEmitError(f"R7: 当前 HVX reduce_sum 支持 128 维向量, 实际 {n}")
        sp = self._buffer_ptr_c(sbuf)
        dp = self._buffer_ptr_c(dbuf)
        dtype = str(sbuf.dtype)
        rhs = "hrt_reduce_sum_f32_128" if dtype == "float32" else "hrt_reduce_sum_f16_128"
        if dtype not in ("float32", "float16"):
            raise HexagonEmitError(f"R7: reduce_sum 只支持 fp16/fp32, 实际 {dtype}")
        if _buffer_scope(dbuf) == "local.var":
            return [self._ind(indent, f"{dp} = {rhs}({sp});")]
        return [self._ind(indent, f"{dp}[0] = {rhs}({sp});")]

    def _emit_reduce_max(self, call: Call, ctx: _Ctx, indent: int) -> list[str]:
        self.has_scalar_kernel = True
        off = 1 if len(call.args) and (_ann_str(call.args[0]) or "").startswith("hexagon.") else 0
        if len(call.args) < off + 2:
            raise HexagonEmitError("T.reduce_max: reduce_max 参数数量不足")
        src = call.args[off]
        dst = call.args[off + 1]
        sbuf = _region_buffer(src) or getattr(src, "buffer", None) or self._buffer_from_data_arg(src)
        dbuf = _region_buffer(dst) or getattr(dst, "buffer", None) or self._buffer_from_data_arg(dst)
        if _is_vtcm(_buffer_scope(dbuf)):
            raise HexagonEmitError(f"R2: 归约结果禁止标量写 VTCM buffer {_buffer_name(dbuf)}{_loc(call)}")
        ext = _region_extents(src)
        n = ext[-1] if ext else ((_i64(sbuf.shape[-1]) if sbuf is not None and len(sbuf.shape) > 1 else _shape_numel(sbuf.shape)) if sbuf is not None else None)
        if n not in (32, 128):
            raise HexagonEmitError(f"FA: 当前 HVX reduce_max 支持 32/128 维行向量, 实际 {n}")
        sp = self._buffer_ptr_c(sbuf)
        dp = self._buffer_ptr_c(dbuf)
        dtype = str(sbuf.dtype)
        if dtype == "float32":
            rhs = "hrt_reduce_max_f32_32" if n == 32 else "hrt_reduce_max_f32_128"
        elif dtype == "float16":
            rhs = "hrt_reduce_max_f16_32" if n == 32 else "hrt_reduce_max_f16_128"
        else:
            rhs = ""
        if dtype not in ("float32", "float16"):
            raise HexagonEmitError(f"FA: reduce_max 只支持 fp16/fp32, 实际 {dtype}")
        if _buffer_scope(dbuf) == "local.var":
            return [self._ind(indent, f"{dp} = {rhs}({sp});")]
        return [self._ind(indent, f"{dp}[0] = {rhs}({sp});")]

    def _data_ptr_expr(self, arg: Any, ctx: _Ctx) -> str:
        if isinstance(arg, BufferLoad):
            ptr = self._buffer_ptr_c(arg.buffer)
            if len(arg.indices) == 2 and _i64(arg.indices[1]) == 0:
                row = self._expr_c(arg.indices[0])
                stride = _i64(arg.buffer.shape[1]) if len(arg.buffer.shape) >= 2 else None
                if stride is not None:
                    return f"{ptr} + (size_t){row} * {stride}"
            idx = self._vector_index_c(arg.indices, ctx, arg.buffer)
            return f"{ptr} + (size_t)({idx})"
        buf = getattr(arg, "buffer", None) or self._buffer_from_data_arg(arg)
        if buf is not None:
            return self._buffer_ptr_c(buf)
        return self._expr_c(arg)

    def _dst_scalar_expr(self, arg: Any, ctx: _Ctx) -> tuple[Buffer | None, str]:
        buf = getattr(arg, "buffer", None) or self._buffer_from_data_arg(arg)
        if buf is None:
            return None, self._expr_c(arg)
        ptr = self._buffer_ptr_c(buf)
        if _buffer_scope(buf) == "local.var":
            return buf, f"&{ptr}"
        if isinstance(arg, BufferLoad):
            idx = self._vector_index_c(arg.indices, ctx, buf)
            return buf, f"&{ptr}[(size_t)({idx})]"
        return buf, f"&{ptr}[0]"

    def _emit_reduce_prod128(self, call: Call, ctx: _Ctx, indent: int) -> list[str]:
        self.has_scalar_kernel = True
        off = 1 if len(call.args) and (_ann_str(call.args[0]) or "").startswith("hexagon.") else 0
        if len(call.args) < off + 3:
            raise HexagonEmitError("T.reduce_sum: reduce_prod128 参数数量不足")
        x = self._data_ptr_expr(call.args[off], ctx)
        y = self._data_ptr_expr(call.args[off + 1], ctx)
        dbuf, dptr = self._dst_scalar_expr(call.args[off + 2], ctx)
        if _is_vtcm(_buffer_scope(dbuf)):
            raise HexagonEmitError(f"R2: 归约结果禁止标量写 VTCM buffer {_buffer_name(dbuf)}{_loc(call)}")
        if dbuf is not None and _buffer_scope(dbuf) == "local.var":
            return [self._ind(indent, f"{dptr[1:]} = hrt_reduce_sum_f32_128_prod({x}, {y});")]
        return [self._ind(indent, f"*{dptr} = hrt_reduce_sum_f32_128_prod({x}, {y});")]

    def _emit_reduce_prod2_128(self, call: Call, ctx: _Ctx, indent: int) -> list[str]:
        self.has_scalar_kernel = True
        off = 1 if len(call.args) and (_ann_str(call.args[0]) or "").startswith("hexagon.") else 0
        if len(call.args) < off + 6:
            raise HexagonEmitError("T.reduce_sum: reduce_prod2_128 参数数量不足")
        x0 = self._data_ptr_expr(call.args[off], ctx)
        y0 = self._data_ptr_expr(call.args[off + 1], ctx)
        d0buf, d0 = self._dst_scalar_expr(call.args[off + 2], ctx)
        x1 = self._data_ptr_expr(call.args[off + 3], ctx)
        y1 = self._data_ptr_expr(call.args[off + 4], ctx)
        d1buf, d1 = self._dst_scalar_expr(call.args[off + 5], ctx)
        if _is_vtcm(_buffer_scope(d0buf)) or _is_vtcm(_buffer_scope(d1buf)):
            raise HexagonEmitError("R2: 归约结果禁止标量写 VTCM buffer")
        return [self._ind(indent, f"hrt_reduce_sum_f32_128_prod2({x0}, {y0}, {x1}, {y1}, {d0}, {d1});")]

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

    def _region_min_indices(self, arg: Any, buf: Buffer | None) -> list[Any]:
        if isinstance(arg, Call) and _call_op_name(arg) == "tl.region" and arg.args:
            load = arg.args[0]
            if isinstance(load, BufferLoad):
                return list(load.indices)
        if isinstance(arg, BufferLoad):
            return list(arg.indices)
        region = getattr(arg, "region", None)
        if region is not None:
            return [r.min for r in region]
        if buf is not None:
            return [IntImm("int32", 0) for _ in buf.shape]
        return []

    def _flatten_view_rc(self, mins: list[Any], buf: Buffer | None) -> tuple[Any, Any]:
        """Fold leading-dim view mins into the row index of the trailing 2-D plane.

        Point views such as state_ah[hv, 0, 0] denote the trailing 2-D tile
        rooted there; the AH/WH byte offset must see hv * shape[-2] rows.
        """

        return hexagon_fold_view_base_rc(mins, buf)

    def _hexagon_layout_mode(self, buf: Buffer | None) -> str:
        """Best-effort Python-side layout mode for Hexagon tiled buffers."""

        scope = _buffer_scope(buf)
        if scope.endswith(".ah"):
            return "ah"
        if scope.endswith(".wh"):
            return "wh"
        if _is_acc(scope):
            return "ah"
        return "rm"

    def _hexagon_tile_layout(self, buf: Buffer, mode: str | None = None) -> Any:
        mode = mode or self._hexagon_layout_mode(buf)
        if mode not in ("ah", "wh"):
            raise HexagonEmitError(f"R5: buffer {_buffer_name(buf)} 不是 AH/WH tiled layout, mode={mode}")
        return make_layout(mode, buf)

    def _layout_linear_index_c(self, layout: Any, indices: list[Any]) -> str:
        vars_ = list(layout.get_forward_vars())
        if len(vars_) != len(indices):
            raise HexagonEmitError(f"R5: layout rank {len(vars_)} 与索引 rank {len(indices)} 不匹配")
        expr = layout.get_linearized_forward_index()
        subst = {v: indices[i] for i, v in enumerate(vars_)}
        try:
            expr = tirx.stmt_functor.substitute(expr, subst)
        except Exception as exc:  # pragma: no cover - defensive across TVM builds
            raise HexagonEmitError(f"R5: 无法求值 Hexagon layout index map: {exc}") from exc
        return self._expr_c(expr)

    def _tile_layout_addr(self, buf: Buffer, row: Any, col: Any,
                          *, mode: str | None = None,
                          region_rows: int | None = None,
                          region_cols: int | None = None,
                          force_anchor: bool = True) -> _TileLayoutAddr:
        """Evaluate the canonical AH/WH layout map and return the tile byte offset.

        The authoritative maps live in ``src/hexagon/layouts.cc``.  This helper
        constructs that registered layout, evaluates it at the logical region
        root (leading dims folded into ``row`` plus tile-local (0,0)), then
        converts the linearized element address to tile granularity.  All HMX
        staging and GEMM operand addressing must route through this function.

        ``region_cols`` is used for point-view staging of a logical tile nested
        under leading dimensions; it mirrors the layout-expanded map for that
        region so existing anchors keep their historical compact C spelling.
        """

        mode = mode or self._hexagon_layout_mode(buf)
        if mode not in ("ah", "wh"):
            raise HexagonEmitError(f"R5: buffer {_buffer_name(buf)} 不是 AH/WH tiled layout, mode={mode}")
        if _i64(row) == 0 and _i64(col) == 0:
            return _TileLayoutAddr(mode, "0", "0", "0", "0")

        # Fast-path anchors: render exactly the historical expressions when the
        # evaluated map reduces to the same tile index.  Tests sweep the scalar
        # formula to guard the equivalence.
        row_c = row if isinstance(row, str) else self._expr_c(row)
        col_c = col if isinstance(col, str) else self._expr_c(col)
        if force_anchor:
            if region_cols is not None:
                tile_cols_c = str(region_cols // 32)
            else:
                tile_cols = _i64(buf.shape[-1]) if len(buf.shape) >= 2 else None
                tile_cols_c = str(tile_cols // 32) if tile_cols is not None else f"(({self._expr_c(buf.shape[-1])}) / 32)"
            tile_index = f"((size_t)({row_c}) / 32) * {tile_cols_c} + ((size_t)({col_c}) / 32)"
            return _TileLayoutAddr(mode, row_c, col_c, tile_index, f"({tile_index}) * HRT_TILE_BYTES")

        shape = list(buf.shape)
        if region_rows is not None and region_cols is not None and len(shape) >= 2:
            shape = list(shape[:-2]) + [IntImm("int32", region_rows), IntImm("int32", region_cols)]
            fake = tirx.decl_buffer(tuple(shape), buf.dtype, name=_buffer_name(buf), scope=_buffer_scope(buf))
            layout = self._hexagon_tile_layout(fake, mode)
        else:
            layout = self._hexagon_tile_layout(buf, mode)
        idxs = [IntImm("int32", 0) for _ in shape]
        idxs[-2], idxs[-1] = row, col
        lin = self._layout_linear_index_c(layout, idxs)
        tile_index = f"((size_t)({lin}) / HRT_TILE_ELEMS)"
        return _TileLayoutAddr(mode, row_c, col_c, tile_index, f"{tile_index} * HRT_TILE_BYTES")

    def _hmx_tile_base_offset_c(self, arg: Any, buf: Buffer) -> str:
        """Byte offset of a BufferLoad/tl.region base in AH/WH tile storage."""

        mins = self._region_min_indices(arg, buf)
        if len(mins) < 2 or len(buf.shape) < 2:
            return "0"
        row, col = self._flatten_view_rc(mins, buf)
        return self._tile_layout_addr(buf, row, col).byte_offset

    def _hmx_tile_base_offset_from_min_c(self, row: Any, col: Any, buf: Buffer, tile_cols_override: int | None = None,
                                         tile_rows_override: int | None = None) -> str:
        region_cols = tile_cols_override * 32 if tile_cols_override is not None else None
        region_rows = tile_rows_override * 32 if tile_rows_override is not None else None
        return self._tile_layout_addr(buf, row, col, region_rows=region_rows, region_cols=region_cols).byte_offset

    def _hmx_tile_base_offset_from_region_c(self, mins: list[Any], buf: Buffer | None, rows: int, cols: int) -> str:
        """Byte offset for AH/WH staging regions using logical 32x32 tiles.

        At this point AH/WH buffer shapes may already be layout-expanded, but
        the copy region extents remain the logical trailing matrix shape.  Use
        those logical extents for leading-dimension stride so ``buf[h, 0, 0]``
        advances by one full logical matrix plane, matching the HMX operand
        base emitted by ``gemm_hmx._flatten_view_base``.
        """

        if len(mins) < 2:
            return "0"
        if all(_i64(m) == 0 for m in mins):
            return "0"
        row_min = mins[-2]
        row_expr: Any = row_min
        col_min = mins[-1]
        if buf is not None and len(mins) > 2 and len(buf.shape) >= len(mins):
            lead_extents = list(buf.shape[:len(mins) - 2])
            for i in range(len(mins) - 2):
                stride: Any = rows
                for d in lead_extents[i + 1:]:
                    iv = _i64(d)
                    if iv is not None and isinstance(stride, int):
                        stride *= int(iv)
                    else:
                        stride = stride * d
                row_expr = row_expr + mins[i] * stride
        assert buf is not None
        return self._hmx_tile_base_offset_from_min_c(row_expr, col_min, buf, cols // 32, rows // 32)

    def _emit_profiled_pool_call(self, indent: int, call_line: str) -> list[str]:
        if not self.hexagon_prof:
            return [self._ind(indent, call_line)]
        slot = self._prof_next_slot
        self._prof_next_slot += 1
        return [
            self._ind(indent, "tl_prof_phase_t0 = HAP_perf_get_qtimer_count();"),
            self._ind(indent, call_line),
            self._ind(indent, f"prof[{slot}] += (int32_t)(HAP_perf_get_qtimer_count() - tl_prof_phase_t0);"),
        ]

    def _emit_profiled_pool_start(self, indent: int, call_line: str) -> list[str]:
        if not self.hexagon_prof:
            self._pending_async_pool_slot = -1
            return [self._ind(indent, call_line)]
        slot = self._prof_next_slot
        self._prof_next_slot += 1
        self._pending_async_pool_slot = slot
        return [
            self._ind(indent, "tl_prof_phase_t0 = HAP_perf_get_qtimer_count();"),
            self._ind(indent, call_line),
        ]

    def _emit_profiled_pool_join(self, indent: int, slot: int) -> list[str]:
        if not self.hexagon_prof or slot < 0:
            return [self._ind(indent, "attnops_pool_join();")]
        return [
            self._ind(indent, "attnops_pool_join();"),
            self._ind(indent, f"prof[{slot}] += (int32_t)(HAP_perf_get_qtimer_count() - tl_prof_phase_t0);"),
        ]

    def _prof_wrap(self, indent: int, slot: int, lines: list[str]) -> list[str]:
        """Wrap an emitted main-thread region in an opt-in qtimer accumulator."""

        if not self.hexagon_prof:
            return lines
        return [self._ind(indent, "t0 = HAP_perf_get_qtimer_count();")] + lines + [
            self._ind(indent, f"prof[{slot}] += (int32_t)(HAP_perf_get_qtimer_count() - t0);")
        ]

    def _prof_reset(self, indent: int) -> list[str]:
        return [self._ind(indent, "t0 = HAP_perf_get_qtimer_count();")] if self.hexagon_prof else []

    def _new_hvx_tmp(self, prefix: str = "hv") -> str:
        n = getattr(self, "_hvx_tmp", 0)
        self._hvx_tmp = n + 1
        return f"{prefix}{n}"

    @staticmethod
    def _is_float_dtype(dtype: str) -> bool:
        return dtype in ("float16", "float32")

    def _expr_dtype_hint(self, e: Any) -> str | None:
        dt = getattr(e, "dtype", None)
        if dt is not None and str(dt) in ("float16", "float32"):
            return str(dt)
        if isinstance(e, BufferLoad):
            return str(e.buffer.dtype)
        cls = type(e).__name__
        if cls == "Cast":
            return str(getattr(e, "dtype", "")) or None
        return None

    def _lower_hvx_exp(self, arg: Any, dtype: str, ctx: _Ctx, lines: list[str], indent: int) -> _HVXVal:
        if dtype == "float32":
            x = self._lower_hvx_expr(arg, ctx, lines, "float32", indent)
            # fp32 vectors use the Hexagon pair convention: a 64-lane
            # T.vectorized expression is carried by two 32-lane HVX vectors.
            a, b = self._new_hvx_tmp(), self._new_hvx_tmp()
            lines.append(self._ind(indent, f"HVX_Vector {a} = hrt_exp_fp32_vec({x.names[0]});"))
            lines.append(self._ind(indent, f"HVX_Vector {b} = hrt_exp_fp32_vec({x.names[1]});"))
            return _HVXVal([a, b], "float32")
        if dtype == "float16":
            x = self._lower_hvx_expr(arg, ctx, lines, "float16", indent)
            name = self._new_hvx_tmp()
            lines.append(self._ind(indent, f"HVX_Vector {name} = hrt_exp_fp16({x.names[0]});"))
            return _HVXVal([name], "float16")
        raise HexagonEmitError(f"R7: T.exp on Hexagon only supports fp32/fp16, got {dtype}")

    def _imm_value(self, e: Any) -> float | int | None:
        iv = _i64(e)
        if iv is not None:
            return iv
        if isinstance(e, float):
            return e
        if type(e).__name__ == "FloatImm" or hasattr(e, "value") and "float" in str(getattr(e, "dtype", "")):
            try:
                return float(e.value)
            except Exception:
                return None
        return None

    @staticmethod
    def _f16_bits(v: float | int) -> int:
        return int.from_bytes(struct.pack("<e", float(v)), "little")

    @staticmethod
    def _f32_bits(v: float | int) -> int:
        return int.from_bytes(struct.pack("<f", float(v)), "little")

    def _const_hvx(self, value: float | int, dtype: str, lines: list[str], indent: int) -> _HVXVal:
        if dtype == "float32":
            name = self._new_hvx_tmp()
            bits = self._f32_bits(value)
            lines.append(self._ind(indent, f"HVX_Vector {name} = hrt_vsplat_f32_bits(0x{bits:08X});"))
            return _HVXVal([name, name], "float32")
        if dtype == "float16":
            name = self._new_hvx_tmp()
            bits = self._f16_bits(value)
            lines.append(self._ind(indent, f"HVX_Vector {name} = hrt_splat_h(0x{bits:04X});"))
            return _HVXVal([name], "float16")
        raise HexagonEmitError(f"R7: 不支持向量常量 dtype={dtype}")

    def _promote_hvx(self, v: _HVXVal, dtype: str, lines: list[str], indent: int) -> _HVXVal:
        if v.dtype == dtype:
            return v
        if v.dtype == "float16" and dtype == "float32":
            lo, hi = self._new_hvx_tmp(), self._new_hvx_tmp()
            lines.append(self._ind(indent, f"HVX_Vector {lo}, {hi};"))
            lines.append(self._ind(indent, f"hrt_h2f_vec_pair({v.names[0]}, &{lo}, &{hi});"))
            return _HVXVal([lo, hi], "float32")
        if v.dtype == "float32" and dtype == "float16":
            name = self._new_hvx_tmp()
            lines.append(self._ind(indent, f"HVX_Vector {name} = hrt_f2h_vec_pair({v.names[0]}, {v.names[1]});"))
            return _HVXVal([name], "float16")
        raise HexagonEmitError(f"R7: 不支持向量 cast {v.dtype}->{dtype}")

    def _lower_hvx_int_expr(self, e: Any, ctx: _Ctx, lines: list[str], indent: int, lane_off: int = 0) -> str:
        iv = _i64(e)
        if iv is not None:
            name = self._new_hvx_tmp("iv")
            lines.append(self._ind(indent, f"HVX_Vector {name} = Q6_V_vsplat_R({iv});"))
            return name
        if hasattr(e, "name") or hasattr(e, "name_hint"):
            vname = _var_name(e)
            name = self._new_hvx_tmp("iv")
            if ctx.vec_var is not None and vname == ctx.vec_var:
                arr = self._new_scalar_tmp("iota_w")
                vals = ", ".join(str(i) for i in range(32))
                lines.append(self._ind(indent, f"static const int32_t {arr}[32] __attribute__((aligned(128))) = {{{vals}}};"))
                base = f"({vname} + {lane_off})" if lane_off else vname
                lines.append(self._ind(indent, f"HVX_Vector {name} = Q6_Vw_vadd_VwVw(*(const HVX_Vector *){arr}, Q6_V_vsplat_R({base}));"))
            else:
                lines.append(self._ind(indent, f"HVX_Vector {name} = Q6_V_vsplat_R({vname});"))
            return name
        cls = type(e).__name__
        if cls in ("Add", "Sub"):
            a = self._lower_hvx_int_expr(e.a, ctx, lines, indent, lane_off)
            b = self._lower_hvx_int_expr(e.b, ctx, lines, indent, lane_off)
            name = self._new_hvx_tmp("iv")
            fn = "Q6_Vw_vadd_VwVw" if cls == "Add" else "Q6_Vw_vsub_VwVw"
            lines.append(self._ind(indent, f"HVX_Vector {name} = {fn}({a}, {b});"))
            return name
        if cls in ("FloorDiv", "FloorMod"):
            # 仅支持 2 的幂常数:div → 算术右移, mod → 按位与 (lane 值非负)。
            k = _i64(e.b)
            if k is None or k <= 0 or (k & (k - 1)) != 0:
                raise HexagonEmitError(f"白名单: 向量整数 {cls} 只支持 2 的幂常数, 实际 {e.b}{_loc(e)}")
            a = self._lower_hvx_int_expr(e.a, ctx, lines, indent, lane_off)
            name = self._new_hvx_tmp("iv")
            if cls == "FloorDiv":
                lines.append(self._ind(indent, f"HVX_Vector {name} = Q6_Vw_vasr_VwR({a}, {k.bit_length() - 1});"))
            else:
                m = self._new_hvx_tmp("iv")
                lines.append(self._ind(indent, f"HVX_Vector {m} = Q6_V_vsplat_R({k - 1});"))
                lines.append(self._ind(indent, f"HVX_Vector {name} = Q6_V_vand_VV({a}, {m});"))
            return name
        if cls == "Cast":
            return self._lower_hvx_int_expr(e.value, ctx, lines, indent, lane_off)
        raise HexagonEmitError(f"白名单: 向量谓词暂不支持整数表达式 {cls}{_loc(e)}")

    def _lower_hvx_pred(self, e: Any, ctx: _Ctx, lines: list[str], indent: int, lane_off: int = 0) -> str:
        cls = type(e).__name__
        if cls in ("LT", "LE", "GT", "GE", "EQ", "NE"):
            a = self._lower_hvx_int_expr(e.a, ctx, lines, indent, lane_off)
            b = self._lower_hvx_int_expr(e.b, ctx, lines, indent, lane_off)
            q = self._new_hvx_tmp("q")
            if cls == "LT":
                lines.append(self._ind(indent, f"HVX_VectorPred {q} = Q6_Q_vcmp_gt_VwVw({b}, {a});"))
            elif cls == "GT":
                lines.append(self._ind(indent, f"HVX_VectorPred {q} = Q6_Q_vcmp_gt_VwVw({a}, {b});"))
            elif cls == "LE":
                one = self._new_hvx_tmp("iv")
                bp1 = self._new_hvx_tmp("iv")
                lines.append(self._ind(indent, f"HVX_Vector {one} = Q6_V_vsplat_R(1);"))
                lines.append(self._ind(indent, f"HVX_Vector {bp1} = Q6_Vw_vadd_VwVw({b}, {one});"))
                lines.append(self._ind(indent, f"HVX_VectorPred {q} = Q6_Q_vcmp_gt_VwVw({bp1}, {a});"))
            elif cls == "GE":
                one = self._new_hvx_tmp("iv")
                ap1 = self._new_hvx_tmp("iv")
                lines.append(self._ind(indent, f"HVX_Vector {one} = Q6_V_vsplat_R(1);"))
                lines.append(self._ind(indent, f"HVX_Vector {ap1} = Q6_Vw_vadd_VwVw({a}, {one});"))
                lines.append(self._ind(indent, f"HVX_VectorPred {q} = Q6_Q_vcmp_gt_VwVw({ap1}, {b});"))
            elif cls == "EQ":
                lines.append(self._ind(indent, f"HVX_VectorPred {q} = Q6_Q_vcmp_eq_VwVw({a}, {b});"))
            else:
                eq = self._new_hvx_tmp("q")
                lines.append(self._ind(indent, f"HVX_VectorPred {eq} = Q6_Q_vcmp_eq_VwVw({a}, {b});"))
                lines.append(self._ind(indent, f"HVX_VectorPred {q} = Q6_Q_not_Q({eq});"))
            return q
        raise HexagonEmitError(f"白名单: 向量谓词暂不支持 {cls}{_loc(e)}")

    def _lower_hvx_expr(self, e: Any, ctx: _Ctx, lines: list[str], want: str | None, indent: int) -> _HVXVal:
        imm = self._imm_value(e)
        if imm is not None:
            return self._const_hvx(imm, want or self._expr_dtype_hint(e) or "float16", lines, indent)
        if isinstance(e, BufferLoad):
            if _buffer_scope(e.buffer) != "global" and not _is_vtcm(_buffer_scope(e.buffer)) and not _is_wscratch(_buffer_scope(e.buffer)):
                raise HexagonEmitError(f"R2: 当前向量表达式 load 只支持 global/vtcm/wscratch, 实际 {_buffer_scope(e.buffer)}{_loc(e)}")
            dtype = str(e.buffer.dtype)
            if dtype not in ("float16", "float32"):
                raise HexagonEmitError(f"R7: 当前向量 load 只支持 fp16/fp32, 实际 {e.buffer.dtype}{_loc(e)}")
            base = self._vector_index_c(e.indices, ctx, e.buffer)
            ptr = self._buffer_ptr_c(e.buffer)
            if ctx.vec_var is not None and ctx.vec_var not in base:
                if _is_vtcm(_buffer_scope(e.buffer)):
                    raise HexagonEmitError(f"R2: 标量广播禁止从 VTCM buffer {_buffer_name(e.buffer)} 标量 load{_loc(e)}")
                name = self._new_hvx_tmp()
                if dtype == "float32":
                    lines.append(self._ind(indent, f"HVX_Vector {name} = hrt_vsplat_f32({ptr}[(size_t)({base})]);"))
                    v = _HVXVal([name, name], "float32")
                else:
                    lines.append(self._ind(indent, f"HVX_Vector {name} = hrt_splat_h((int){ptr}[(size_t)({base})]);"))
                    v = _HVXVal([name], "float16")
                return self._promote_hvx(v, want, lines, indent) if want else v
            self._check_vector_alignment(e.indices, ctx, e, e.buffer, base)
            name = self._new_hvx_tmp()
            cty = "const HVX_Vector *"
            if dtype == "float32" and ctx.vec_extent is not None and ctx.vec_extent != 32:
                name_hi = self._new_hvx_tmp()
                lines.append(self._ind(indent, f"HVX_Vector {name} = (*({cty})({ptr} + (size_t)({base})));"))
                lines.append(self._ind(indent, f"HVX_Vector {name_hi} = (*({cty})({ptr} + (size_t)({base}) + 32));"))
                v = _HVXVal([name, name_hi], dtype)
                return self._promote_hvx(v, want, lines, indent) if want else v
            lines.append(self._ind(indent, f"HVX_Vector {name} = (*({cty})({ptr} + (size_t)({base})));"))
            v = _HVXVal([name, name] if dtype == "float32" else [name], dtype)
            return self._promote_hvx(v, want, lines, indent) if want else v
        if hasattr(e, "name") or hasattr(e, "name_hint"):
            name = _var_name(e)
            dtype = want or self.scalar_vars.get(name, "float32")
            hv = self._new_hvx_tmp()
            if dtype == "float32":
                lines.append(self._ind(indent, f"HVX_Vector {hv} = hrt_vsplat_f32({name});"))
                return _HVXVal([hv, hv], "float32")
            lines.append(self._ind(indent, f"HVX_Vector {hv} = hrt_splat_h((int){name});"))
            return _HVXVal([hv], "float16")
        if isinstance(e, Call):
            opname = _call_op_name(e)
            callee = None
            arg0 = 0
            if opname in ("tir.call_pure_extern", "tirx.call_pure_extern") and len(e.args) >= 1:
                callee = _ann_str(e.args[0]); arg0 = 1
            if opname in ("tir.if_then_else", "tirx.if_then_else") or str(opname).endswith("if_then_else"):
                if len(e.args) < 3:
                    raise HexagonEmitError(f"白名单: if_then_else 参数数量不足{_loc(e)}")
                dtype = want or self._expr_dtype_hint(e.args[1]) or self._expr_dtype_hint(e.args[2]) or "float16"
                tv = self._lower_hvx_expr(e.args[1], ctx, lines, dtype, indent)
                fv = self._lower_hvx_expr(e.args[2], ctx, lines, dtype, indent)
                if dtype == "float16":
                    pred = self._lower_hvx_pred(e.args[0], ctx, lines, indent)
                    name = self._new_hvx_tmp()
                    lines.append(self._ind(indent, f"HVX_Vector {name} = Q6_V_vmux_QVV({pred}, {tv.names[0]}, {fv.names[0]});"))
                    return _HVXVal([name], "float16")
                if dtype == "float32":
                    # fp32 每 32 lane 一个寄存器;64-lane pair 的上半必须用
                    # 自己的谓词(lane 下标 +32)。[name, name] 伪 pair 只算一次。
                    tn = tv.names if not (len(tv.names) == 2 and tv.names[0] == tv.names[1]) else tv.names[:1]
                    fn = fv.names if not (len(fv.names) == 2 and fv.names[0] == fv.names[1]) else fv.names[:1]
                    n = max(len(tn), len(fn))
                    out: list[str] = []
                    for i in range(n):
                        pred_i = self._lower_hvx_pred(e.args[0], ctx, lines, indent, lane_off=32 * i)
                        ti = tn[i] if i < len(tn) else tn[-1]
                        fi = fn[i] if i < len(fn) else fn[-1]
                        name = self._new_hvx_tmp()
                        lines.append(self._ind(indent, f"HVX_Vector {name} = Q6_V_vmux_QVV({pred_i}, {ti}, {fi});"))
                        out.append(name)
                    return _HVXVal(out, "float32")
            if callee == "hexagon.silu_fp16":
                x = self._lower_hvx_expr(e.args[arg0], ctx, lines, "float16", indent)
                one = self._const_hvx(1.0, "float16", lines, indent)
                name = self._new_hvx_tmp()
                lines.append(self._ind(indent, f"HVX_Vector {name} = hrt_silu_fp16_vec({x.names[0]}, {one.names[0]});"))
                return _HVXVal([name], "float16")
            if callee == "hexagon.exp_fp16":
                x = self._lower_hvx_expr(e.args[arg0], ctx, lines, "float16", indent)
                name = self._new_hvx_tmp()
                lines.append(self._ind(indent, f"HVX_Vector {name} = hrt_exp_fp16({x.names[0]});"))
                return _HVXVal([name], "float16")
            if callee == "hexagon.exp_fp32":
                return self._lower_hvx_exp(e.args[arg0], "float32", ctx, lines, indent)
            if opname in ("tir.exp", "tirx.exp", "exp") or str(opname).endswith(".exp"):
                dtype = want or self._expr_dtype_hint(e.args[0]) or "float16"
                return self._lower_hvx_exp(e.args[0], dtype, ctx, lines, indent)
            raise HexagonEmitError(f"白名单: 向量表达式暂不支持 Call {_call_op_name(e)}{_loc(e)}")
        cls = type(e).__name__
        if cls == "Cast":
            dst = str(getattr(e, "dtype", ""))
            if dst not in ("float16", "float32"):
                raise HexagonEmitError(f"R7: 不支持向量 Cast 到 {dst}{_loc(e)}")
            src = self._lower_hvx_expr(e.value, ctx, lines, None, indent)
            return self._promote_hvx(src, dst, lines, indent)
        if cls in ("Add", "Sub", "Mul", "Div", "Min", "Max"):
            ah = self._expr_dtype_hint(e.a)
            bh = self._expr_dtype_hint(e.b)
            dtype = "float32" if "float32" in (ah, bh, want) else "float16"
            a = self._lower_hvx_expr(e.a, ctx, lines, dtype, indent)
            b = self._lower_hvx_expr(e.b, ctx, lines, dtype, indent)
            if dtype == "float16":
                fn = {"Add": "Q6_Vhf_vadd_VhfVhf", "Sub": "Q6_Vhf_vsub_VhfVhf",
                      "Mul": "Q6_Vhf_vmpy_VhfVhf", "Div": "hrt_vhf_div_approx",
                      "Min": "Q6_Vhf_vfmin_VhfVhf", "Max": "Q6_Vhf_vfmax_VhfVhf"}[cls]
                name = self._new_hvx_tmp()
                lines.append(self._ind(indent, f"HVX_Vector {name} = {fn}({a.names[0]}, {b.names[0]});"))
                return _HVXVal([name], "float16")
            fn = {"Add": "Q6_Vsf_vadd_VsfVsf", "Sub": "Q6_Vsf_vsub_VsfVsf",
                  "Mul": "Q6_Vsf_vmpy_VsfVsf", "Div": "hrt_vsf_div_approx",
                  "Min": "Q6_Vsf_vmin_VsfVsf", "Max": "Q6_Vsf_vmax_VsfVsf"}[cls]
            n0, n1 = self._new_hvx_tmp(), self._new_hvx_tmp()
            lines.append(self._ind(indent, f"HVX_Vector {n0} = {fn}({a.names[0]}, {b.names[0]});"))
            lines.append(self._ind(indent, f"HVX_Vector {n1} = {fn}({a.names[1]}, {b.names[1]});"))
            return _HVXVal([n0, n1], "float32")
        if cls in ("Neg",):
            x = self._lower_hvx_expr(e.value, ctx, lines, want or self._expr_dtype_hint(e.value) or "float16", indent)
            if x.dtype == "float16":
                name = self._new_hvx_tmp()
                lines.append(self._ind(indent, f"HVX_Vector {name} = Q6_Vhf_vsub_VhfVhf(Q6_V_vzero(), {x.names[0]});"))
                return _HVXVal([name], "float16")
            n0, n1 = self._new_hvx_tmp(), self._new_hvx_tmp()
            lines.append(self._ind(indent, f"HVX_Vector {n0} = Q6_Vsf_vsub_VsfVsf(Q6_V_vzero(), {x.names[0]});"))
            lines.append(self._ind(indent, f"HVX_Vector {n1} = Q6_Vsf_vsub_VsfVsf(Q6_V_vzero(), {x.names[1]});"))
            return _HVXVal([n0, n1], "float32")
        raise HexagonEmitError(f"白名单: 向量表达式暂不支持 {cls}{_loc(e)}")

    def _emit_gemm_intrin_with_copy(self, gemm: Call, copy: Call, ctx: _Ctx, indent: int) -> list[str]:
        self._visit_gemm_intrin(gemm)
        self._visit_copy_intrin(copy)
        if self._generic_gemm_recipe_mode():
            return self._emit_gemm_recipe_generic(gemm, copy, ctx, indent)
        return self._emit_gemm_recipe(ctx, indent)

    def _emit_gemm_recipe_generic(self, gemm: Call, copy: Call, ctx: _Ctx, indent: int) -> list[str]:
        if len(gemm.args) < 7:
            raise HexagonEmitError("R5: hexagon.gemm_hmx 参数数量不足")
        # Trailing args: M, N, K, and (since the transB flag was added) an
        # optional 0/1 transpose_B marker.  Legacy 11-arg form (with callee)
        # has no flag and means transpose_B=True, the only historically
        # exercised convention.
        if len(gemm.args) >= 12:
            m, n, k = (_i64(gemm.args[-4]), _i64(gemm.args[-3]), _i64(gemm.args[-2]))
            trans_b = bool(_i64(gemm.args[-1]) or 0)
        else:
            m, n, k = (_i64(gemm.args[-3]), _i64(gemm.args[-2]), _i64(gemm.args[-1]))
            trans_b = True
        if None in (m, n, k):
            raise HexagonEmitError("R5: 通用 hexagon.gemm_hmx 的 M/N/K 必须是静态整数")
        assert m is not None and n is not None and k is not None
        if m % 32 or n % 32 or k % 32:
            raise HexagonEmitError(f"R5: hexagon.gemm_hmx tile 维度必须 32 整除, 实际 M={m},N={n},K={k}")
        a_buf = self._buffer_from_data_arg(gemm.args[1])
        b_buf = self._buffer_from_data_arg(gemm.args[2])
        if a_buf is None or b_buf is None:
            raise HexagonEmitError(f"R5: 通用 gemm 找不到 A/B VTCM buffer{_loc(gemm)}")
        if not _is_vtcm(_buffer_scope(a_buf)) or not _is_vtcm(_buffer_scope(b_buf)):
            raise HexagonEmitError(f"R5: 通用 gemm 要求 A/B 已 stage 到 VTCM AH/WH buffer{_loc(gemm)}")
        mt, nt, kt = m // 32, n // 32, k // 32
        a_base = self._vtcm_offset_c(a_buf)
        b_base = self._vtcm_offset_c(b_buf)
        if len(gemm.args) >= 11:
            a_view = self._hmx_tile_base_offset_from_min_c(gemm.args[4], gemm.args[5], a_buf, kt, mt)
            b_view = self._hmx_tile_base_offset_from_min_c(gemm.args[6], gemm.args[7], b_buf, kt, nt)
        else:
            a_view = self._hmx_tile_base_offset_c(gemm.args[1], a_buf)
            b_view = self._hmx_tile_base_offset_c(gemm.args[2], b_buf)
        a_off = a_base if a_view == "0" else f"{a_base} + {a_view}"
        b_off = b_base if b_view == "0" else f"{b_base} + {b_view}"
        if len(copy.args) < 4:
            raise HexagonEmitError("copy_acc_rm 参数数量不足")
        dst = self._buffer_from_data_arg(copy.args[2])
        if dst is None:
            raise HexagonEmitError(f"copy_acc_rm: 找不到目标 buffer{_loc(copy)}")
        dnd_start = 4 + 2 * int(_i64(copy.args[3]) or 0)
        dnd = int(_i64(copy.args[dnd_start]) or 0) if len(copy.args) > dnd_start else 0
        dst_base = self._copy_region_linear_base(copy, dnd_start + 1, dnd, dst)
        dst_scope = _buffer_scope(dst)
        dst_is_vtcm = _is_vtcm(dst_scope)
        dst_stride_i = _i64(dst.shape[-1]) if len(dst.shape) >= 1 else None
        dst_stride = int(dst_stride_i) if dst_stride_i is not None else n
        if str(dst.dtype) != "float16":
            raise HexagonEmitError(f"copy_acc_rm: 通用 gemm 写回只支持 fp16, 实际 {dst.dtype}")
        acc0 = "TL_ACC"
        prefix = self._new_scalar_tmp("gm")
        # HMX has no non-transposed B mode: the WH operand always encodes an
        # [N, K] source (C = A @ B^T).  transpose_B=False would require staging
        # B^T from an [K, N] row-major source (transposed reads), which the
        # rm->WH recipe does not implement; reject it explicitly instead of
        # emitting silently wrong code.  Kernel-side remedy: materialize B^T
        # in a scratch buffer and use transpose_B=True.
        if not trans_b:
            raise HexagonEmitError(
                "R5: hexagon.gemm_hmx 暂不支持 transpose_B=False: HMX WH 操作数恒编码 [N,K] 源; "
                "请先在 scratch 里显式转置 B 再使用 transpose_B=True")
        b_tile = f"(size_t){prefix}_c * {kt} + {prefix}_kb"
        lines: list[str] = [
            *self._prof_reset(indent),
            self._ind(indent, f"for (int {prefix}_r = 0; {prefix}_r < {mt}; {prefix}_r++) {{"),
            self._ind(indent + 1, f"for (int {prefix}_c = 0; {prefix}_c < {nt}; {prefix}_c++) {{"),
            self._ind(indent + 2, "if (!(abl & 4)) {"),
            self._ind(indent + 3, "int e = hrt_acc_clear_f16();"),
            self._ind(indent + 3, "if (e) return e;"),
            self._ind(indent + 3, f"for (int {prefix}_kb = 0; {prefix}_kb < {kt}; {prefix}_kb++) {{"),
            self._ind(indent + 4, f"e = hrt_hmx_mm_f16(V, {a_off} + ((size_t){prefix}_r * {kt} + {prefix}_kb) * HRT_TILE_BYTES,"),
            self._ind(indent + 5, f"{b_off} + ({b_tile}) * HRT_TILE_BYTES);"),
            self._ind(indent + 4, "if (e) return e;"),
            self._ind(indent + 3, "}"),
            self._ind(indent + 3, f"if (!(abl & 16)) {{ e = hrt_acc_read_f16(V, g_v.CFG, {acc0}); if (e) return e; }}"),
            self._ind(indent + 2, "}"),
            *( [self._ind(indent + 2, "prof[2] += (int32_t)(HAP_perf_get_qtimer_count() - t0);"),
                self._ind(indent + 2, "t0 = HAP_perf_get_qtimer_count();")] if self.hexagon_prof else [] ),
            self._ind(indent + 2, "if (!(abl & 8)) {"),
        ]
        if dst_is_vtcm:
            dst_off = self._vtcm_offset_c(dst)
            lines += [
                self._ind(indent + 3, f"hrt_tlgdn_acc_tile_to_vtcm_rm(V, {acc0}, (f16 *)(V + {dst_off}), {dst_stride}, {prefix}_r, {prefix}_c);"),
            ]
        elif dst_scope == "global":
            dst_ptr = self._buffer_ptr_c(dst)
            lines += [
                self._ind(indent + 3, "hrt_mask_init();"),
                self._ind(indent + 3, f"hrt_unperm_single_hvx(V + {acc0}, {dst_ptr} + (size_t)({dst_base}) + (size_t){prefix}_r * 32 * {dst_stride} + {prefix}_c * 32, {dst_stride});"),
            ]
        else:
            raise HexagonEmitError(f"copy_acc_rm: 目标必须是 global 或 VTCM RM, 实际 scope={dst_scope}")
        lines += [
            self._ind(indent + 2, "}"),
            *( [self._ind(indent + 2, "prof[3] += (int32_t)(HAP_perf_get_qtimer_count() - t0);"),
                self._ind(indent + 2, "t0 = HAP_perf_get_qtimer_count();")] if self.hexagon_prof else [] ),
            self._ind(indent + 1, "}"),
            self._ind(indent, "}"),
        ]
        return lines

    def _emit_gemm_recipe(self, ctx: _Ctx, indent: int) -> list[str]:
        rv = ctx.row_var or "0"
        bv = ctx.block_var or "bx"
        if not self._legacy_gemm_mode():
            bm = self.block_M_hint or 32
            mbt_expr = "mbt" if ctx.row_var is not None else str(bm // 32)
            rr_open = [] if bm == 32 else [self._ind(indent, f"for (int rb = 0; rb < {mbt_expr}; rb++) {{")]
            rr_close = [] if bm == 32 else [self._ind(indent, "}")]
            rr = "0" if bm == 32 else "rb"
            ii = indent if bm == 32 else indent + 1
            return [
                *rr_open,
                self._ind(ii, "t0 = HAP_perf_get_qtimer_count();"),
                self._ind(ii, "for (int c2 = 0; c2 < nct; c2++) {"),
                self._ind(ii + 1, "if (!(abl & 4)) {"),
                self._ind(ii + 2, "int e = hrt_acc_clear_f16();"),
                self._ind(ii + 2, "if (e) return e;"),
                self._ind(ii + 2, "for (int kb = 0; kb < kt; kb++) {"),
                self._ind(ii + 3, f"e = hrt_hmx_mm_f16(V, GM_ACT + ((size_t){rr} * kt + kb) * HRT_TILE_BYTES,"),
                self._ind(ii + 4, "GM_WA + ((size_t)c2 * kt + kb) * HRT_TILE_BYTES);"),
                self._ind(ii + 3, "if (e) return e;"),
                self._ind(ii + 2, "}"),
                self._ind(ii + 2, "if (!(abl & 16)) {"),
                self._ind(ii + 3, "e = hrt_acc_read_f16(V, g_v.CFG, GM_OUT + (size_t)(c2 & 1) * HRT_TILE_BYTES);"),
                self._ind(ii + 3, "if (e) return e;"),
                self._ind(ii + 2, "}"),
                self._ind(ii + 1, "}"),
                self._ind(ii + 1, "prof[2] += (int32_t)(HAP_perf_get_qtimer_count() - t0);"),
                self._ind(ii + 1, "t0 = HAP_perf_get_qtimer_count();"),
                self._ind(ii + 1, "if (!(abl & 8) && (c2 & 1)) {"),
                self._ind(ii + 2, f"int ncol = {bv} * NP + (c2 - 1) * 32;"),
                self._ind(ii + 2, "hrt_mask_init();"),
                self._ind(ii + 2, f"hrt_unperm_pair_hvx(V + GM_OUT, V + GM_OUT + HRT_TILE_BYTES, C + ((size_t){rv} * GM_BLOCK_M + {rr} * 32) * N + ncol, N);"),
                self._ind(ii + 1, "}"),
                self._ind(ii, "}"),
                self._ind(ii, "if (!(abl & 8) && (nct & 1)) {"),
                self._ind(ii + 1, f"int ncol = {bv} * NP + (nct - 1) * 32;"),
                self._ind(ii + 1, "hrt_mask_init();"),
                self._ind(ii + 1, f"hrt_unperm_single_hvx(V + GM_OUT, C + ((size_t){rv} * GM_BLOCK_M + {rr} * 32) * N + ncol, N);"),
                self._ind(ii, "}"),
                self._ind(ii, "prof[3] += (int32_t)(HAP_perf_get_qtimer_count() - t0);"),
                *rr_close,
            ]
        return [
            self._ind(indent, "t0 = HAP_perf_get_qtimer_count();"),
            self._ind(indent, "for (int c2 = 0; c2 < nct; c2++) {"),
            self._ind(indent + 1, "if (!(abl & 4)) {"),
            self._ind(indent + 2, "int e = hrt_acc_clear_f16();"),
            self._ind(indent + 2, "if (e) return e;"),
            self._ind(indent + 2, "for (int kb = 0; kb < kt; kb++) {"),
            self._ind(indent + 3, "e = hrt_hmx_mm_f16(V, GM_ACT + (size_t)kb * HRT_TILE_BYTES,"),
            self._ind(indent + 4, "GM_WA + ((size_t)c2 * kt + kb) * HRT_TILE_BYTES);"),
            self._ind(indent + 3, "if (e) return e;"),
            self._ind(indent + 2, "}"),
            self._ind(indent + 2, "if (!(abl & 16)) {"),
            self._ind(indent + 3, "e = hrt_acc_read_f16(V, g_v.CFG, GM_OUT + (size_t)(c2 & 1) * HRT_TILE_BYTES);"),
            self._ind(indent + 3, "if (e) return e;"),
            self._ind(indent + 2, "}"),
            self._ind(indent + 1, "}"),
            self._ind(indent + 1, "prof[2] += (int32_t)(HAP_perf_get_qtimer_count() - t0);"),
            self._ind(indent + 1, "t0 = HAP_perf_get_qtimer_count();"),
            self._ind(indent + 1, "if (!(abl & 8) && (c2 & 1)) {"),
            self._ind(indent + 2, f"int ncol = {bv} * NP + (c2 - 1) * 32;"),
            self._ind(indent + 2, "hrt_mask_init();"),
            self._ind(indent + 2, f"hrt_unperm_pair_hvx(V + GM_OUT, V + GM_OUT + HRT_TILE_BYTES, C + (size_t)({rv} * 32) * N + ncol, N);"),
            self._ind(indent + 1, "}"),
            self._ind(indent, "}"),
            self._ind(indent, "prof[3] += (int32_t)(HAP_perf_get_qtimer_count() - t0);"),
        ]

    # ---- collection / verification ----------------------------------------

    def _record_params(self, func: PrimFunc) -> None:
        sym = func.attrs.get("global_symbol") if func.attrs else None
        if sym and str(sym) != "main":
            self.func_name = str(sym)
        elif self.func_name is None:
            self.func_name = self._derive_func_name(func)
        for _, buf in func.buffer_map.items():
            self.buffers[_buffer_name(buf)] = buf
        self._record_stmt_buffers(func.body)
        attrs = func.attrs or {}
        self.hexagon_prof = _bool_arg(attrs.get("tl.hexagon_prof", False))
        self._prof_next_slot = 5
        self._written_globals = {str(x) for x in (attrs.get("hexagon.write_set") or [])}
        raw_offsets = attrs.get("hexagon.vtcm_offsets") or {}
        self._vtcm_offsets = {str(k): int(v.value if hasattr(v, "value") else v) for k, v in raw_offsets.items()}
        raw_wscratch = attrs.get("hexagon.wscratch_slots")
        if raw_wscratch is None:
            raw_wscratch = ""
        raw_wscratch_s = str(raw_wscratch.value if hasattr(raw_wscratch, "value") else raw_wscratch)
        self._wscratch_slots = {}
        if raw_wscratch_s:
            for item in raw_wscratch_s.split(","):
                if not item:
                    continue
                name, field = item.split(":", 1)
                self._wscratch_slots[name] = field
        self._derive_global_args(func)
        self._record_gemm_operands(func.body)

    def _record_stmt_buffers(self, op: Any) -> None:
        def visit(node: Any) -> None:
            if isinstance(node, SBlockRealize):
                for buf in node.block.alloc_buffers:
                    self.buffers[_buffer_name(buf)] = buf
            elif isinstance(node, SBlock):
                for buf in node.alloc_buffers:
                    self.buffers[_buffer_name(buf)] = buf

        try:
            tirx.stmt_functor.post_order_visit(op, visit)
        except Exception:
            pass

    def _derive_func_name(self, func: PrimFunc) -> str:
        attrs = func.attrs or {}
        for key in ("tir.noalias", "calling_conv"):
            _ = attrs.get(key)
        name = getattr(func, "name", None) or getattr(func, "name_hint", None)
        if name and str(name) != "main":
            return str(name)
        raise HexagonEmitError("Hexagon emitter requires PrimFunc global_symbol (not 'main') for skel ABI entry")

    def _derive_global_args(self, func: PrimFunc) -> None:
        args: list[_GlobalArg] = []
        used: set[str] = set()
        for _, buf in func.buffer_map.items():
            if _buffer_scope(buf) != "global":
                continue
            bname = _buffer_name(buf)
            cname = self._c_var_name(bname, used)
            args.append(_GlobalArg(buf=buf, cname=cname))
            self.global_ptrs[bname] = cname
        self.global_args = args

    def _derive_shape_params(self, func: PrimFunc) -> list[str]:
        params: list[str] = []
        seen: set[str] = set()
        for _, buf in func.buffer_map.items():
            for dim in buf.shape:
                if _i64(dim) is not None:
                    continue
                name = self._expr_c(dim)
                if name not in seen:
                    seen.add(name)
                    params.append(name)
        return params

    @staticmethod
    def _c_var_name(name: str, used: set[str]) -> str:
        aliases = {
            "Q": "q", "K": "k", "V": "v", "G": "g", "B": "b", "S0": "s0", "O": "o", "S1": "s1",
        }
        raw = aliases.get(name, name)
        out = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in raw)
        if not out or out[0].isdigit():
            out = f"buf_{out}"
        base = out
        i = 1
        while out in used:
            out = f"{base}_{i}"
            i += 1
        used.add(out)
        return out

    def _global_slab_args(self, weight: Buffer | None = None) -> list[_GlobalArg]:
        wname = _buffer_name(weight) if weight is not None else None
        return [arg for arg in self.global_args if _buffer_name(arg.buf) != wname]

    def _global_numel_expr(self, buf: Buffer, shape_params: dict[str, str]) -> str:
        terms: list[str] = []
        for dim in buf.shape:
            iv = _i64(dim)
            if iv is None:
                terms.append(self._expr_c(dim))
            else:
                terms.append(shape_params.get(str(iv), str(iv)))
        return " * ".join(f"(size_t){x}" for x in terms) if terms else "1"

    def _emit_slab_layout(self, indent: int, shape_params: dict[str, str], weight: Buffer | None = None,
                          prof_bytes: int = 4) -> tuple[list[str], str]:
        lines = [self._ind(indent, "size_t off = 0;")]
        for arg in self._global_slab_args(weight):
            buf = arg.buf
            cname = arg.cname
            ctype = self._buffer_elem_ctype(buf)
            qual = "const " if self._is_readonly_global(buf) else ""
            lines.append(self._ind(indent, f"{qual}{ctype} *{cname} = ({qual}{ctype} *)(slab + off);"))
            n = self._global_numel_expr(buf, shape_params)
            sz = _dtype_bytes(str(buf.dtype))
            lines.append(self._ind(indent, f"off = HRT_ALIGN128(off + ({n}) * {sz});"))
        lines.append(self._ind(indent, "size_t prof_off = off;"))
        lines.append(self._ind(indent, f"if ((size_t)slabLen < prof_off + {prof_bytes}) return -1;"))
        return lines, "prof_off"

    def _is_readonly_global(self, buf: Buffer) -> bool:
        # out_idx buffers are not preserved in lowered PrimFunc attrs, so infer
        # mutability from the examples' TIR use: buffers written by BufferStore are
        # outputs, all other global params are const slab inputs.
        name = _buffer_name(buf)
        return name not in getattr(self, "_written_globals", set())

    def _record_alloc(self, buf: Buffer) -> None:
        scope = _buffer_scope(buf)
        self.buffers[_buffer_name(buf)] = buf
        if _is_acc(scope):
            for dim in buf.shape:
                iv = _i64(dim)
                if iv is not None and iv % 32:
                    raise HexagonEmitError(f"R5: hmx.acc buffer {_buffer_name(buf)} 维度 {iv} 不是 32 的倍数")
            shape = [_i64(d) for d in buf.shape]
            if len(shape) >= 2 and shape[0] and shape[1]:
                self.block_M_hint = int(shape[0])
                self.block_N_hint = int(shape[1])
            numel = _shape_numel(buf.shape)
            nbytes = _align(2 * GM_TILE if self._generic_gemm_recipe_mode() else (numel or 0) * 2)
            name = _buffer_name(buf)
            self.allocs.append(_Alloc(name, scope, nbytes if (self._generic_gemm_recipe_mode() or not self._legacy_gemm_mode()) else 0, self._vtcm_offsets.get(name)))
            return
        if _is_vtcm(scope):
            numel = _shape_numel(buf.shape)
            if numel is None:
                raise HexagonEmitError(f"R3: VTCM buffer {_buffer_name(buf)} shape 必须静态")
            nbytes = _align(numel * _dtype_bytes(str(buf.dtype)))
            name = _buffer_name(buf)
            self.allocs.append(_Alloc(name, scope, nbytes, self._vtcm_offsets.get(name)))
            return
        if _is_wscratch(scope):
            if _buffer_name(buf) not in self._wscratch_slots:
                raise HexagonEmitError(f"wscratch: buffer {_buffer_name(buf)} 缺少 slot 映射")
            return

    def _finalize_allocs(self) -> None:
        if self._uses_gdn_shell():
            off = self._assign_vtcm_offsets()
            if self.gdn_mode == "leaf":
                off *= (self.nworkers or 6)
            if off > VTCM_BUDGET:
                details = [f"{a.name}:{a.bytes}" for a in self.allocs if a.bytes]
                raise HexagonEmitError(f"R3: GDN VTCM 静态预算超限 {off} > 8MB; 明细: {', '.join(details)}")
            return
        if self.has_vector_kernel or self.has_scalar_kernel:
            off = self._assign_vtcm_offsets()
            if off > VTCM_BUDGET:
                details = [f"{a.name}:{a.bytes}" for a in self.allocs if a.bytes]
                raise HexagonEmitError(f"R3: VTCM 静态预算超限 {off} > 8MB; 明细: {', '.join(details)}")
            return
        if not self._legacy_gemm_mode():
            off = self._assign_vtcm_offsets()
            if off > VTCM_BUDGET:
                details = [f"{a.name}:{a.bytes}" for a in self.allocs if a.bytes]
                raise HexagonEmitError(f"R3: VTCM 静态预算超限 {off} > 8MB; 明细: {', '.join(details)}")
            return
        m, _, k = self._infer_problem_shape()
        bm = self.block_M_hint or 32
        if bm != 32:
            raise HexagonEmitError(f"R5: v2 GEMM_NT 目前 block_M 必须为 32, 实际 {bm}")
        gm_kmax = max(k, 4096)
        gm_act = gm_kmax * 64
        gm_wa = gm_act * 2
        gm_out = gm_wa + GM_WA_MAX
        gm_end = gm_out + 2 * GM_TILE
        if gm_end > VTCM_BUDGET:
            raise HexagonEmitError(f"R3: 生成物 VTCM map 超 8MB: GM_END={gm_end}")
        # Also report user alloc accounting; generated map can contain compiler
        # temporaries (LIN/OUT) in addition to alloc_shared/fragment.
        off = 0
        details = []
        for alloc in self.allocs:
            if alloc.bytes == 0:
                continue
            off = _align(off)
            alloc.offset = off
            details.append(f"{alloc.name}:{alloc.bytes}")
            off += alloc.bytes
        if off > VTCM_BUDGET:
            raise HexagonEmitError(f"R3: VTCM 静态预算超限 {off} > 8MB; 明细: {', '.join(details)}")

    def _visit_gemm_intrin(self, call: Call) -> None:
        self.has_gemm = True
        if len(call.args) < 7:
            raise HexagonEmitError("R5: hexagon.gemm_hmx 参数数量不足")
        if len(call.args) >= 12:
            m, n, k = (_i64(call.args[-4]), _i64(call.args[-3]), _i64(call.args[-2]))
        else:
            m, n, k = (_i64(call.args[-3]), _i64(call.args[-2]), _i64(call.args[-1]))
        if None in (m, n, k):
            raise HexagonEmitError("R5: hexagon.gemm_hmx 的 M/N/K 必须是静态整数")
        assert m is not None and n is not None and k is not None
        if m % 32 or n % 32 or k % 32:
            raise HexagonEmitError(f"R5: hexagon.gemm_hmx tile 维度必须 32 整除, 实际 M={m},N={n},K={k}")
        self.block_M_hint = int(m)
        self.block_N_hint = int(n)
        if self.gemm_mnk is None:
            self.gemm_mnk = (m, n, k)
        self._record_gemm_b_arg(call)

    def _legacy_gemm_mode(self) -> bool:
        """Keep the original Qwen GEMM_NT generation byte-for-byte."""

        bm = self.block_M_hint or 32
        bn = self.block_N_hint or 1024
        return bm == 32 and bn == 1024

    def _generic_gemm_recipe_mode(self) -> bool:
        """Use per-call GEMM lowering when HMX appears inside a non-GEMM shell.

        The legacy GEMM_NT anchor (block_M=32, block_N=1024, no view operands)
        always keeps the legacy path so its emitted ABI stays byte-identical.
        """

        if not self._pre_has_gemm:
            return False
        if self._legacy_gemm_mode() and not self._pre_has_gemm_view:
            return False
        return bool(self._pre_has_gemm_view or self._pre_has_vector or self._pre_has_parallel or self.has_scalar_kernel or self._uses_gdn_shell())

    def _stmt_contains_gemm_intrin(self, op: Any) -> bool:
        if isinstance(op, Evaluate) and isinstance(op.value, Call):
            return self._extern_callee_name(op.value) == "hexagon.gemm_hmx"
        if isinstance(op, For):
            return self._stmt_contains_gemm_intrin(op.body)
        if isinstance(op, AttrStmt):
            return self._stmt_contains_gemm_intrin(op.body)
        if isinstance(op, (SBlockRealize, SBlock)):
            return self._stmt_contains_gemm_intrin(op.block if isinstance(op, SBlockRealize) else op.body)
        if isinstance(op, SeqStmt):
            return any(self._stmt_contains_gemm_intrin(x) for x in op.seq)
        if isinstance(op, IfThenElse):
            return self._stmt_contains_gemm_intrin(op.then_case) or (op.else_case is not None and self._stmt_contains_gemm_intrin(op.else_case))
        return False

    def _record_gemm_operands(self, op: Any) -> None:
        def visit(node: Any) -> None:
            if isinstance(node, Call) and self._extern_callee_name(node) == "hexagon.gemm_hmx":
                # 11-arg form carries (A_base0, A_base1, B_base0, B_base1).
                # Only a genuinely non-zero/non-constant base means a view
                # operand; the plain 7-arg-equivalent form (all bases zero)
                # must NOT flip legacy GEMM kernels into the generic path.
                if len(node.args) >= 11:
                    bases = node.args[4:8]
                    def _is_zero(x: Any) -> bool:
                        v = _i64(x)
                        return v == 0
                    if not all(_is_zero(b) for b in bases):
                        self._pre_has_gemm_view = True
                self._record_gemm_b_arg(node)

        try:
            tirx.stmt_functor.post_order_visit(op, visit)
        except Exception:
            pass

    def _record_gemm_b_arg(self, call: Call) -> None:
        callee_i = 0 if call.args and _ann_str(call.args[0]) == "hexagon.gemm_hmx" else -1
        b_idx = callee_i + 2
        if b_idx < 0 or len(call.args) <= b_idx:
            return
        buf = self._buffer_from_data_arg(call.args[b_idx])
        if buf is not None:
            self._gemm_b_buffers.add(_var_name(getattr(buf, "data", "")))
            self._gemm_b_buffers.add(_buffer_name(buf))

    def _is_gemm_b_operand(self, buf: Buffer | None) -> bool:
        if buf is None:
            return False
        return (_var_name(getattr(buf, "data", "")) in self._gemm_b_buffers
                or _buffer_name(buf) in self._gemm_b_buffers)

    def _uses_gdn_shell(self) -> bool:
        return bool(self._wscratch_slots) or self.gdn_mode == "escape"

    def _visit_copy_intrin(self, call: Call) -> None:
        self.has_copy = True
        callee = _ann_str(call.args[0]) if call.args else None
        if callee in ("hexagon.copy_ah_rm", "hexagon.copy_acc_rm"):
            self.has_unperm = True

    # ---- predicates / expression lowering ---------------------------------

    def _is_gemm_intrin_eval(self, op: Any) -> bool:
        call = self._eval_call(op)
        return call is not None and self._extern_callee_name(call) == "hexagon.gemm_hmx"

    def _is_acc_copy_intrin_eval(self, op: Any) -> bool:
        call = self._eval_call(op)
        return call is not None and self._extern_callee_name(call) in ("hexagon.copy_acc_rm", "hexagon.copy_ah_rm")

    @staticmethod
    def _eval_call(op: Any) -> Call | None:
        if isinstance(op, Evaluate) and isinstance(op.value, Call):
            return op.value
        if isinstance(op, SBlockRealize):
            op = op.block
        if isinstance(op, SBlock):
            body = op.body
            if isinstance(body, SeqStmt) and len(body.seq) == 1:
                return HexagonEmitter._eval_call(body.seq[0])
            if isinstance(body, Evaluate) and isinstance(body.value, Call):
                return body.value
            return HexagonEmitter._eval_call(body)
        if isinstance(op, AttrStmt):
            return HexagonEmitter._eval_call(op.body)
        return None

    @staticmethod
    def _extern_callee_name(call: Call) -> str | None:
        if _call_op_name(call) not in ("tir.call_pure_extern", "tirx.call_pure_extern"):
            return None
        return _ann_str(call.args[0]) if call.args else None

    def _looks_like_row_loop(self, op: For) -> bool:
        var = _var_name(op.loop_var)
        return var in ("m", "rb")

    def _expr_c(self, e: Any) -> str:
        iv = _i64(e)
        if iv is not None:
            return str(iv)
        if hasattr(e, "name") or hasattr(e, "name_hint"):
            return _var_name(e)
        cls = type(e).__name__
        cls_key = cls.lower()
        if cls_key in ("add", "sub", "mul", "div", "floordiv", "floormod", "mod", "lt", "le", "gt", "ge", "eq", "ne", "and", "or"):
            a, b = e.a, e.b
            op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "floordiv": "/", "floormod": "%", "mod": "%",
                  "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!=", "and": "&&", "or": "||"}[cls_key]
            return f"({self._expr_c(a)} {op} {self._expr_c(b)})"
        if cls == "Cast":
            return self._expr_c(e.value)
        raise HexagonEmitError(f"白名单: 不支持表达式 {cls}{_loc(e)}")

    # ---- C rendering -------------------------------------------------------

    def _infer_problem_shape(self) -> tuple[int, int, int]:
        kg = self.gemm_mnk[2] if self.gemm_mnk is not None else None
        globals_2d = [b for b in self.buffers.values() if _buffer_scope(b) == "global" and len(b.shape) == 2]
        shapes = [tuple(_i64(d) for d in b.shape) for b in globals_2d]
        shapes = [s for s in shapes if len(s) == 2 and s[0] is not None and s[1] is not None]
        if len(shapes) >= 3:
            a = max(shapes, key=lambda s: s[1])
            k = int(kg or a[1])
            candidates_c = [s for s in shapes if s[0] == a[0] and s[1] != k]
            c = max(candidates_c, key=lambda s: s[1]) if candidates_c else None
            if c is not None:
                return int(a[0]), int(c[1]), k
        if self.gemm_mnk is not None:
            return self.gemm_mnk
        if self._uses_gdn_shell():
            return (1024, 0, 0)
        raise HexagonEmitError("R5: v2 emitter 目前只支持包含 T.gemm 的 GEMM_NT kernel")

    def _render_c(self) -> str:
        if self._uses_gdn_shell():
            return self._render_gdn_c()
        if (self.has_vector_kernel or self.has_scalar_kernel) and not self.has_gemm:
            return self._render_vector_c()
        if not self.has_gemm:
            raise HexagonEmitError("R5: v2 emitter 目前只支持包含 T.gemm 的 GEMM_NT kernel")
        m, n, k = self._infer_problem_shape()
        if self._generic_gemm_recipe_mode() or self.has_vector_kernel or self.has_scalar_kernel or self._pool_workers:
            return self._render_generic_c(m, n, k)
        if not self._legacy_gemm_mode():
            return self._render_gemm_c_generic(m, n, k)
        if m % 32 or n % 256 or k % 64:
            raise HexagonEmitError(f"R5/R6: 入口形状必须 M%32,N%256,K%64, 实际 M={m},N={n},K={k}")
        gm_kmax = max(k, 4096)
        gm_act = gm_kmax * 64
        gm_wa = gm_act * 2
        gm_out = gm_wa + GM_WA_MAX
        gm_end = gm_out + 2 * GM_TILE
        block_n = self.block_N_hint or 1024
        if block_n not in (1024, 512, 256) or block_n % 256:
            raise HexagonEmitError(f"R6: block_N 必须为 1024/512/256 且 %256, 实际 {block_n}")
        if gm_end > VTCM_BUDGET:
            raise HexagonEmitError(f"R3: 生成物 VTCM map 超 8MB: GM_END={gm_end}")
        body = "\n".join(self.body_lines)
        return f'''// Generated by tilelang.hexagon.emitter: statement-level GEMM_NT v2.
// Low-level HVX/HMX recipes live in hexagon_rt.h; this file only lowers TIR control flow.
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "attnops.h"
#include "HAP_perf.h"
#include "hexagon_rt.h"

typedef hrt_f16 f16;
typedef hrt_f32 f32;

#define GM_TILE HRT_TILE_BYTES
#define GM_KMAX {gm_kmax}
#define GM_BLOCK_N {block_n}
#define GM_LIN  (0)
#define GM_ACT  (GM_LIN + GM_KMAX * 64)
#define GM_WA   (GM_ACT + GM_KMAX * 64)
#define GM_WA_MAX ((size_t){GM_WA_MAX})
#define GM_OUT  (GM_WA + GM_WA_MAX)
#define GM_END  (GM_OUT + 2 * GM_TILE)

int {self.func_name}(remote_handle64 h, unsigned char *slab, int slabLen,
                     unsigned char *w, int wLen, int M, int N, int K, int abl) {{
    (void)h;
    if (!HRT_VTCM_READY()) return -3;
    if (M <= 0 || N <= 0 || K <= 0 || M % 32 || N % 256 || K % 64 || K > GM_KMAX) return -2;
    if ((size_t)GM_END > HRT_VTCM_SIZE()) return -4;
    int NP = 0;
    for (int t = GM_BLOCK_N; t >= 256; t >>= 1)
        if (N % t == 0 && (size_t)K * t * 2 <= GM_WA_MAX) {{ NP = t; break; }}
    if (!NP) return -2;
    size_t x_sz = (size_t)M * K * 2;
    size_t c_off = HRT_ALIGN128(x_sz);
    size_t c_sz = (size_t)M * N * 2;
    size_t prof_off = HRT_ALIGN128(c_off + c_sz);
    if ((size_t)slabLen < prof_off + 20) return -1;
    if ((size_t)wLen < (size_t)K * N * 2) return -1;
    const f16 *A = (const f16 *)slab;
    f16 *C = (f16 *)(slab + c_off);
    HRT_PROF_DECL(prof, slab, prof_off);
    HRT_PROF_CLEAR(prof, 20);

    uint8_t *V = HRT_VTCM_BASE();
    int kt = K / 32, nct = NP / 32;
    uint64_t t0, tt = HAP_perf_get_qtimer_count();
{body}
    prof[4] = (int32_t)(HAP_perf_get_qtimer_count() - tt);
    return 0;
}}
'''

    def _render_gemm_c_generic(self, m: int, n: int, k: int) -> str:
        if m % 32 or n % 32 or k % 32:
            raise HexagonEmitError(f"R5: 入口形状必须 M/N/K 32 整除, 实际 M={m},N={n},K={k}")
        block_n = self.block_N_hint or 1024
        if block_n % 32:
            raise HexagonEmitError(f"R5: block_N 必须 32 整除, 实际 {block_n}")
        block_m = self.block_M_hint or 32
        if block_m % 32:
            raise HexagonEmitError(f"R5: block_M 必须 32 整除, 实际 {block_m}")
        gm_kmax = k
        offsets = {a.name: a.offset for a in self.allocs if a.offset is not None}
        vtcm = [a for a in self.allocs if a.bytes]
        accs = [a for a in self.allocs if _is_acc(a.scope)]
        if len(vtcm) < 2 or not accs:
            raise HexagonEmitError("R5: GEMM 需要 A/B alloc_shared 和 C alloc_fragment")
        gm_act = vtcm[0].offset or 0
        gm_wa = vtcm[1].offset or 0
        gm_out = accs[0].offset or 0
        gm_end = max((a.offset or 0) + a.bytes for a in self.allocs if a.bytes)
        body = "\n".join(self.body_lines)
        body = body.replace(
            "for (int bx = 0; bx < N / NP; bx++) {\n",
            "for (int bx = 0; bx < (N + NP - 1) / NP; bx++) {\n"
            "        int n_rem_bx = N - bx * NP;\n"
            "        int nct = (n_rem_bx < NP ? n_rem_bx : NP) / 32;\n",
            1,
        )
        return f'''// Generated by tilelang.hexagon.emitter: statement-level GEMM_NT v2 (user-tiled).
// Low-level HVX/HMX recipes live in hexagon_rt.h; this file only lowers TIR control flow.
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "attnops.h"
#include "HAP_perf.h"
#include "hexagon_rt.h"

typedef hrt_f16 f16;
typedef hrt_f32 f32;

#define GM_TILE HRT_TILE_BYTES
#define GM_KMAX {gm_kmax}
#define GM_BLOCK_M {block_m}
#define GM_BLOCK_N {block_n}
#define GM_ACT  ((size_t){gm_act})
#define GM_WA   ((size_t){gm_wa})
#define GM_OUT  ((size_t){gm_out})
#define GM_END  ((size_t){gm_end})

int {self.func_name}(remote_handle64 h, unsigned char *slab, int slabLen,
                     unsigned char *w, int wLen, int M, int N, int K, int abl) {{
    (void)h;
    if (!HRT_VTCM_READY()) return -3;
    if (M <= 0 || N <= 0 || K <= 0 || M % 32 || N % 32 || K % 32 || K > GM_KMAX) return -2;
    if ((size_t)GM_END > HRT_VTCM_SIZE()) return -4;
    int NP = GM_BLOCK_N;
    size_t x_sz = (size_t)M * K * 2;
    size_t c_off = HRT_ALIGN128(x_sz);
    size_t c_sz = (size_t)M * N * 2;
    size_t prof_off = HRT_ALIGN128(c_off + c_sz);
    if ((size_t)slabLen < prof_off + 20) return -1;
    if ((size_t)wLen < (size_t)K * N * 2) return -1;
    const f16 *A = (const f16 *)slab;
    f16 *C = (f16 *)(slab + c_off);
    HRT_PROF_DECL(prof, slab, prof_off);
    HRT_PROF_CLEAR(prof, 20);

    uint8_t *V = HRT_VTCM_BASE();
    int kt = K / 32, nct = NP / 32;
    (void)nct;  // user-tiled 路径内层循环重声明,入口副本可能不引用
    uint64_t t0, tt = HAP_perf_get_qtimer_count();
{body}
    prof[4] = (int32_t)(HAP_perf_get_qtimer_count() - tt);
    return 0;
}}
'''

    def _render_pool_worker_defs(self) -> str:
        if not self._pool_workers:
            return ""
        chunks: list[str] = []
        for worker in self._pool_workers:
            fields: list[str] = []
            locals_: list[str] = []
            for arg in self.global_args:
                ct = self._buffer_elem_ctype(arg.buf)
                qual = "const " if self._is_readonly_global(arg.buf) else ""
                fields.append(f"    {qual}{ct} *{arg.cname};")
                locals_.append(f"    {qual}{ct} *{arg.cname} = ctx->{arg.cname};")
            for name in self._shape_params:
                fields.append(f"    int {name};")
                locals_.append(f"    const int {name} = ctx->{name};")
            for name in worker.outer_vars:
                fields.append(f"    int {name};")
                locals_.append(f"    const int {name} = ctx->{name};")
            fields.append("    int abl;")
            fields.append("    int32_t *prof;")
            locals_.append("    const int abl = ctx->abl;")
            locals_.append("    int32_t *prof = ctx->prof;")
            body = "\n".join(worker.body)
            # Only unpack ctx fields the worker body actually references;
            # unpacking everything would trip -Wunused-variable (-Werror in
            # the skel build) for pool phases that touch a subset of args.
            locals_ = [
                line for line in locals_
                if re.search(r"\b%s\b" % re.escape(line.split("=", 1)[0].strip().split()[-1].lstrip("*")), body)
            ]
            fields_s = "\n".join(fields)
            locals_s = "\n".join(locals_)
            # (void)-cast every unpacked local that survives (plus V/t0) so an
            # unused-but-referenced-elsewhere decl can never warn; keep the
            # cast list in sync with the filtered locals.
            local_names = [
                line.split("=", 1)[0].strip().split()[-1].lstrip("*")
                for line in locals_
            ]
            voids = "".join(f" (void){n};" for n in ["V", "t0", *local_names])
            chunks.append(f'''
typedef struct {{
{fields_s}
}} {worker.ctx_type};

static void {worker.name}(int job, void *opaque) {{
    const {worker.ctx_type} *ctx = (const {worker.ctx_type} *)opaque;
{locals_s}
    uint8_t *V = HRT_VTCM_BASE();
    uint64_t t0;{voids}
{body}
}}
''')
        return "\n".join(chunks)

    def _render_generic_c(self, m: int, n: int, k: int) -> str:
        del m, n, k  # Generic shell uses per-call static recipe sizes and ABI scalars.
        body = "\n".join(self.body_lines)
        func_name = self.func_name
        vtcm_bytes = self._assign_vtcm_offsets()
        shape_params = {name: name for name in self._shape_params}
        prof_slots = max(5, self._prof_next_slot) if self.hexagon_prof else 5
        slab_lines, prof_off = self._emit_slab_layout(1, shape_params, prof_bytes=prof_slots * 4)
        slab = "\n".join(slab_lines)
        worker_defs = self._render_pool_worker_defs()
        gm_out_define = ""
        accs = [a for a in self.allocs if _is_acc(a.scope) and a.offset is not None]
        if accs:
            gm_out_define = f"\n#define TL_ACC ((size_t){accs[0].offset})"
        scalar_sig = "".join(f", int {p}" for p in self._shape_params)
        time_decl = "uint64_t t0, tl_prof_phase_t0, tt = HAP_perf_get_qtimer_count();" if self.hexagon_prof else ("uint64_t t0, tt = HAP_perf_get_qtimer_count();" if "t0 =" in body else "uint64_t tt = HAP_perf_get_qtimer_count();")
        prof_comment = ""
        if self.hexagon_prof:
            prof_comment = "\n".join([
                "",
                "// tl.hexagon_prof profile slots (int32 qtimer ticks, accumulated like gsp_prof):",
                "//   prof[0]: pre-loop/weight-staging aggregate when emitted (legacy generic slot)",
                "//   prof[1]: main-thread staging copies (rm/f32 -> AH/WH) aggregate",
                "//   prof[2]: main-thread HMX matmul + accumulator read aggregate",
                "//   prof[3]: main-thread accumulator readback/unpermute/writeback aggregate",
                "//   prof[4]: whole entry wall time",
                f"//   prof[5..{prof_slots - 1}]: worker-pool calls in source order (one slot per synchronous T.parallel phase)",
            ])
        return f'''// Generated by tilelang.hexagon.emitter: generic mixed pool/HMX shell v1.
// T.parallel phases are synchronous worker-pool jobs; serial/T.gemm phases run on the caller thread.
{prof_comment}
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "attnops.h"
#include "HAP_perf.h"
#include "hexagon_rt.h"

typedef hrt_f16 f16;
typedef hrt_f32 f32;

#define TL_GENERIC_VTCM_BYTES ((size_t){vtcm_bytes}){gm_out_define}
{worker_defs}

int {func_name}(remote_handle64 h, unsigned char *slab, int slabLen{scalar_sig}, int abl) {{
    (void)h;
    if (!HRT_VTCM_READY()) return -3;
    if (TL_GENERIC_VTCM_BYTES > HRT_VTCM_SIZE()) return -4;
{slab}
    HRT_PROF_DECL(prof, slab, {prof_off});
    HRT_PROF_CLEAR(prof, {prof_slots * 4});
    uint8_t *V = HRT_VTCM_BASE();
    const size_t total = (size_t)-1;
    (void)V; (void)total;
    {time_decl}
{body}
    prof[4] = (int32_t)(HAP_perf_get_qtimer_count() - tt);
    return 0;
}}
'''

    def _render_gdn_c(self) -> str:
        body = "\n".join(self.body_lines)
        func_name = self.func_name
        if self.gdn_mode == "leaf":
            vtcm_slot_bytes = 0
            if True:
                worker = f'''
typedef struct {{
    const f16 *q;
    const f16 *k;
    const f16 *v;
    const f32 *g;
    const f32 *b;
    const f32 *s0;
    f16 *o;
    f32 *s1;
    int T;
    int hk_heads;
    int hv_heads;
}} {func_name}_ctx_t;

static void {func_name}_worker(int job, void *opaque) {{
    const {func_name}_ctx_t *ctx = (const {func_name}_ctx_t *)opaque;
    const f16 *q = ctx->q;
    const f16 *k = ctx->k;
    const f16 *v = ctx->v;
    const f32 *g = ctx->g;
    const f32 *b = ctx->b;
    const f32 *s0 = ctx->s0;
    f16 *o = ctx->o;
    f32 *s1 = ctx->s1;
    const int T = ctx->T;
    const int hk_heads = ctx->hk_heads;
    uint8_t *V = HRT_VTCM_BASE();
    hrt_tlgdn_slot_t *slot = &g_hrt_gdn_slots[job % NWORKERS];
    (void)V; (void)slot; (void)q; (void)k; (void)v; (void)g; (void)b; (void)s0; (void)o; (void)s1; (void)T; (void)hk_heads;
{body}
}}
'''
                run_body = f'''    {func_name}_ctx_t ctx = {{ q, k, v, g, b, s0, o, s1, T, hk_heads, hv_heads }};
    attnops_pool_run_ctx({func_name}_worker, &ctx, hv_heads);'''
            else:
                worker = ""
                run_body = body
            return f'''// Generated by tilelang.hexagon.emitter: GDN prefill leaf-primitive v1.
// User TIR owns head/chunk/token/data-flow structure; hexagon_rt.h only supplies hardware-action leaves.
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "attnops.h"
#include "HAP_perf.h"
#include "hexagon_rt.h"

typedef hrt_f16 f16;
typedef hrt_f32 f32;
static const size_t TLGDN_VTCM_SLOT_BYTES = (size_t){vtcm_slot_bytes};
{worker}

int {func_name}(remote_handle64 h, unsigned char *slab, int slabLen, int T, int hk_heads, int hv_heads) {{
    (void)h;
    if (T <= 0 || T % 32 || hk_heads <= 0 || hv_heads <= 0 || hv_heads % hk_heads) return -2;
    size_t qk = (size_t)hk_heads * T * 128;
    size_t vv = (size_t)hv_heads * T * 128;
    size_t gb = (size_t)hv_heads * T;
    size_t st = (size_t)hv_heads * 128 * 128;
    size_t off = 0;
    const f16 *q = (const f16 *)(slab + off); off += qk * 2;
    const f16 *k = (const f16 *)(slab + off); off += qk * 2;
    const f16 *v = (const f16 *)(slab + off); off += vv * 2;
    const f32 *g = (const f32 *)(slab + off); off += gb * 4;
    const f32 *b = (const f32 *)(slab + off); off += gb * 4;
    const f32 *s0 = (const f32 *)(slab + off); off += st * 4;
    f16 *o = (f16 *)(slab + off); off += vv * 2;
    f32 *s1 = (f32 *)(slab + off); off += st * 4;
    size_t prof_off = HRT_ALIGN128(off);
    if ((size_t)slabLen < prof_off + 4) return -1;
    HRT_PROF_DECL(prof, slab, prof_off);
    HRT_PROF_CLEAR(prof, 4);
    if (!HRT_VTCM_READY()) return -3;
    if (TLGDN_VTCM_SLOT_BYTES * NWORKERS > HRT_VTCM_SIZE()) return -4;
    uint64_t tt = HAP_perf_get_qtimer_count();
{run_body}
    prof[0] = (int32_t)(HAP_perf_get_qtimer_count() - tt);
    return 0;
}}
'''
        return f'''// Generated by tilelang.hexagon.emitter: GDN prefill v0.
// TileLang lowers the T.Kernel/head structure; chunk HVX lowering lives in hexagon_rt.h.
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "attnops.h"
#include "HAP_perf.h"
#include "hexagon_rt.h"

typedef hrt_f16 f16;
typedef hrt_f32 f32;

int {func_name}(remote_handle64 h, unsigned char *slab, int slabLen, int T, int hk_heads, int hv_heads) {{
    (void)h;
    if (T != 1024 || hk_heads != 16 || hv_heads != 32) return -2;
    size_t qk = (size_t)hk_heads * T * 128;
    size_t vv = (size_t)hv_heads * T * 128;
    size_t gb = (size_t)hv_heads * T;
    size_t st = (size_t)hv_heads * 128 * 128;
    size_t off = qk * 2 + qk * 2 + vv * 2 + gb * 4 + gb * 4 + st * 4 + vv * 2 + st * 4;
    size_t prof_off = HRT_ALIGN128(off);
    if ((size_t)slabLen < prof_off + 4) return -1;
    HRT_PROF_DECL(prof, slab, prof_off);
    HRT_PROF_CLEAR(prof, 4);
{body}
    return 0;
}}
'''

    def _render_vector_c(self) -> str:
        body = "\n".join(self.body_lines)
        func_name = self.func_name
        vtcm_bytes = self._assign_vtcm_offsets()
        nworkers = self.nworkers or 1
        if not self.global_args:
            raise HexagonEmitError("R5: elementwise kernel 需要至少一个 global 张量参数")
        if nworkers > 6:
            raise HexagonEmitError(f"R10: Hexagon worker 数必须 ≤6, 实际 {nworkers}")
        # 入口指针、slab 布局、worker 静态通道全部由 prim_func 的 global 张量
        # 声明顺序驱动(global_args),不再认 G/U/O 之类的固定名字。const 由
        # “是否被 BufferStore 写过”推导(_is_readonly_global)。elementwise v1
        # 约定:所有 global 张量都是 M*FF 平铺,布局 = 声明序连续 128B 对齐。
        statics: list[str] = []
        locals_: list[str] = []
        assigns: list[str] = []
        decls: list[str] = ["    size_t off = 0;"]
        for arg in self.global_args:
            ct = self._buffer_elem_ctype(arg.buf)
            qual = "const " if self._is_readonly_global(arg.buf) else ""
            esz = 4 if ct == "f32" else 2
            statics.append(f"static {qual}{ct} *g_velem_{arg.cname};")
            locals_.append(f"    {qual}{ct} *{arg.cname} = g_velem_{arg.cname};")
            assigns.append(f"    g_velem_{arg.cname} = {arg.cname};")
            decls.append(f"    {qual}{ct} *{arg.cname} = ({qual}{ct} *)(slab + off);")
            decls.append(f"    off = HRT_ALIGN128(off + elems * {esz});")
        statics_s = "\n".join(statics)
        locals_s = "\n".join(locals_)
        assigns_s = "\n".join(assigns)
        decls_s = "\n".join(decls)
        accs = [a for a in self.allocs if _is_acc(a.scope) and a.offset is not None]
        gm_out_define = f"\n#define TL_ACC ((size_t){accs[0].offset})" if accs else ""
        if self.vector_uses_pool:
            worker = f'''
{statics_s}
static size_t g_velem_total;
static int g_velem_FF;
static int g_velem_full_panels;

static void {func_name}_worker(int job) {{
{locals_s}
    const size_t total = g_velem_total;
    const int FF = g_velem_FF;
    uint8_t *V = HRT_VTCM_BASE() + (size_t)(job % NWORKERS) * VELEM_VTCM_BYTES;
    (void)FF;
    (void)V;
{body}
}}
'''
            launch = f'''{assigns_s}
    g_velem_total = total;
    g_velem_FF = FF;
    g_velem_full_panels = (int)(total / PANEL);
    attnops_pool_run({func_name}_worker, g_velem_full_panels);
    if ((size_t)g_velem_full_panels * PANEL < total) {{
        {func_name}_worker(g_velem_full_panels);
    }}'''
        else:
            worker = ""
            launch = body
        time_decl = "uint64_t t0, tt = HAP_perf_get_qtimer_count();" if "t0 =" in launch else "uint64_t tt = HAP_perf_get_qtimer_count();"
        return f'''// Generated by tilelang.hexagon.emitter: elementwise HVX vector v1.
// T.Kernel threads<=6 is lowered to attnops_pool_run over blockIdx.x panels;
// the caller thread handles the post-join partial tail panel, if any.
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "attnops.h"
#include "HAP_perf.h"
#include "hexagon_rt.h"

typedef hrt_f16 f16;
typedef hrt_f32 f32;

#define PANEL 1024
#define VELEM_VTCM_BYTES ((size_t){vtcm_bytes})
#define NWORKERS {nworkers}{gm_out_define}
{worker}

int {func_name}(remote_handle64 h, unsigned char *slab, int slabLen,
                int M, int FF, int abl) {{
    (void)h; (void)abl;
    if (M <= 0 || FF <= 0 || FF % 64) return -2;
    size_t elems = (size_t)M * FF;
{decls_s}
    size_t prof_off = HRT_ALIGN128(off);
    if ((size_t)slabLen < prof_off + 4) return -1;
    if (VELEM_VTCM_BYTES && (!HRT_VTCM_READY() || VELEM_VTCM_BYTES * NWORKERS > HRT_VTCM_SIZE())) return -4;
    uint8_t *V = HRT_VTCM_BASE();
    (void)V;
    HRT_PROF_DECL(prof, slab, prof_off);
    HRT_PROF_CLEAR(prof, 4);
    const size_t total = elems;
    {time_decl}
{launch}
    prof[0] = (int32_t)(HAP_perf_get_qtimer_count() - tt);
    return 0;
}}
'''


    @staticmethod
    def _ind(n: int, s: str) -> str:
        return "    " * n + s


def emit_hexagon_c(mod: IRModule | PrimFunc, target: Target | None = None, output_path: str | os.PathLike[str] | None = None) -> str:
    return HexagonEmitter(mod, target).emit(output_path=output_path)
