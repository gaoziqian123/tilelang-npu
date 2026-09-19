from __future__ import annotations

"""OpenCL TIR-side fragment vectorization metadata pass.

The legacy OpenCL printer already understands OpenCL vector types when the IR
contains vector values, but TileLang's 8x8 GEMM fragment lowering reaches the
printer as a fully unrolled private ``float acc[64]`` plus scalar global/shared
loads/stores.  This pass recognizes the canonical post-UnrollLoop TIR shape and
records the vectorized lowering contract on the PrimFunc before codegen.  The
OpenCL codegen wrapper consumes this pass-produced metadata instead of applying
the old GEMM_NT body regex rewrites.  A much narrower accumulator-only fallback
remains in codegen.py for texture GEMM, whose texture-scope TIR has not yet been
normalized to the buffer-fragment shape matched here.

Keeping this in Python is intentional: the match is OpenCL-example specific and
depends on tirx buffer remapping after TileLang's existing lowering passes.  A
C++ pass would need to duplicate the same fragile shape knowledge while making
iteration slower; once the upstream TIR carries first-class OpenCL vector memory
ops for this pattern, this file can be replaced by a general C++ SROA pass.
"""

import re
import os
from dataclasses import dataclass

from tvm import IRModule, tirx
from tvm.script.parser import parse
from tvm.tirx.transform import prim_func_pass


@dataclass(frozen=True)
class FragmentPlan:
    kind: str
    b_layout: str
    M: int
    N: int
    K: int
    bm: int
    bn: int
    bk: int
    threads: int

def _int(v) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _shape(buf) -> tuple[int, ...] | None:
    vals = []
    for dim in getattr(buf, "shape", ()):  # tirx Array[IntImm]
        iv = _int(dim)
        if iv is None:
            return None
        vals.append(iv)
    return tuple(vals)


def _buffer_shapes(func: tirx.PrimFunc, src: str | None = None) -> tuple[int, int, int, str] | None:
    attrs = getattr(func, "attrs", None)
    attr_b_layout = None
    if attrs and "tl.opencl.b_layout" in attrs:
        attr_b_layout = str(attrs["tl.opencl.b_layout"]).strip('"')
    attr_b_layout = attr_b_layout or os.environ.get("TL_OPENCL_B_LAYOUT")
    buffer_map = getattr(func, "buffer_map", None)
    if not buffer_map or len(buffer_map) < 3:
        if src is None:
            return None
        # After MakePackedAPI/LowerDeviceKernelLaunch the device PrimFunc has
        # raw handle parameters and explicit decl_buffer nodes.  Recover the
        # three flattened MK/NK/MN sizes from the script form; this is only a
        # recognizer for the canonical static OpenCL examples, not a semantic
        # parser for arbitrary TIR.
        vals = {m.group(2): int(m.group(1)) for m in re.finditer(r"T\.decl_buffer\(\((\d+),\), \"float16\", data=([ABC])\)", src)}
        if not {"A", "B", "C"}.issubset(vals):
            return None
        # A=M*K, B=N*K, C=M*N.  Static examples use integer dimensions.
        AK, BK, CN = vals["A"], vals["B"], vals["C"]
        if AK <= 0 or BK <= 0 or CN <= 0:
            return None
        # K^2 = (A*K * B*K) / (M*N)
        k2_num = AK * BK
        if k2_num % CN:
            return None
        k2 = k2_num // CN
        K = int(k2 ** 0.5)
        if K * K != k2 or AK % K or BK % K:
            return None
        M, N = AK // K, BK // K
        if M * N != CN:
            return None
        # Flattened packed-API recovery loses the 2-D B layout.  The canonical
        # direct-Bt fragment indexes B as B[kidx, n] before this pass; recover
        # that contract from the script text when the BufferMap is gone.
        b_layout = attr_b_layout or ("kn" if re.search(r"B\[[^,\]]+kidx[^,\]]*,\s*[^\]]+tn", src) else "nk")
        return M, N, K, b_layout
    shapes = [_shape(buf) for buf in buffer_map.values()]
    if any(s is None or len(s) != 2 for s in shapes[:3]):
        return None
    a, b, c = shapes[:3]
    if a[1] == b[1] and c == (a[0], b[0]):
        return a[0], b[0], a[1], attr_b_layout or "nk"
    if a[1] == b[0] and c == (a[0], b[1]):
        return a[0], b[1], a[1], attr_b_layout or "kn"
    return None


def _thread_extents(func: tirx.PrimFunc) -> dict[str, int]:
    out: dict[str, int] = {}
    attrs = getattr(func, "attrs", None)
    te = attrs.get("thread_extent") if attrs and "thread_extent" in attrs else None
    if te:
        for k, v in te.items():
            iv = _int(v)
            if iv is not None:
                out[str(k)] = iv
    return out


def _has_acc64(src: str) -> bool:
    return 'acc = T.alloc_buffer((64,), scope="local")' in src and src.count("acc[") >= 128


def _detect_bk_direct(src: str, K: int) -> int | None:
    # Post-unroll direct fragment keeps a two-dimensional grid(ko_extent, bk).
    m = re.search(r"for ko, kk in T\.grid\((\d+), (\d+)\):", src)
    if not m:
        # If the outer ko loop was simplified away (common when BK==K), the
        # direct fragment is a single scalar K loop.
        m1 = re.search(r"for kk in range\((\d+)\):", src)
        return K if m1 and int(m1.group(1)) == K else None
    ko, bk = int(m.group(1)), int(m.group(2))
    return bk if ko * bk == K else None


def _detect_bk_tiled(src: str, K: int) -> int | None:
    m = re.search(r"for ko in range\((\d+)\):", src)
    if not m:
        return None
    ko = int(m.group(1))
    if ko <= 0 or K % ko:
        return None
    return K // ko


def _detect_plan(func: tirx.PrimFunc) -> FragmentPlan | None:
    src = str(func)
    dims = _buffer_shapes(func, src)
    if dims is None:
        return None
    M, N, K, b_layout = dims
    te = _thread_extents(func)
    bx = te.get("blockIdx.x")
    by = te.get("blockIdx.y")
    threads = te.get("threadIdx.x")
    if not bx or not by or not threads:
        return None
    if N % bx or M % by:
        return None
    bn = N // bx
    bm = M // by
    if bm % 8 or bn % 8 or threads != (bm // 8) * (bn // 8):
        return None
    if not _has_acc64(src):
        return None
    tiled = "T.tvm_storage_sync(\"shared\")" in src and "scope=\"shared" in src
    if tiled:
        bk = _detect_bk_tiled(src, K)
        kind = "tiled_fragment_8x8"
    else:
        bk = _detect_bk_direct(src, K)
        kind = "fragment_8x8"
    if bk is None or bk <= 0 or bk % 4:
        return None
    return FragmentPlan(kind, b_layout, M, N, K, bm, bn, bk, threads)


def _attrs(threads: int, bx: int, by: int) -> str:
    return (
        'T.func_attr({"calling_conv": 2, "tirx.is_global_func": T.bool(True), '
        f'"thread_extent": {{"blockIdx.x": {bx}, "blockIdx.y": {by}, '
        f'"threadIdx.x": {threads}, "threadIdx.y": 1, "threadIdx.z": 1}}, '
        '"tirx.kernel_launch_params": ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z"], '
        '"tirx.noalias": True, "tl.readonly_param_indices": [0, 1]})'
    )


def _shuffle_vec(names: list[str], comp: int) -> str:
    idx = [comp + 4 * j for j in range(8)]
    return f"T.Shuffle([{', '.join(names)}], [{', '.join(str(i) for i in idx)}])"


def _direct_fragment_script(p: FragmentPlan) -> str:
    bx = p.N // p.bn
    by = p.M // p.bm
    tiles_n = p.bn // 8
    lines = [
        "# from tvm.script import ir as I",
        "# from tvm.script import tirx as T",
        "@I.ir_module",
        "class Module:",
        "    @T.prim_func",
        '    def gemm_nt_kernel_kernel(A: T.handle("float16", "global"), B: T.handle("float16", "global"), C: T.handle("float16", "global")):',
        "        " + _attrs(p.threads, bx, by),
        f'        A_1 = T.decl_buffer(({p.M * p.K},), "float16", data=A)',
        f'        B_1 = T.decl_buffer(({p.N * p.K},), "float16", data=B)',
        f'        C_1 = T.decl_buffer(({p.M * p.N},), "float16", data=C)',
        f'        bx = T.launch_thread("blockIdx.x", {bx})',
        f'        by = T.launch_thread("blockIdx.y", {by})',
        f'        tx = T.launch_thread("threadIdx.x", {p.threads})',
        '        ty = T.launch_thread("threadIdx.y", 1)',
        '        tz = T.launch_thread("threadIdx.z", 1)',
        '        acc = T.alloc_buffer((8,), "float32x8", scope="local")',
        '        tm = tx // %d' % tiles_n,
        '        tn = tx - tm * %d' % tiles_n,
        f'        r0 = by * {p.bm} + tm * 8',
        f'        c0 = bx * {p.bn} + tn * 8',
        f'        for pos in range(0, {p.K}, 4):',
    ]
    lines[-1:-1] = [f'        acc[{i}] = T.Broadcast(T.float32(0), 8)' for i in range(8)]
    for i in range(8):
        lines.append(f'            a{i}: T.float32x4 = T.call_extern("float32x4", "convert_float4", T.call_extern("float16x4", "vload4", 0, T.address_of(A_1[(r0 + {i}) * {p.K} + pos])))')
    if p.b_layout == "kn":
        for comp in range(4):
            lines.append(f'            b{comp}: T.float32x8 = T.call_extern("float32x8", "convert_float8", T.call_extern("float16x8", "vload8", 0, T.address_of(B_1[(pos + {comp}) * {p.N} + c0])))')
        for i in range(8):
            for comp in range(4):
                lines.append(f'            acc[{i}] = acc[{i}] + T.Broadcast(T.Shuffle([a{i}], [{comp}]), 8) * b{comp}')
    else:
        for j in range(8):
            lines.append(f'            b{j}: T.float32x4 = T.call_extern("float32x4", "convert_float4", T.call_extern("float16x4", "vload4", 0, T.address_of(B_1[(c0 + {j}) * {p.K} + pos])))')
        for i in range(8):
            for comp in range(4):
                args = ", ".join(f"T.Shuffle([b{j}], [{comp}])" for j in range(8))
                lines.append(f'            acc[{i}] = acc[{i}] + T.Broadcast(T.Shuffle([a{i}], [{comp}]), 8) * T.call_extern("float32x8", "(float8)", {args})')
    for i in range(8):
        lines.append(f'        C_1[T.Ramp((r0 + {i}) * {p.N} + c0, 1, 8)] = T.Cast("float16x8", acc[{i}])')
    return "\n".join(lines) + "\n"


def _tiled_fragment_script(p: FragmentPlan) -> str:
    bx = p.N // p.bn
    by = p.M // p.bm
    tiles_n = p.bn // 8
    lines = [
        "# from tvm.script import ir as I",
        "# from tvm.script import tirx as T",
        "@I.ir_module",
        "class Module:",
        "    @T.prim_func",
        '    def gemm_nt_kernel_kernel(A: T.handle("float16", "global"), B: T.handle("float16", "global"), C: T.handle("float16", "global")):',
        "        " + _attrs(p.threads, bx, by),
        f'        A_1 = T.decl_buffer(({p.M * p.K},), "float16", data=A)',
        f'        B_1 = T.decl_buffer(({p.N * p.K},), "float16", data=B)',
        f'        C_1 = T.decl_buffer(({p.M * p.N},), "float16", data=C)',
        f'        bx = T.launch_thread("blockIdx.x", {bx})',
        f'        by = T.launch_thread("blockIdx.y", {by})',
        f'        tx = T.launch_thread("threadIdx.x", {p.threads})',
        '        ty = T.launch_thread("threadIdx.y", 1)',
        '        tz = T.launch_thread("threadIdx.z", 1)',
        f'        As = T.alloc_buffer(({p.bk * p.bm},), "float16", scope="shared")',
        f'        Bs = T.alloc_buffer(({p.bk * p.bn},), "float16", scope="shared")',
        '        acc = T.alloc_buffer((8,), "float32x8", scope="local")',
        '        tm = tx // %d' % tiles_n,
        '        tn = tx - tm * %d' % tiles_n,
        f'        rbase = by * {p.bm}',
        f'        cbase = bx * {p.bn}',
    ]
    lines += [f'        acc[{i}] = T.Broadcast(T.float32(0), 8)' for i in range(8)]
    lines += [
        f'        for kb in range(0, {p.K}, {p.bk}):',
        f'            for v in range(tx, {p.bm * p.bk // 4}, {p.threads}):',
        f'                r = v // ({p.bk} // 4)',
        f'                c4 = v - r * ({p.bk} // 4)',
        f'                aval: T.float16x4 = T.call_extern("float16x4", "vload4", 0, T.address_of(A_1[(rbase + r) * {p.K} + kb + c4 * 4]))',
    ]
    for lane in range(4):
        lines.append(f'                As[(c4 * 4 + {lane}) * {p.bm} + r] = T.Shuffle([aval], [{lane}])')
    lines += [
        f'            for v in range(tx, {p.bn * p.bk // 4}, {p.threads}):',
        f'                c = v // ({p.bk} // 4)',
        f'                k4 = v - c * ({p.bk} // 4)',
        f'                bval: T.float16x4 = T.call_extern("float16x4", "vload4", 0, T.address_of(B_1[(cbase + c) * {p.K} + kb + k4 * 4]))',
    ]
    for lane in range(4):
        lines.append(f'                Bs[(k4 * 4 + {lane}) * {p.bn} + c] = T.Shuffle([bval], [{lane}])')
    lines += [
        '            T.tvm_storage_sync("shared")',
        f'            for kk in T.unroll({p.bk}):',
        f'                a8: T.float32x8 = T.Cast("float32x8", As[T.Ramp(kk * {p.bm} + tm * 8, 1, 8)])',
        f'                b8: T.float32x8 = T.Cast("float32x8", Bs[T.Ramp(kk * {p.bn} + tn * 8, 1, 8)])',
    ]
    for i in range(8):
        lines.append(f'                acc[{i}] = acc[{i}] + T.Broadcast(T.Shuffle([a8], [{i}]), 8) * b8')
    lines += [
        '            T.tvm_storage_sync("shared")',
        '        r0 = rbase + tm * 8',
        '        c0 = cbase + tn * 8',
    ]
    for i in range(8):
        lines.append(f'        C_1[T.Ramp((r0 + {i}) * {p.N} + c0, 1, 8)] = T.Cast("float16x8", acc[{i}])')
    return "\n".join(lines) + "\n"


def _vectorized_func(plan: FragmentPlan, orig: tirx.PrimFunc):
    script = _tiled_fragment_script(plan) if plan.kind == "tiled_fragment_8x8" else _direct_fragment_script(plan)
    mod = parse(script)
    return next(iter(mod.functions.values())).with_attrs(orig.attrs)


def VectorizePrivateFragment():
    """Rewrite canonical OpenCL GEMM_NT 8x8 fragment TIR to vector TIR.

    Placement: after ``UnrollLoop`` and simplification, because only then are all
    private-fragment accesses compile-time constants and safe to scalar-replace.
    """

    @prim_func_pass(opt_level=0)
    def _pass(func: tirx.PrimFunc, _mod: IRModule, _ctx) -> tirx.PrimFunc:
        plan = _detect_plan(func)
        if plan is None:
            return func
        return _vectorized_func(plan, func)

    return _pass
