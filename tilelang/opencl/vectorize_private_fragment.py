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
from dataclasses import dataclass

from tvm import IRModule, tirx
from tvm.tirx.transform import prim_func_pass


@dataclass(frozen=True)
class FragmentPlan:
    kind: str
    M: int
    N: int
    K: int
    bm: int
    bn: int
    bk: int
    threads: int

    def as_attr(self) -> str:
        return f"kind={self.kind};M={self.M};N={self.N};K={self.K};BM={self.bm};BN={self.bn};BK={self.bk};THREADS={self.threads}"


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


def _buffer_shapes(func: tirx.PrimFunc, src: str | None = None) -> tuple[int, int, int] | None:
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
        return M, N, K
    shapes = [_shape(buf) for buf in buffer_map.values()]
    if any(s is None or len(s) != 2 for s in shapes[:3]):
        return None
    a, b, c = shapes[:3]
    if a[1] != b[1] or c != (a[0], b[0]):
        return None
    return a[0], b[0], a[1]


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
        return None
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
    M, N, K = dims
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
    return FragmentPlan(kind, M, N, K, bm, bn, bk, threads)


def VectorizePrivateFragment():
    """Annotate canonical OpenCL GEMM_NT 8x8 fragment TIR for vector lowering.

    Placement: after ``UnrollLoop`` and simplification, because only then are all
    private-fragment accesses compile-time constants and safe to scalar-replace.
    """

    @prim_func_pass(opt_level=0)
    def _pass(func: tirx.PrimFunc, _mod: IRModule, _ctx) -> tirx.PrimFunc:
        plan = _detect_plan(func)
        if plan is None:
            return func
        return func.with_attr("tl.opencl.vectorize_private_fragment", plan.as_attr())

    return _pass


def parse_fragment_plan_attr(value) -> dict[str, int | str] | None:
    if value is None:
        return None
    text = str(value)
    # tvm StringImm stringifies as the payload in this tirx fork.
    parts: dict[str, int | str] = {}
    for item in text.split(";"):
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        if k == "kind":
            parts[k] = v
        else:
            try:
                parts[k] = int(v)
            except ValueError:
                return None
    req = {"kind", "M", "N", "K", "BM", "BN", "BK", "THREADS"}
    return parts if req.issubset(parts) else None
