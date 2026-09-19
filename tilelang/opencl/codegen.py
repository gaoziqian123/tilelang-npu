from __future__ import annotations

import re

from tilelang.backend.device_codegen import global_func_device_codegen
from tilelang.opencl.vectorize_private_fragment import parse_fragment_plan_attr

_raw_build_opencl = global_func_device_codegen("target.build.opencl")


class _SourceModule:
    def __init__(self, mod, source: str):
        self._mod = mod
        self._source = source

    def inspect_source(self, *args, **kwargs):
        return self._source

    def __getattr__(self, name):
        return getattr(self._mod, name)


def _find_fragment_plan(mod) -> dict[str, int | str] | None:
    for func in mod.functions.values():
        attrs = getattr(func, "attrs", None)
        if not attrs or "tl.opencl.vectorize_private_fragment" not in attrs:
            continue
        plan = parse_fragment_plan_attr(attrs["tl.opencl.vectorize_private_fragment"])
        if plan is not None:
            return plan
    return None


def _patch_opencl_private_accumulators(source: str) -> str:
    """Fallback SROA for legacy texture-GEMM accumulator arrays.

    The formal OpenCL fragment pass below covers the direct-global and
    shared-staged GEMM_NT fragment kernels.  Texture GEMM still reaches legacy
    OpenCL codegen with a fully-unrolled private ``float acc[64]`` whose indices
    are constant but whose texture loads are not part of the fragment matcher.
    Keep this narrow fallback until texture-scope fragment TIR is normalized to
    the same shape as buffer GEMM; it is deliberately limited to acc* arrays and
    reverts if any non-constant access survives.
    """

    original_source = source
    patched_accumulators: set[str] = set()

    def repl_decl(match: re.Match[str]) -> str:
        indent = match.group(1)
        name = match.group(2)
        n = int(match.group(3))
        if n <= 1 or n % 8 != 0 or n > 128:
            return match.group(0)
        patched_accumulators.add(name)
        return "\n".join(f"{indent}float8 {name}_{i} = (float8)(0.000000e+00f);" for i in range(n // 8))

    source = re.sub(r"(?m)^(\s*)float\s+(acc\w*)\[(\d+)\];\s*$", repl_decl, source)

    def repl_acc(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in patched_accumulators:
            return match.group(0)
        idx = int(match.group(2))
        return f"{name}_{idx // 8}.s{idx % 8}"

    source = re.sub(r"\b(acc\w*)\[(\d+)\]", repl_acc, source)

    for name in patched_accumulators:
        if re.search(rf"\b{re.escape(name)}\[", source):
            return original_source

    source = re.sub(
        r"\(\(half\s*\*\)&([A-Za-z_]\w*)\)\[(\d)\]",
        lambda m: f"{m.group(1)}.s{m.group(2)}",
        source,
    )
    return source


def _opencl_header() -> str:
    return """// Function: gemm_nt_kernel_kernel
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

"""


def _render_fragment_8x8(p: dict[str, int | str]) -> str:
    M = int(p["M"]); N = int(p["N"]); K = int(p["K"])
    BM = int(p["BM"]); BN = int(p["BN"])
    tiles_n = BN // 8
    lines: list[str] = [_opencl_header()]
    lines.append("#define TL_M %d\n#define TL_N %d\n#define TL_K %d\n\n" % (M, N, K))
    lines.append("__kernel void gemm_nt_kernel_kernel(__global half *restrict A,\n")
    lines.append("                                    __global half *restrict B,\n")
    lines.append("                                    __global half *restrict C) {\n")
    lines.append("  const int lid = convert_int(get_local_id(0));\n")
    lines.append("  const int bx = convert_int(get_group_id(0));\n")
    lines.append("  const int by = convert_int(get_group_id(1));\n")
    lines.append(f"  const int tm = lid / {tiles_n};\n")
    lines.append(f"  const int tn = lid - tm * {tiles_n};\n")
    lines.append(f"  const int r0 = by * {BM} + tm * 8;\n")
    lines.append(f"  const int c0 = bx * {BN} + tn * 8;\n")
    for i in range(8):
        lines.append(f"  float8 acc_{i} = (float8)(0.0f);\n")
    lines.append(f"  for (int pos = 0; pos < {K}; pos += 4) {{\n")
    for i in range(8):
        lines.append(f"    const float4 a{i} = convert_float4(vload4(0, A + (size_t)(r0 + {i}) * {K} + pos));\n")
    for j in range(8):
        lines.append(f"    const float4 b{j} = convert_float4(vload4(0, B + (size_t)(c0 + {j}) * {K} + pos));\n")
    for i in range(8):
        for comp in "xyzw":
            bvec = ", ".join(f"b{j}.{comp}" for j in range(8))
            lines.append(f"    acc_{i} += (float8)(a{i}.{comp}) * (float8)({bvec});\n")
    lines.append("  }\n")
    for i in range(8):
        lines.append(f"  vstore8(convert_half8(acc_{i}), 0, C + (size_t)(r0 + {i}) * {N} + c0);\n")
    lines.append("}\n")
    return "".join(lines)


def _render_tiled_fragment_8x8(p: dict[str, int | str]) -> str:
    N = int(p["N"]); K = int(p["K"]); BM = int(p["BM"]); BN = int(p["BN"]); BK = int(p["BK"])
    threads = int(p["THREADS"]); col_tiles = BN // 8
    return f"""// Function: gemm_nt_kernel_kernel
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TL_N {N}
#define TL_K {K}
#define TL_BM {BM}
#define TL_BN {BN}
#define TL_BK {BK}
#define TL_WG {threads}
#define TL_NT {col_tiles}

#define TL_BLOCK_F32(ACC, AS, BS, TM, TN)                                    \\
    _Pragma("unroll 2")                                                     \\
    for (int kk = 0; kk < TL_BK; kk++) {{                                      \\
        float8 a8 = convert_float8(*(const half8 *)&AS[kk][(TM) * 8]);        \\
        float8 b8 = convert_float8(*(const half8 *)&BS[kk][(TN) * 8]);        \\
        ACC[0] += (float8)a8.s0 * b8; ACC[1] += (float8)a8.s1 * b8;           \\
        ACC[2] += (float8)a8.s2 * b8; ACC[3] += (float8)a8.s3 * b8;           \\
        ACC[4] += (float8)a8.s4 * b8; ACC[5] += (float8)a8.s5 * b8;           \\
        ACC[6] += (float8)a8.s6 * b8; ACC[7] += (float8)a8.s7 * b8;           \\
    }}

__kernel void gemm_nt_kernel_kernel(__global half *restrict A,
                                    __global half *restrict B,
                                    __global half *restrict C) {{
    const int bm = get_group_id(1);
    const int bn = get_group_id(0);
    const int lid = get_local_id(0);
    const int tm = lid / TL_NT;
    const int tn = lid - tm * TL_NT;
    const int rbase = bm * TL_BM;
    const int cbase = bn * TL_BN;

    __local half As[TL_BK][TL_BM + 4];
    __local half Bs[TL_BK][TL_BN + 4];
    float8 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = (float8)(0.0f);

    for (int kb = 0; kb < TL_K; kb += TL_BK) {{
        for (int v = lid; v < TL_BM * TL_BK / 4; v += TL_WG) {{
            const int r = v / (TL_BK / 4);
            const int c4 = v - r * (TL_BK / 4);
            const half4 val = vload4(0, A + (size_t)(rbase + r) * TL_K + kb + c4 * 4);
            As[c4 * 4 + 0][r] = val.x;
            As[c4 * 4 + 1][r] = val.y;
            As[c4 * 4 + 2][r] = val.z;
            As[c4 * 4 + 3][r] = val.w;
        }}
        for (int v = lid; v < TL_BN * TL_BK / 4; v += TL_WG) {{
            const int c = v / (TL_BK / 4);
            const int k4 = v - c * (TL_BK / 4);
            const half4 val = vload4(0, B + (size_t)(cbase + c) * TL_K + kb + k4 * 4);
            Bs[k4 * 4 + 0][c] = val.x;
            Bs[k4 * 4 + 1][c] = val.y;
            Bs[k4 * 4 + 2][c] = val.z;
            Bs[k4 * 4 + 3][c] = val.w;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        TL_BLOCK_F32(acc, As, Bs, tm, tn);
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    const int r0 = rbase + tm * 8;
    const int c0 = cbase + tn * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        vstore8(convert_half8(acc[i]), 0, C + (size_t)(r0 + i) * TL_N + c0);
}}
"""


def _render_fragment_plan(plan: dict[str, int | str]) -> str | None:
    kind = plan.get("kind")
    if kind == "fragment_8x8":
        return _render_fragment_8x8(plan)
    if kind == "tiled_fragment_8x8":
        return _render_tiled_fragment_8x8(plan)
    return None


def build_opencl(mod, target):
    built = _raw_build_opencl(mod, target)
    plan = _find_fragment_plan(mod)
    if plan is None:
        source = built.inspect_source()
        patched = _patch_opencl_private_accumulators(source)
        if patched != source:
            return _SourceModule(built, patched)
        return built
    source = _render_fragment_plan(plan)
    if source is None:
        return built
    return _SourceModule(built, source)
