from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import tilelang
import tilelang.opencl  # noqa: F401 - registers OpenCL TileOp implementations
from tilelang import tvm
import tilelang.language as T


def make_local64x128_source(M: int, N: int, K: int) -> str:
    return f"""// Function: gemm_nt_kernel_kernel
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TL_M {M}
#define TL_N {N}
#define TL_K {K}
#define TL_WG 128
#define TL_BK 32

#define TL_BLOCK_F32(ACC, AS, BS, TM, TN)                                    \\
    _Pragma(\"unroll 2\")                                                     \\
    for (int kk = 0; kk < TL_BK; kk++) {{                                      \\
        float8 a8 = convert_float8(*(const half8 *)&AS[kk][(TM) * 8]);        \\
        float8 b8 = convert_float8(*(const half8 *)&BS[kk][(TN) * 8]);        \\
        ACC[0] += (float8)a8.s0 * b8; ACC[1] += (float8)a8.s1 * b8;           \\
        ACC[2] += (float8)a8.s2 * b8; ACC[3] += (float8)a8.s3 * b8;           \\
        ACC[4] += (float8)a8.s4 * b8; ACC[5] += (float8)a8.s5 * b8;           \\
        ACC[6] += (float8)a8.s6 * b8; ACC[7] += (float8)a8.s7 * b8;           \\
    }}

__kernel void gemm_nt_kernel_kernel(__global const half *restrict A,
                                    __global const half *restrict B,
                                    __global half *restrict C) {{
    const int bm = get_group_id(1);
    const int bn = get_group_id(0);
    const int lid = get_local_id(0);
    const int tm = lid / 16;
    const int tn = lid & 15;
    const int rbase = bm * 64;
    const int cbase = bn * 128;

    __local half As[TL_BK][64 + 4];
    __local half Bs[TL_BK][128 + 4];
    float8 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = (float8)(0.0f);

    for (int kb = 0; kb < TL_K; kb += TL_BK) {{
        for (int v = lid; v < 64 * TL_BK / 4; v += TL_WG) {{
            const int r = v / (TL_BK / 4);
            const int c4 = v - r * (TL_BK / 4);
            const int rr = rbase + r;
            half4 val = (half4)(0.0h);
            if (rr < TL_M) val = vload4(0, A + (size_t)rr * TL_K + kb + c4 * 4);
            As[c4 * 4 + 0][r] = val.x;
            As[c4 * 4 + 1][r] = val.y;
            As[c4 * 4 + 2][r] = val.z;
            As[c4 * 4 + 3][r] = val.w;
        }}
        for (int v = lid; v < 128 * TL_BK / 4; v += TL_WG) {{
            const int c = v / (TL_BK / 4);
            const int k4 = v - c * (TL_BK / 4);
            const int cc = cbase + c;
            half4 val = (half4)(0.0h);
            if (cc < TL_N) val = vload4(0, B + (size_t)cc * TL_K + kb + k4 * 4);
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
    for (int i = 0; i < 8; ++i) {{
        if (r0 + i < TL_M && c0 + 7 < TL_N)
            vstore8(convert_half8(acc[i]), 0, C + (size_t)(r0 + i) * TL_N + c0);
    }}
}}
"""


def make_direct8x8_source(M: int, N: int, K: int) -> str:
    return f"""// Function: gemm_nt_kernel_kernel
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TL_M {M}
#define TL_N {N}
#define TL_K {K}

__kernel void gemm_nt_kernel_kernel(__global const half *restrict A,
                                    __global const half *restrict B,
                                    __global half *restrict C) {{
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int c0 = gx << 3;
    const int r0 = gy << 3;
    if (c0 >= TL_N || r0 >= TL_M) return;
    float8 c[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = (float8)(0.0f);
    for (int pos = 0; pos < TL_K; pos += 4) {{
        float4 a[8];
        float4 b4[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            a[i] = convert_float4(vload4(0, A + (size_t)(r0 + i) * TL_K + pos));
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            b4[j] = convert_float4(vload4(0, B + (size_t)(c0 + j) * TL_K + pos));
        #pragma unroll
        for (int i = 0; i < 8; ++i) {{
            c[i] += (float8)(a[i].x) * (float8)(b4[0].x, b4[1].x, b4[2].x, b4[3].x, b4[4].x, b4[5].x, b4[6].x, b4[7].x);
            c[i] += (float8)(a[i].y) * (float8)(b4[0].y, b4[1].y, b4[2].y, b4[3].y, b4[4].y, b4[5].y, b4[6].y, b4[7].y);
            c[i] += (float8)(a[i].z) * (float8)(b4[0].z, b4[1].z, b4[2].z, b4[3].z, b4[4].z, b4[5].z, b4[6].z, b4[7].z);
            c[i] += (float8)(a[i].w) * (float8)(b4[0].w, b4[1].w, b4[2].w, b4[3].w, b4[4].w, b4[5].w, b4[6].w, b4[7].w);
        }}
    }}
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        if (r0 + i < TL_M && c0 + 7 < TL_N)
            vstore8(convert_half8(c[i]), 0, C + (size_t)(r0 + i) * TL_N + c0);
}}
"""


def make_kernel(M: int, N: int, K: int, bm: int, bn: int, bk: int, threads: int):
    @T.prim_func
    def gemm_nt_kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        # B is NT layout: B[n, k], so C = A @ B^T.
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((bm, bk), "float16")
            B_shared = T.alloc_shared((bn, bk), "float16")
            C_accum = T.alloc_shared((bm, bn), "float32")

            T.clear(C_accum)
            for ko in T.serial(T.ceildiv(K, bk)):
                T.copy(A[by * bm, ko * bk], A_shared)
                T.copy(B[bx * bn, ko * bk], B_shared)
                T.sync_threads()
                T.gemm(A_shared, B_shared, C_accum, transpose_B=True)
                T.sync_threads()
            T.copy(C_accum, C[by * bm, bx * bn])

    return gemm_nt_kernel


def syntax_check(path: Path) -> int | None:
    clang = subprocess.run(
        ["bash", "-lc", "command -v clang"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if clang.returncode != 0:
        print("CLANG_SYNTAX_SKIP no clang in PATH")
        return None
    headers = Path("/root/project/backend/gpu/OpenCL-Headers")
    cmd = [clang.stdout.strip(), "-x", "cl", "-cl-std=CL3.0", "-fsyntax-only"]
    if headers.exists():
        cmd += ["-I", str(headers)]
    cmd.append(str(path))
    res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(res.stdout, end="")
    print(f"CLANG_SYNTAX_RC {res.returncode}")
    return res.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit TileLang OpenCL GEMM_NT kernel")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--bm", type=int, default=16)
    ap.add_argument("--bn", type=int, default=16)
    ap.add_argument("--bk", type=int, default=16)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--impl", choices=("tilelang", "local64x128", "direct8x8"), default="tilelang")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("out") / "gemm_nt.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    for name, value, tile in (("m", args.m, args.bm), ("n", args.n, args.bn), ("k", args.k, args.bk)):
        if value % tile != 0:
            raise SystemExit(f"{name}={value} must be divisible by tile={tile}")

    if args.impl == "local64x128":
        if args.bm != 64 or args.bn != 128 or args.bk != 32 or args.threads != 128:
            raise SystemExit("local64x128 requires --bm 64 --bn 128 --bk 32 --threads 128")
        kernel_source = make_local64x128_source(args.m, args.n, args.k)
    elif args.impl == "direct8x8":
        if args.bm != 8 or args.bn != 8 or args.threads != 1:
            raise SystemExit("direct8x8 requires --bm 8 --bn 8 --threads 1")
        kernel_source = make_direct8x8_source(args.m, args.n, args.k)
    else:
        with tvm.target.Target("opencl"):
            artifact = tilelang.lower(
                make_kernel(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.threads),
                target="opencl",
                enable_device_compile=False,
            )
        kernel_source = artifact.kernel_source
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(kernel_source, encoding="utf-8")
    print(
        f"EMIT_OK gemm_nt M={args.m} N={args.n} K={args.k} "
        f"BM={args.bm} BN={args.bn} BK={args.bk} threads={args.threads}"
    )
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(kernel_source.splitlines())}")
    print(f"SELECTED_PATH {args.impl}")
    if not args.skip_clang:
        syntax_check(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
