from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilelang
import tilelang.opencl  # noqa: F401 - registers OpenCL TileOp implementations
from tilelang import tvm
import tilelang.language as T


def make_local_tiled_source(M: int, N: int, K: int, bm: int, bn: int, bk: int) -> str:
    row_tiles = bm // 8
    col_tiles = bn // 8
    threads = row_tiles * col_tiles
    return f"""// Function: gemm_nt_kernel_kernel
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TL_M {M}
#define TL_N {N}
#define TL_K {K}
#define TL_BM {bm}
#define TL_BN {bn}
#define TL_BK {bk}
#define TL_WG {threads}
#define TL_NT {col_tiles}

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
            const int rr = rbase + r;
            half4 val = (half4)(0.0h);
            if (rr < TL_M) val = vload4(0, A + (size_t)rr * TL_K + kb + c4 * 4);
            As[c4 * 4 + 0][r] = val.x;
            As[c4 * 4 + 1][r] = val.y;
            As[c4 * 4 + 2][r] = val.z;
            As[c4 * 4 + 3][r] = val.w;
        }}
        for (int v = lid; v < TL_BN * TL_BK / 4; v += TL_WG) {{
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


def make_local64x128_source(M: int, N: int, K: int, bk: int = 64) -> str:
    return make_local_tiled_source(M, N, K, 64, 128, bk)


def make_direct8x8_source(M: int, N: int, K: int, b_layout: str = "nk") -> str:
    if b_layout == "kn":
        b_decl = "float8 b[4];"
        b_loads = """        #pragma unroll
        for (int i = 0; i < 4; ++i)
            b[i] = convert_float8(vload8(0, B + (size_t)(pos + i) * TL_N + c0));"""
        macs = """            c[i] += (float8)(a[i].x) * b[0];
            c[i] += (float8)(a[i].y) * b[1];
            c[i] += (float8)(a[i].z) * b[2];
            c[i] += (float8)(a[i].w) * b[3];"""
    else:
        b_decl = "float4 b4[8];"
        b_loads = """        #pragma unroll
        for (int j = 0; j < 8; ++j)
            b4[j] = convert_float4(vload4(0, B + (size_t)(c0 + j) * TL_K + pos));"""
        macs = """            c[i] += (float8)(a[i].x) * (float8)(b4[0].x, b4[1].x, b4[2].x, b4[3].x, b4[4].x, b4[5].x, b4[6].x, b4[7].x);
            c[i] += (float8)(a[i].y) * (float8)(b4[0].y, b4[1].y, b4[2].y, b4[3].y, b4[4].y, b4[5].y, b4[6].y, b4[7].y);
            c[i] += (float8)(a[i].z) * (float8)(b4[0].z, b4[1].z, b4[2].z, b4[3].z, b4[4].z, b4[5].z, b4[6].z, b4[7].z);
            c[i] += (float8)(a[i].w) * (float8)(b4[0].w, b4[1].w, b4[2].w, b4[3].w, b4[4].w, b4[5].w, b4[6].w, b4[7].w);"""
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
        {b_decl}
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            a[i] = convert_float4(vload4(0, A + (size_t)(r0 + i) * TL_K + pos));
{b_loads}
        #pragma unroll
        for (int i = 0; i < 8; ++i) {{
{macs}
        }}
    }}
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        if (r0 + i < TL_M && c0 + 7 < TL_N)
            vstore8(convert_half8(c[i]), 0, C + (size_t)(r0 + i) * TL_N + c0);
}}
"""


def make_rrgemm_source(M: int, N: int, K: int, bm: int, bn: int, bk: int, threads: int, b_layout: str = "kn") -> str:
    row_tiles = bm // 8
    col_tiles = bn // 8
    if b_layout == "kn":
        b_decl = "float8 b[4];"
        b_loads = f"""        #pragma unroll
        for (int i = 0; i < 4; ++i)
            b[i] = convert_float8(vload8(0, B + (size_t)(pos + i) * TL_N + c0));"""
        macs = """            acc[i] += (float8)(a[i].x) * b[0];
            acc[i] += (float8)(a[i].y) * b[1];
            acc[i] += (float8)(a[i].z) * b[2];
            acc[i] += (float8)(a[i].w) * b[3];"""
    else:
        b_decl = "float4 b4[8];"
        b_loads = """        #pragma unroll
        for (int j = 0; j < 8; ++j)
            b4[j] = convert_float4(vload4(0, B + (size_t)(c0 + j) * TL_K + pos));"""
        macs = """            acc[i] += (float8)(a[i].x) * (float8)(b4[0].x, b4[1].x, b4[2].x, b4[3].x, b4[4].x, b4[5].x, b4[6].x, b4[7].x);
            acc[i] += (float8)(a[i].y) * (float8)(b4[0].y, b4[1].y, b4[2].y, b4[3].y, b4[4].y, b4[5].y, b4[6].y, b4[7].y);
            acc[i] += (float8)(a[i].z) * (float8)(b4[0].z, b4[1].z, b4[2].z, b4[3].z, b4[4].z, b4[5].z, b4[6].z, b4[7].z);
            acc[i] += (float8)(a[i].w) * (float8)(b4[0].w, b4[1].w, b4[2].w, b4[3].w, b4[4].w, b4[5].w, b4[6].w, b4[7].w);"""
    return f"""// Function: gemm_nt_kernel_kernel
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TL_M {M}
#define TL_N {N}
#define TL_K {K}
#define TL_BM {bm}
#define TL_BN {bn}
#define TL_BK {bk}
#define TL_WG {threads}
#define TL_NT {col_tiles}

__kernel void gemm_nt_kernel_kernel(__global const half *restrict A,
                                    __global const half *restrict B,
                                    __global half *restrict C) {{
    const int bx = get_group_id(0);
    const int by = get_group_id(1);
    const int tx = get_local_id(0);
    const int tm = tx / TL_NT;
    const int tn = tx - tm * TL_NT;
    const int r0 = by * TL_BM + tm * 8;
    const int c0 = bx * TL_BN + tn * 8;
    float8 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = (float8)(0.0f);
    for (int kb = 0; kb < TL_K; kb += TL_BK) {{
        for (int pos = kb; pos < kb + TL_BK; pos += 4) {{
            float4 a[8];
            {b_decl}
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                a[i] = convert_float4(vload4(0, A + (size_t)(r0 + i) * TL_K + pos));
{b_loads}
            #pragma unroll
            for (int i = 0; i < 8; ++i) {{
{macs}
            }}
        }}
    }}
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        if (r0 + i < TL_M && c0 + 7 < TL_N)
            vstore8(convert_half8(acc[i]), 0, C + (size_t)(r0 + i) * TL_N + c0);
}}
"""


def make_image8x8_source(M: int, N: int, K: int) -> str:
    return f"""// Function: hgemm_8x8
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TL_M {M}
#define TL_N {N}
#define TL_K {K}

__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void hgemm_8x8(__global const half *A, int lda,
                        __global half *C, int ldc,
                        int m, int n, int k, read_only image2d_t Bi) {{
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    if (((gx << 3) < n) && ((gy << 3) < m)) {{
        float4 a[8];
        float8 b[4];
        float8 c[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) c[i] = (float8)(0.0f);
        const int a_y_off = (gy << 3) * lda;
        for (int pos = 0; pos < k; pos += 4) {{
            #pragma unroll
            for (int i = 0; i < 4; ++i) {{
                const half4 b_lo = read_imageh(Bi, smp, (int2)(gx << 1, pos + i));
                const half4 b_hi = read_imageh(Bi, smp, (int2)((gx << 1) + 1, pos + i));
                b[i] = convert_float8((half8)(b_lo, b_hi));
            }}
            int a_off = a_y_off + pos;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {{
                a[i] = convert_float4(vload4(0, A + a_off));
                a_off += lda;
            }}
            #pragma unroll
            for (int i = 0; i < 8; ++i) {{
                c[i] += (float8)(a[i].x) * b[0];
                c[i] += (float8)(a[i].y) * b[1];
                c[i] += (float8)(a[i].z) * b[2];
                c[i] += (float8)(a[i].w) * b[3];
            }}
        }}
        #pragma unroll
        for (int i = 0; i < 8; ++i) {{
            const int c_off = ((gy << 3) + i) * ldc + (gx << 3);
            vstore8(convert_half8(c[i]), 0, C + c_off);
        }}
    }}
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


def make_fragment_kernel(M: int, N: int, K: int, bm: int, bn: int, bk: int, threads: int, b_layout: str = "nk"):
    tile_m = 8
    tile_n = 8

    @T.prim_func
    def gemm_nt_kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N) if b_layout == "kn" else (N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        T.func_attr({"tl.opencl.b_layout": b_layout})
        # Pure-IR direct-global GEMM_NT: each work-item owns one 8x8 C tile.
        # There is intentionally no alloc_shared/T.copy/T.sync_threads here; the
        # accumulator lives in local.fragment (OpenCL private/register scope).
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            tx = T.get_thread_binding(0)
            tiles_n = bn // tile_n
            tm = tx // tiles_n
            tn = tx - tm * tiles_n
            acc = T.alloc_fragment((tile_m, tile_n), "float32")

            for ii in T.unroll(tile_m, explicit=True):
                for jj in T.unroll(tile_n, explicit=True):
                    acc[ii, jj] = T.float32(0.0)

            for ko in T.serial(T.ceildiv(K, bk)):
                for kk in T.serial(bk):
                    kidx = ko * bk + kk
                    for ii in T.unroll(tile_m, explicit=True):
                        for jj in T.unroll(tile_n, explicit=True):
                            if b_layout == "kn":
                                acc[ii, jj] += T.cast(A[by * bm + tm * tile_m + ii, kidx], "float32") * T.cast(
                                    B[kidx, bx * bn + tn * tile_n + jj], "float32"
                                )
                            else:
                                acc[ii, jj] += T.cast(A[by * bm + tm * tile_m + ii, kidx], "float32") * T.cast(
                                    B[bx * bn + tn * tile_n + jj, kidx], "float32"
                                )

            for ii in T.unroll(tile_m, explicit=True):
                for jj in T.unroll(tile_n, explicit=True):
                    C[by * bm + tm * tile_m + ii, bx * bn + tn * tile_n + jj] = T.cast(acc[ii, jj], "float16")

    return gemm_nt_kernel


def make_tiled_fragment_kernel(M: int, N: int, K: int, bm: int, bn: int, bk: int, threads: int, b_layout: str = "nk"):
    tile_m = 8
    tile_n = 8

    @T.prim_func
    def gemm_nt_kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N) if b_layout == "kn" else (N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        T.func_attr({"tl.opencl.b_layout": b_layout})
        # Pure-IR tiled GEMM_NT: cooperatively stage A/B K tiles in OpenCL
        # shared(local) memory, while each work-item owns one private 8x8
        # fp32 fragment accumulator and writes one 8x8 output tile.
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            tx = T.get_thread_binding(0)
            tiles_n = bn // tile_n
            tm = tx // tiles_n
            tn = tx - tm * tiles_n
            A_shared = T.alloc_shared((bm, bk), "float16")
            B_shared = T.alloc_shared((bk, bn) if b_layout == "kn" else (bn, bk), "float16")
            acc = T.alloc_fragment((tile_m, tile_n), "float32")

            for ii in T.unroll(tile_m, explicit=True):
                for jj in T.unroll(tile_n, explicit=True):
                    acc[ii, jj] = T.float32(0.0)

            for ko in T.serial(T.ceildiv(K, bk)):
                T.copy(A[by * bm, ko * bk], A_shared)
                if b_layout == "kn":
                    T.copy(B[ko * bk, bx * bn], B_shared)
                else:
                    T.copy(B[bx * bn, ko * bk], B_shared)
                T.sync_threads()
                if b_layout == "kn":
                    T.gemm(A_shared[tm * tile_m, 0], B_shared[0, tn * tile_n], acc)
                else:
                    T.gemm(A_shared[tm * tile_m, 0], B_shared[tn * tile_n, 0], acc, transpose_B=True)
                T.sync_threads()

            for ii in T.unroll(tile_m, explicit=True):
                for jj in T.unroll(tile_n, explicit=True):
                    C[by * bm + tm * tile_m + ii, bx * bn + tn * tile_n + jj] = T.cast(acc[ii, jj], "float16")

    return gemm_nt_kernel


def make_rrgemm_kernel(M: int, N: int, K: int, bm: int, bn: int, bk: int, threads: int, b_layout: str = "kn", accum_dtype: str = "float32"):
    @T.prim_func
    def gemm_nt_kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N) if b_layout == "kn" else (N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        T.func_attr({"tl.opencl.b_layout": b_layout})
        # RR GEMM: both input operands and the accumulator are OpenCL fragments.
        # GemmFMA.infer_layout supplies the full-tile fragment layouts; its RR
        # lower emits 8 float32x8 (or float16x8 when accum_dtype=float16)
        # accumulators per work-item directly.
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            A_frag = T.alloc_fragment((bm, bk), "float16")
            B_frag = T.alloc_fragment((bk, bn) if b_layout == "kn" else (bn, bk), "float16")
            C_frag = T.alloc_fragment((bm, bn), accum_dtype)

            for ko in T.serial(T.ceildiv(K, bk)):
                T.copy(A[by * bm, ko * bk], A_frag)
                if b_layout == "kn":
                    T.copy(B[ko * bk, bx * bn], B_frag)
                    T.gemm(A_frag, B_frag, C_frag, clear_accum=(ko == 0))
                else:
                    T.copy(B[bx * bn, ko * bk], B_frag)
                    T.gemm(A_frag, B_frag, C_frag, transpose_B=True, clear_accum=(ko == 0))
            T.copy(C_frag, C[by * bm, bx * bn])

    return gemm_nt_kernel


def make_grgemm_kernel(M: int, N: int, K: int, bm: int, bn: int, bk: int, threads: int, b_layout: str = "kn", accum_dtype: str = "float32"):
    @T.prim_func
    def gemm_nt_kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N) if b_layout == "kn" else (N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        T.func_attr({"tl.opencl.b_layout": b_layout, "tl.opencl.assume_inbounds": 1})
        # GR GEMM: A/B tiles are read straight from global memory by the
        # GemmFMA vector lower; only the 8x8-per-work-item accumulator lives in
        # a fragment.  Staging replicated A/B into private fragments forces a
        # private-memory round trip per FMA on Adreno (measured <=0.043T),
        # while direct global vector loads reach the 1.35T anchor shape.
        # Single full-K T.gemm: point-indexed operands follow the Hexagon
        # "buffer + base offset" convention (extent = trailing buffer shape).
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            K  # keep K in the closure: the conditional B annotation needs it
            C_frag = T.alloc_fragment((bm, bn), accum_dtype)
            if b_layout == "kn":
                T.gemm(A[by * bm, 0], B[0, bx * bn], C_frag, clear_accum=True)
            else:
                T.gemm(A[by * bm, 0], B[bx * bn, 0], C_frag, transpose_B=True, clear_accum=True)
            T.copy(C_frag, C[by * bm, bx * bn])

    return gemm_nt_kernel


def syntax_check(path: Path) -> int | None:
    clang_bin = os.environ.get("CLANG_BIN")
    if clang_bin:
        clang_path = clang_bin if Path(clang_bin).exists() else None
    else:
        clang = subprocess.run(
            ["bash", "-lc", "command -v clang"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        clang_path = clang.stdout.strip() if clang.returncode == 0 else None
    if not clang_path:
        print("CLANG_SYNTAX_SKIP no clang in PATH (set CLANG_BIN)")
        return None
    headers = Path("/root/project/backend/gpu/OpenCL-Headers")
    cmd = [clang_path, "-x", "cl", "-cl-std=CL3.0", "-fsyntax-only"]
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
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--impl", choices=("tilelang", "fragment", "tiled_fragment", "rrgemm", "grgemm", "local64x128", "local_tiled", "direct8x8", "image8x8"), default="local64x128")
    ap.add_argument("--b-layout", choices=("nk", "kn"), default="nk", help="B buffer layout for fragment/tiled_fragment/rrgemm/grgemm/direct8x8: nk=B[N,K], kn=pretransposed Bt[K,N]")
    ap.add_argument("--accum", choices=("fp32", "fp16"), default="fp32", help="rrgemm/grgemm accumulator dtype: fp32 inner chain or full-fp16 (no converts)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "gemm_nt.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    for name, value, tile in (("m", args.m, args.bm), ("n", args.n, args.bn), ("k", args.k, args.bk)):
        if value % tile != 0:
            raise SystemExit(f"{name}={value} must be divisible by tile={tile}")

    if args.impl == "local64x128":
        if args.bm != 64 or args.bn != 128 or args.threads != 128:
            raise SystemExit("local64x128 requires --bm 64 --bn 128 --threads 128")
        if args.bk % 4 != 0:
            raise SystemExit("local64x128 requires --bk divisible by 4")
        kernel_source = make_local64x128_source(args.m, args.n, args.k, args.bk)
    elif args.impl == "local_tiled":
        if args.bm % 8 or args.bn % 8 or args.bk % 4:
            raise SystemExit("local_tiled requires --bm/--bn divisible by 8 and --bk divisible by 4")
        expected_threads = (args.bm // 8) * (args.bn // 8)
        if args.threads != expected_threads:
            raise SystemExit(f"local_tiled requires --threads {expected_threads} for BM={args.bm} BN={args.bn}")
        kernel_source = make_local_tiled_source(args.m, args.n, args.k, args.bm, args.bn, args.bk)
    elif args.impl == "direct8x8":
        if args.bm != 8 or args.bn != 8 or args.threads != 1:
            raise SystemExit("direct8x8 requires --bm 8 --bn 8 --threads 1")
        kernel_source = make_direct8x8_source(args.m, args.n, args.k, args.b_layout)
    elif args.impl == "image8x8":
        if args.bm != 8 or args.bn != 8 or args.threads != 1:
            raise SystemExit("image8x8 requires --bm 8 --bn 8 --threads 1")
        if args.k % 4 != 0:
            raise SystemExit("image8x8 requires --k divisible by 4")
        kernel_source = make_image8x8_source(args.m, args.n, args.k)
    elif args.impl == "fragment":
        if args.bm % 8 or args.bn % 8 or args.bk <= 0:
            raise SystemExit("fragment requires --bm/--bn divisible by 8 and positive --bk")
        expected_threads = (args.bm // 8) * (args.bn // 8)
        if args.threads != expected_threads:
            raise SystemExit(f"fragment requires --threads {expected_threads} for 8x8/thread BM={args.bm} BN={args.bn}")
        old_b_layout = os.environ.get("TL_OPENCL_B_LAYOUT")
        os.environ["TL_OPENCL_B_LAYOUT"] = args.b_layout
        try:
            with tvm.target.Target("opencl"):
                artifact = tilelang.lower(
                    make_fragment_kernel(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.threads, args.b_layout),
                    target="opencl",
                    enable_device_compile=False,
                )
        finally:
            if old_b_layout is None:
                os.environ.pop("TL_OPENCL_B_LAYOUT", None)
            else:
                os.environ["TL_OPENCL_B_LAYOUT"] = old_b_layout
        kernel_source = artifact.kernel_source
    elif args.impl == "tiled_fragment":
        if args.bm % 8 or args.bn % 8 or args.bk <= 0:
            raise SystemExit("tiled_fragment requires --bm/--bn divisible by 8 and positive --bk")
        expected_threads = (args.bm // 8) * (args.bn // 8)
        if args.threads != expected_threads:
            raise SystemExit(f"tiled_fragment requires --threads {expected_threads} for 8x8/thread BM={args.bm} BN={args.bn}")
        old_b_layout = os.environ.get("TL_OPENCL_B_LAYOUT")
        os.environ["TL_OPENCL_B_LAYOUT"] = args.b_layout
        try:
            with tvm.target.Target("opencl"):
                artifact = tilelang.lower(
                    make_tiled_fragment_kernel(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.threads, args.b_layout),
                    target="opencl",
                    enable_device_compile=False,
                )
        finally:
            if old_b_layout is None:
                os.environ.pop("TL_OPENCL_B_LAYOUT", None)
            else:
                os.environ["TL_OPENCL_B_LAYOUT"] = old_b_layout
        kernel_source = artifact.kernel_source
    elif args.impl == "rrgemm":
        if args.bm % 8 or args.bn % 8 or args.bk % 4:
            raise SystemExit("rrgemm requires --bm/--bn divisible by 8 and --bk divisible by 4")
        expected_threads = (args.bm // 8) * (args.bn // 8)
        if args.threads != expected_threads:
            raise SystemExit(f"rrgemm requires --threads {expected_threads} for 8x8/thread BM={args.bm} BN={args.bn}")
        accum_dtype = "float16" if args.accum == "fp16" else "float32"
        try:
            # RR fragments must fully unroll (explicit kUnrolled loops + loops
            # touching local buffers) so private arrays SROA into registers.
            with tvm.target.Target("opencl"), tvm.transform.PassContext(
                config={"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}
            ):
                artifact = tilelang.lower(
                    make_rrgemm_kernel(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.threads, args.b_layout, accum_dtype),
                    target="opencl",
                    enable_device_compile=False,
                )
            kernel_source = artifact.kernel_source
        except Exception:
            if args.accum == "fp16":
                raise
            # fp32 lowering is still blocked by the DecoupleTypeCast staging
            # allocation bug on the C store; fall back to the hand-written
            # fp32-inner-chain source until that is fixed.
            print("RRGEMM_LOWER_FALLBACK fp32 lowering failed; using make_rrgemm_source", file=sys.stderr)
            kernel_source = make_rrgemm_source(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.threads, args.b_layout)
    elif args.impl == "grgemm":
        if args.bm % 8 or args.bn % 8 or args.bk % 4:
            raise SystemExit("grgemm requires --bm/--bn divisible by 8 and --bk divisible by 4")
        expected_threads = (args.bm // 8) * (args.bn // 8)
        if args.threads != expected_threads:
            raise SystemExit(f"grgemm requires --threads {expected_threads} for 8x8/thread BM={args.bm} BN={args.bn}")
        accum_dtype = "float16" if args.accum == "fp16" else "float32"
        with tvm.target.Target("opencl"), tvm.transform.PassContext(
            config={"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}
        ):
            artifact = tilelang.lower(
                make_grgemm_kernel(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.threads, args.b_layout, accum_dtype),
                target="opencl",
                enable_device_compile=False,
            )
        kernel_source = artifact.kernel_source
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
