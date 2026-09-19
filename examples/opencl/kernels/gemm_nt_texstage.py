from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def make_texstage_source(M: int, N: int, K: int, bm: int, bn: int, bk: int) -> str:
    row_tiles = bm // 8
    col_tiles = bn // 8
    threads = row_tiles * col_tiles
    return f"""// Function: gemm_nt_texstage_kernel
// B is supplied as an RGBA fp16 image laid out as texel(x=n/4, y=k).
// Each K-block stages B image texels through __local memory: one work-item
// issues one READ_IMAGEH(half4) and one vstore4-equivalent local write group.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TL_M {M}
#define TL_N {N}
#define TL_K {K}
#define TL_BM {bm}
#define TL_BN {bn}
#define TL_BK {bk}
#define TL_WG {threads}
#define TL_NT {col_tiles}

__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
#define READ_IMAGEH(image, sampler, coord) read_imageh(image, sampler, coord)

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

__kernel void gemm_nt_texstage_kernel(__global const half *restrict A,
                                      read_only image2d_t Bi,
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
            const int kk = v / (TL_BN / 4);
            const int c4 = v - kk * (TL_BN / 4);
            const int cc4 = (cbase >> 2) + c4;
            half4 val = (half4)(0.0h);
            if ((cbase + c4 * 4 + 3) < TL_N)
                val = READ_IMAGEH(Bi, smp, (int2)(cc4, kb + kk));
            vstore4(val, 0, &Bs[kk][c4 * 4]);
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


def syntax_check(path: Path) -> int | None:
    clang = subprocess.run(["bash", "-lc", "command -v clang"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
    ap = argparse.ArgumentParser(description="Emit OpenCL GEMM_NT B-texture-to-local-staging kernel")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=32)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out" / "gemm_nt_texstage.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()
    if args.bm % 8 or args.bn % 8 or args.bk % 4 or args.bn % 4:
        raise SystemExit("BM/BN must be divisible by 8, BN/BK by 4")
    expected_threads = (args.bm // 8) * (args.bn // 8)
    if args.threads != expected_threads:
        raise SystemExit(f"threads must be {expected_threads} for BM={args.bm} BN={args.bn}")
    for name, value, tile in (("m", args.m, args.bm), ("n", args.n, args.bn), ("k", args.k, args.bk)):
        if value % tile != 0:
            raise SystemExit(f"{name}={value} must be divisible by tile={tile}")
    src = make_texstage_source(args.m, args.n, args.k, args.bm, args.bn, args.bk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(src, encoding="utf-8")
    print(f"EMIT_OK gemm_nt_texstage M={args.m} N={args.n} K={args.k} BM={args.bm} BN={args.bn} BK={args.bk} threads={args.threads}")
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(src.splitlines())}")
    if not args.skip_clang:
        syntax_check(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
