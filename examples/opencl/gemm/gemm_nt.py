from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import tilelang
import tilelang.opencl  # noqa: F401 - registers OpenCL TileOp implementations
from tilelang import tvm
import tilelang.language as T


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
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("out") / "gemm_nt.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    for name, value, tile in (("m", args.m, args.bm), ("n", args.n, args.bn), ("k", args.k, args.bk)):
        if value % tile != 0:
            raise SystemExit(f"{name}={value} must be divisible by tile={tile}")

    with tvm.target.Target("opencl"):
        artifact = tilelang.lower(
            make_kernel(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.threads),
            target="opencl",
            enable_device_compile=False,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(artifact.kernel_source, encoding="utf-8")
    print(
        f"EMIT_OK gemm_nt M={args.m} N={args.n} K={args.k} "
        f"BM={args.bm} BN={args.bn} BK={args.bk} threads={args.threads}"
    )
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")
    print("SELECTED_PATH T.copy+T.gemm transpose_B=True")
    if not args.skip_clang:
        syntax_check(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
