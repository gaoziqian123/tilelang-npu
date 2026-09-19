from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import tilelang
import tilelang.opencl  # noqa: F401 - registers OpenCL TileOp implementations
from tilelang import tvm
import tilelang.language as T


def make_kernel(M: int, N: int, K: int):
    @T.prim_func
    def gemm_nt_tex_kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N // 4, 4), "float16", scope="global.texture"),
        C: T.Tensor((M, N), "float16"),
    ):
        # One work-item computes an 8x8 C tile.  B is stored as RGBA half4 texels:
        # B[k, n//4, n%4] == original NT weight B_nt[n, k].
        with T.Kernel(N // 8, M // 8, threads=1) as (bx, by):
            acc = T.alloc_fragment((8, 8), "float32")
            avec = T.alloc_fragment((8, 4), "float32")

            for i in T.unroll(8, explicit=True):
                for j in T.unroll(8, explicit=True):
                    acc[i, j] = T.float32(0.0)

            for ko in T.serial(0, K, 4):
                for i in T.unroll(8, explicit=True):
                    for kk in T.unroll(4, explicit=True):
                        avec[i, kk] = T.Cast("float32", A[by * 8 + i, ko + kk])

                for kk in T.unroll(4):
                    for i in T.unroll(8, explicit=True):
                        # Keep the channel axis static; TextureFlatten/CSE emits one
                        # READ_IMAGEH half4 for columns [0..3] and one for [4..7].
                        for c in T.unroll(4, explicit=True):
                            acc[i, c] = acc[i, c] + avec[i, kk] * T.Cast("float32", B[ko + kk, bx * 2, c])
                        for c in T.unroll(4, explicit=True):
                            acc[i, c + 4] = acc[i, c + 4] + avec[i, kk] * T.Cast("float32", B[ko + kk, bx * 2 + 1, c])

            for i in T.unroll(8, explicit=True):
                for j in T.unroll(8, explicit=True):
                    C[by * 8 + i, bx * 8 + j] = T.Cast("float16", acc[i, j])

    return gemm_nt_tex_kernel


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
    ap = argparse.ArgumentParser(description="Emit pure TileLang OpenCL GEMM_NT texture kernel")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out" / "gemm_nt_tex.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    if args.m % 8 or args.n % 8 or args.n % 4 or args.k % 4:
        raise SystemExit("gemm_nt_tex requires M,N divisible by 8 and N,K divisible by 4")

    with tvm.target.Target("opencl"):
        artifact = tilelang.lower(make_kernel(args.m, args.n, args.k), target="opencl", enable_device_compile=False)
    kernel_source = artifact.kernel_source
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(kernel_source, encoding="utf-8")
    print(f"EMIT_OK gemm_nt_tex M={args.m} N={args.n} K={args.k}")
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(kernel_source.splitlines())}")
    if not args.skip_clang:
        syntax_check(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
