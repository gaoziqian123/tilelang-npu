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
from tilelang.layout import make_swizzled_layout
import tilelang.language as T


def make_gemm_std_kernel(
    M: int,
    N: int,
    K: int,
    bm: int,
    bn: int,
    bk: int,
    threads: int,
    swizzle: bool = False,
):
    @T.prim_func
    def gemm_std_kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        T.func_attr({"tl.opencl.b_layout": "kn", "tl.opencl.assume_inbounds": 1})

        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            A_sh = T.alloc_shared((bm, bk), "float16")
            B_sh = T.alloc_shared((bk, bn), "float16")
            C_frag = T.alloc_fragment((bm, bn), "float32")

            if swizzle:
                T.annotate_layout(
                    {
                        A_sh: make_swizzled_layout(A_sh),
                        B_sh: make_swizzled_layout(B_sh),
                    }
                )

            for i, j in T.Parallel(bm, bn):
                C_frag[i, j] = 0.0

            for ko in T.serial(K // bk):
                T.copy(A[by * bm, ko * bk], A_sh)
                T.copy(B[ko * bk, bx * bn], B_sh)
                T.gemm(A_sh, B_sh, C_frag, clear_accum=False)

            T.copy(C_frag, C[by * bm, bx * bn])

    return gemm_std_kernel


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
    ap = argparse.ArgumentParser(
        description="Emit a standard TileLang OpenCL GEMM: global->shared T.copy + shared T.gemm + fragment accumulator."
    )
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=2560)
    ap.add_argument("--k", type=int, default=2560)
    ap.add_argument("--bm", type=int, default=32)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--swizzle", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    if args.bm % 8 or args.bn % 8 or args.bk % 4:
        raise SystemExit("standard OpenCL GEMM requires bm%8==0, bn%8==0, bk%4==0")
    if args.threads != (args.bm // 8) * (args.bn // 8):
        raise SystemExit(f"requires threads == (bm/8)*(bn/8) = {(args.bm // 8) * (args.bn // 8)}")
    if args.k % args.bk != 0:
        raise SystemExit("requires K % BK == 0")

    out = args.out
    if out is None:
        suffix = "swz" if args.swizzle else "noswz"
        out = Path(__file__).resolve().parent / "out" / f"gemm_std_{suffix}.cl"

    with tvm.target.Target("opencl"), tvm.transform.PassContext(
        config={"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}
    ):
        artifact = tilelang.lower(
            make_gemm_std_kernel(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.threads, args.swizzle),
            target="opencl",
            enable_device_compile=False,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(artifact.kernel_source, encoding="utf-8")
    print(f"SELECTED_PATH gemm_std swizzle={int(args.swizzle)}")
    print(f"SOURCE_PATH {out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")
    if not args.skip_clang:
        rc = syntax_check(out)
        if rc not in (0, None):
            raise SystemExit(f"clang syntax failed: {rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
