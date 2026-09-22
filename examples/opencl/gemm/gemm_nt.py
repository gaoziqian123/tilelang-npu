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
    ap = argparse.ArgumentParser(
        description="Emit the TileLang GR GEMM_NT OpenCL kernel (direct-global fp16 mad lower).")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--bm", type=int, default=32)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--b-layout", choices=("kn", "nk"), default="kn")
    ap.add_argument("--accum", choices=("fp32", "fp16"), default="fp16")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "gemm_nt.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    if args.bm % 8 or args.bn % 8 or args.bk % 4:
        raise SystemExit("grgemm requires bm%8==0, bn%8==0, bk%4==0")
    if args.threads != (args.bm // 8) * (args.bn // 8):
        raise SystemExit(f"grgemm requires threads == (bm/8)*(bn/8) = {(args.bm // 8) * (args.bn // 8)}")

    accum_dtype = "float16" if args.accum == "fp16" else "float32"
    with tvm.target.Target("opencl"), tvm.transform.PassContext(
        config={"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}
    ):
        artifact = tilelang.lower(
            make_grgemm_kernel(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.threads,
                               args.b_layout, accum_dtype),
            target="opencl",
            enable_device_compile=False,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(artifact.kernel_source, encoding="utf-8")
    print(f"SELECTED_PATH grgemm")
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")
    if not args.skip_clang:
        rc = syntax_check(args.out)
        if rc not in (0, None):
            raise SystemExit(f"clang syntax failed: {rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
