#!/usr/bin/env python3
"""Emit a Hexagon intrinsic C GEMM_NT kernel with TileLang.

The default shape is the Qwen prefill projection slice used on OnePlus 13:
M=960, N=8192, K=2560.  The generated C is written under
examples/hexagon/out/ by default and can be compiled into the FastRPC skel
project for device execution.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "gemm_nt.c"
HEXAGON_CLANG = Path("/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang")
HEXAGON_CLANG_FLAGS = [
    "-mv79",
    "-mhvx",
    "-mhvx-length=128B",
    "-fsyntax-only",
    "-I/root/hexagon-deps/HEXKL_DIR/hexkl_addon/include",
    "-I/root/hexagon-deps/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/incs",
    "-I/root/hexagon-deps/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/incs/stddef",
    "-I/root/project/backend/npu/attn/skel/hexagon_Release_toolv19_v79",
    "-I/root/project/backend/npu/attn/skel/src",
]

os.environ["PYTHONPATH"] = str(ROOT)
sys.path.insert(0, str(ROOT))

import tilelang  # noqa: E402
import tilelang.hexagon  # noqa: F401,E402 - registers backend/target
import tilelang.hexagon.language as T  # noqa: E402


@tilelang.jit(out_idx=[2], target="hexagon", execution_backend="aot")
def gemm_nt(M: int, N: int, K: int, block_M: int = 32, block_N: int = 1024, dtype=T.float16):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), threads=6) as bx:
            A_sh = T.alloc_shared((block_M, K), dtype, layout="ah")
            B_sh = T.alloc_shared((block_N, K), dtype, layout="wh")
            C_fr = T.alloc_fragment((block_M, block_N), T.float32)

            T.copy(B[bx * block_N, 0], B_sh)
            for m in T.serial(T.ceildiv(M, block_M)):
                T.copy(A[m * block_M, 0], A_sh, layout=("rm", "ah"))
                T.gemm(A_sh, B_sh, C_fr, transpose_B=True, clear_accum=True)
                T.copy(C_fr, C[m * block_M, bx * block_N], layout=("ah", "rm"))

    return main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit the Hexagon GEMM_NT example kernel.")
    parser.add_argument("--m", type=int, default=960)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=2560)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-clang", action="store_true", help="Skip hexagon-clang -fsyntax-only.")
    return parser.parse_args()


def emit(args: argparse.Namespace) -> str:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_HEXAGON_EMIT_C"] = str(args.out)
    artifact = tilelang.engine.lower(
        gemm_nt.get_tir(M=args.m, N=args.n, K=args.k, block_M=args.block_m, block_N=args.block_n)
        .with_attr("global_symbol", getattr(args, "name", None) or "attnops_tl_gemm_nt"),
        target="hexagon",
        enable_device_compile=False,
    )
    src = artifact.kernel_source
    args.out.write_text(src, encoding="utf-8")
    return src


def run_syntax_check(out: Path, skip: bool) -> int:
    if skip:
        print("HEXAGON_CLANG_SKIP requested")
        return 0
    if not HEXAGON_CLANG.exists():
        print(f"HEXAGON_CLANG_SKIP missing={HEXAGON_CLANG}")
        return 0
    cmd = [str(HEXAGON_CLANG), *HEXAGON_CLANG_FLAGS, str(out)]
    res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(res.stdout, end="")
    if res.returncode:
        print(f"HEXAGON_CLANG_FAIL rc={res.returncode}")
        return res.returncode
    print("HEXAGON_CLANG_PASS")
    return 0


def main() -> int:
    args = parse_args()
    src = emit(args)
    print(f"EMIT_OK path={args.out} lines={len(src.splitlines())}")
    return run_syntax_check(args.out, args.skip_clang)


if __name__ == "__main__":
    raise SystemExit(main())
