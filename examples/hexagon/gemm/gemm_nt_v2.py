#!/usr/bin/env python3
"""Emit the Hexagon-v2 Phase-1 GEMM_NT kernel.

The ``get_tir`` function uses the same standard ``T.gemm`` structure as the
legacy example so the intended compiler input is visible.  Phase 1 lowers this
shape through the new v2 source-codegen bridge, which emits explicit
``hexagon.hmx_mma_deep`` deep-chain asm instead of the legacy whole-recipe
``hexagon.gemm_hmx`` call.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "gemm_nt_v2.c"
HEXAGON_CLANG = Path("/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang")
if not HEXAGON_CLANG.exists():
    HEXAGON_CLANG = Path("/root/hexagon-deps/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang")
HEXAGON_CLANG_FLAGS = [
    "-mv79",
    "-mhvx",
    "-mhvx-length=128B",
    "-mhmx",
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
import tilelang.hexagon.language as T  # noqa: E402
import tilelang.hexagon_v2  # noqa: F401,E402
from tilelang.engine import lower as engine_lower  # noqa: E402


@tilelang.jit(out_idx=[2], target="hexagon", execution_backend="aot")
def gemm_nt_v2_tir(M: int, N: int, K: int, block_M: int = 32, block_N: int = 1024, dtype=T.float16):
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


def get_tir(M: int, N: int, K: int, block_M: int = 32, block_N: int = 1024):
    return gemm_nt_v2_tir.get_tir(M=M, N=N, K=K, block_M=block_M, block_N=block_N)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit the Hexagon-v2 GEMM_NT kernel.")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=12288)
    parser.add_argument("--k", type=int, default=2560)
    parser.add_argument("--name", default="attnops_tl_gemm_nt_v2")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-clang", action="store_true", help="Skip hexagon-clang -fsyntax-only.")
    parser.add_argument("--dump-tir", action="store_true", help="Print the standard T.gemm TIR input.")
    return parser.parse_args()


def emit(args: argparse.Namespace) -> str:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tir = get_tir(args.m, args.n, args.k)
    if args.dump_tir:
        print(tir.script())
    old = {
        key: os.environ.get(key)
        for key in (
            "TILELANG_HEXAGON_V2_M",
            "TILELANG_HEXAGON_V2_N",
            "TILELANG_HEXAGON_V2_K",
            "TILELANG_HEXAGON_V2_SYMBOL",
        )
    }
    os.environ["TILELANG_HEXAGON_V2_M"] = str(args.m)
    os.environ["TILELANG_HEXAGON_V2_N"] = str(args.n)
    os.environ["TILELANG_HEXAGON_V2_K"] = str(args.k)
    os.environ["TILELANG_HEXAGON_V2_SYMBOL"] = args.name
    try:
        src = engine_lower(tir, target="hexagon_v2").kernel_source
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
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
