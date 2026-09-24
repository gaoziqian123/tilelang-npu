#!/usr/bin/env python3
"""Emit a Hexagon kernel that relies on automatic T.copy partitioning.

The kernel has a single large WH->WH staging copy and intentionally contains no
manual ``T.parallel`` slicing around it.  HexagonCopyPartition should rewrite it
into a worker-pool phase before emission.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "copy_auto.c"
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

os.environ.setdefault("PYTHONPATH", f"{ROOT / 'build/lib'}:{ROOT / '3rdparty/tvm/python'}:{ROOT}")
sys.path[:0] = [str(ROOT / "build/lib"), str(ROOT / "3rdparty/tvm/python"), str(ROOT)]

import tilelang  # noqa: E402
import tilelang.hexagon  # noqa: F401,E402 - registers backend/target
import tilelang.hexagon.language as T  # noqa: E402


@tilelang.jit(out_idx=[1], target="hexagon", execution_backend="aot")
def copy_auto(rows: int = 192, cols: int = 2560):
    @T.prim_func
    def main(W: T.Tensor((rows, cols), T.float16, layout="wh"), C: T.Tensor((32, rows), T.float16)):
        with T.Kernel(1, threads=1):
            A_a = T.alloc_shared((32, cols), T.float16, layout="ah")
            W_b = T.alloc_shared((rows, cols), T.float16, layout="wh")
            C_acc = T.alloc_fragment((32, rows), T.float32)
            T.copy(W[0:rows, 0:cols], W_b, layout=("wh", "wh"))
            T.gemm(A_a, W_b, C_acc, transpose_B=True, clear_accum=True)
            T.copy(C_acc, C, layout=("ah", "rm"))

    return main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit Hexagon automatic copy partition example.")
    parser.add_argument("--rows", type=int, default=192)
    parser.add_argument("--cols", type=int, default=2560)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-clang", action="store_true", help="Skip hexagon-clang -fsyntax-only.")
    return parser.parse_args()


def emit(args: argparse.Namespace) -> str:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_HEXAGON_EMIT_C"] = str(args.out)
    artifact = tilelang.engine.lower(
        copy_auto.get_tir(rows=args.rows, cols=args.cols).with_attr("global_symbol", "attnops_tl_copy_auto"),
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
