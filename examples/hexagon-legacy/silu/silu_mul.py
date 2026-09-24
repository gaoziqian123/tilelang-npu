#!/usr/bin/env python3
"""Emit Hexagon intrinsic C SwiGLU activation kernels with TileLang.

三个变体只改变数据搬运策略,kernel 算术写法保持一致:
- direct:直读 DDR + HVX 流式计算,真机 15.5 GB/s,当前冠军。
- vtcm:先 staging 到 VTCM 再算,真机 15.1 GB/s,无复用/无重叠时略亏。

实验结论:轻量流式算子直读最优,staging 只在有复用时赚。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIRECT_OUT = Path(__file__).resolve().parent / "out" / "silu_mul.c"
DEFAULT_VTCM_OUT = Path(__file__).resolve().parent / "out" / "silu_mul_vtcm.c"
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
def silu_mul(M: int, FF: int):
    # direct:DDR 直读 + HVX 流式计算。OnePlus 13 实测 15.5 GB/s
    # (15.53 GB/s),max_rel=0.0027,三变体中最快。
    @T.prim_func
    def main(g: T.Tensor((M * FF,), T.float16), u: T.Tensor((M * FF,), T.float16), o: T.Tensor((M * FF,), T.float16)):
        with T.Kernel(T.ceildiv(M * FF, 1024), threads=6) as bx:
            for p in T.Parallel(16):
                for v in T.vectorized(64):
                    i = bx * 1024 + p * 64 + v
                    # silu(g) * u,纯算术展开;emitter 逐节点 lowering
                    o[i] = g[i] * u[i] / (T.float16(1) + T.exp(-g[i]))

    return main


@tilelang.jit(out_idx=[2], target="hexagon", execution_backend="aot")
def silu_mul_vtcm(M: int, FF: int):
    # vtcm:把 1024 元素 tile staging 到 VTCM 后计算。OnePlus 13 实测
    # 15.1 GB/s(15.11 GB/s):轻量逐元素算子没有数据复用,copy 无法被
    # 有效重叠,所以比 direct 略亏。
    @T.prim_func
    def main(g: T.Tensor((M * FF,), T.float16), u: T.Tensor((M * FF,), T.float16), o: T.Tensor((M * FF,), T.float16)):
        with T.Kernel(T.ceildiv(M * FF, 1024), threads=6) as bx:
            G_sh = T.alloc_shared((1024,), T.float16)
            U_sh = T.alloc_shared((1024,), T.float16)
            O_sh = T.alloc_shared((1024,), T.float16)
            T.copy(g[bx * 1024:(bx + 1) * 1024], G_sh)
            T.copy(u[bx * 1024:(bx + 1) * 1024], U_sh)
            for p in T.Parallel(16):
                for v in T.vectorized(64):
                    O_sh[p * 64 + v] = G_sh[p * 64 + v] * U_sh[p * 64 + v] / (T.float16(1) + T.exp(-G_sh[p * 64 + v]))
            T.copy(O_sh, o[bx * 1024:(bx + 1) * 1024])

    return main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit the Hexagon silu(x) * up SwiGLU example kernel.")
    parser.add_argument("--impl", choices=("direct", "vtcm"), default="direct", help="Kernel variant to emit.")
    parser.add_argument("--m", type=int, default=960)
    parser.add_argument("--ff", type=int, default=9216)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-clang", action="store_true", help="Skip hexagon-clang -fsyntax-only.")
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_DIRECT_OUT if args.impl == "direct" else DEFAULT_VTCM_OUT
    return args


def emit(args: argparse.Namespace) -> str:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_HEXAGON_EMIT_C"] = str(args.out)
    if args.impl == "direct":
        tir = silu_mul.get_tir(M=args.m, FF=args.ff).with_attr("global_symbol", "attnops_tl_silu_generic")
    else:
        tir = silu_mul_vtcm.get_tir(M=args.m, FF=args.ff).with_attr("global_symbol", "attnops_tl_silu_vtcm")
    artifact = tilelang.engine.lower(
        tir,
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
