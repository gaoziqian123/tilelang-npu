"""Elementwise silu(G) * U OpenCL kernel for the split-FFN plan B chain.

H[m, n] = silu(G[m, n]) * U[m, n], fp16 in/out. Flat 1D, 8 contiguous
elements per thread (vectorizes to half8 on Adreno).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilelang
import tilelang.opencl  # noqa: F401
from tilelang import tvm
import tilelang.language as T

PASS_CFG = {"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}


def make_silu_mul_kernel(total: int, threads: int, elems_per_thread: int):
    per_wg = threads * elems_per_thread

    @T.prim_func
    def silu_mul_kernel(
        G: T.Tensor((total,), "float16"),
        H: T.Tensor((total,), "float16"),
        U: T.Tensor((total,), "float16"),
    ):
        T.func_attr({"tl.opencl.assume_inbounds": 1})
        with T.Kernel(total // per_wg, threads=threads) as bx:
            base = bx * per_wg + T.get_thread_binding() * elems_per_thread
            for i in T.serial(elems_per_thread):
                g = T.Cast("float32", G[base + i])
                H[base + i] = T.Cast(
                    "float16", (g / (1.0 + T.exp(-g))) * T.Cast("float32", U[base + i])
                )

    return silu_mul_kernel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=960)
    ap.add_argument("--n", type=int, default=9216)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--ept", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "silu_mul.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    total = args.m * args.n
    per_wg = args.threads * args.ept
    if total % per_wg != 0:
        raise SystemExit(f"total {total} not divisible by wg size {per_wg}")

    with tvm.target.Target("opencl"), tvm.transform.PassContext(config=PASS_CFG):
        artifact = tilelang.lower(
            make_silu_mul_kernel(total, args.threads, args.ept),
            target="opencl",
            enable_device_compile=False,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(artifact.kernel_source, encoding="utf-8")
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")
    if not args.skip_clang:
        import os, subprocess
        clang = os.environ.get("CLANG_BIN", "clang")
        headers = Path("/root/project/backend/gpu/OpenCL-Headers")
        cmd = [clang, "-x", "cl", "-cl-std=CL3.0", "-fsyntax-only"]
        if headers.exists():
            cmd += ["-I", str(headers)]
        cmd.append(str(args.out))
        res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(res.stdout, end="")
        print(f"CLANG_SYNTAX_RC {res.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
