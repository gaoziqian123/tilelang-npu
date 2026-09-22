"""Fused FFN gate/up kernel for Adreno OpenCL via TileLang GR GEMM.

Computes H[m, n] = silu(sum_k x[m, k] * Wg[k, n]) * (sum_k x[m, k] * Wu[k, n])
in one kernel: two GR fp16 GEMMs (direct-global vector loads, 8x8-per-thread
fp32 accumulators) with a fused silu-mul epilogue writing H as fp16.

The down projection out = H @ Wd is a plain GR GEMM and is emitted/tested via
examples/opencl/gemm/gemm_nt.py + gemm_nt_test.c.

Qwen3.5-4B shapes: M=960 (prompt960), K=2560 (EMBD), N=9216 (FF).
"""

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

PASS_CFG = {"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}


def make_ffn_gate_up_kernel(M: int, N: int, K: int, bm: int, bn: int, threads: int):
    @T.prim_func
    def ffn_gate_up_kernel(
        A: T.Tensor((M, K), "float16"),
        H: T.Tensor((M, N), "float16"),
        Wg: T.Tensor((K, N), "float16"),
        Wu: T.Tensor((K, N), "float16"),
    ):
        T.func_attr({"tl.opencl.assume_inbounds": 1})
        K  # keep K in the closure: the point-indexed gemm operands need it
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            # Measured on OnePlus 13 (M=960 N=9216 K=2560, fp32 accum):
            # fused gate+up 234ms (0.387T) == split gate/up 115+115ms.
            # The double accumulator (2x64 fp32/thread, GR 8x8 layout pins
            # 64 per accumulator) buys nothing over the split chain; the
            # split chain (gemm_nt + silu_mul + gemm_nt, see ffn_chain_test.c)
            # is the shipping path.  A two-half-N-pass variant of this fused
            # kernel produced scrambled layouts (fragment inside a serial
            # loop) - do not resurrect it without a layout check.
            acc_g = T.alloc_fragment((bm, bn), "float32")
            acc_u = T.alloc_fragment((bm, bn), "float32")
            T.gemm(A[by * bm, 0], Wg[0, bx * bn], acc_g, clear_accum=True)
            T.gemm(A[by * bm, 0], Wu[0, bx * bn], acc_u, clear_accum=True)
            for i, j in T.Parallel(bm, bn):
                g = acc_g[i, j]
                H[by * bm + i, bx * bn + j] = T.Cast(
                    "float16", (g / (1.0 + T.exp(-g))) * acc_u[i, j]
                )

    return ffn_gate_up_kernel


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
    ap = argparse.ArgumentParser(description="Emit the TileLang fused FFN gate/up OpenCL kernel.")
    ap.add_argument("--m", type=int, default=960)
    ap.add_argument("--n", type=int, default=9216)
    ap.add_argument("--k", type=int, default=2560)
    ap.add_argument("--bm", type=int, default=32)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "ffn_gate_up.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    if args.bm % 8 or args.bn % 8:
        raise SystemExit("grgemm requires bm%8==0, bn%8==0")
    if args.threads != (args.bm // 8) * (args.bn // 8):
        raise SystemExit(f"grgemm requires threads == (bm/8)*(bn/8) = {(args.bm // 8) * (args.bn // 8)}")

    with tvm.target.Target("opencl"), tvm.transform.PassContext(config=PASS_CFG):
        artifact = tilelang.lower(
            make_ffn_gate_up_kernel(args.m, args.n, args.k, args.bm, args.bn, args.threads),
            target="opencl",
            enable_device_compile=False,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(artifact.kernel_source, encoding="utf-8")
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")
    if not args.skip_clang:
        rc = syntax_check(args.out)
        if rc not in (0, None):
            raise SystemExit(f"clang syntax failed: {rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
