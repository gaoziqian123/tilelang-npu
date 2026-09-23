"""Split FFN-chain kernels for Adreno OpenCL via TileLang GR GEMM.

Shipping path: G = A @ Wg, U = A @ Wu, H = silu(G) * U, out = H @ Wd,
emitted as four serial OpenCL kernels. The old fused gate/up factory remains as
a reference only.

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

GEMM_DIR = REPO_ROOT / "examples" / "opencl" / "gemm"
if str(GEMM_DIR) not in sys.path:
    sys.path.insert(0, str(GEMM_DIR))
from gemm_nt import make_grgemm_kernel  # noqa: E402

FFN_DIR = REPO_ROOT / "examples" / "opencl" / "ffn"
if str(FFN_DIR) not in sys.path:
    sys.path.insert(0, str(FFN_DIR))
from silu_mul import make_silu_mul_kernel  # noqa: E402

PASS_CFG = {"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}


def make_ffn_gate_up_kernel(M: int, N: int, K: int, bm: int, bn: int, threads: int):
    @T.prim_func
    def ffn_gate_up_kernel(
        A: T.Tensor((M, K), "float16"),
        Wg: T.Tensor((K, N), "float16"),
        Wu: T.Tensor((K, N), "float16"),
        G: T.Tensor((M, N), "float16"),
        U: T.Tensor((M, N), "float16"),
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
            T.copy(acc_g, G[by * bm, bx * bn])
            T.copy(acc_u, U[by * bm, bx * bn])

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


def emit_kernel(func, out: Path) -> None:
    with tvm.target.Target("opencl"), tvm.transform.PassContext(config=PASS_CFG):
        artifact = tilelang.lower(func, target="opencl", enable_device_compile=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(artifact.kernel_source, encoding="utf-8")
    print(f"SOURCE_PATH {out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit the TileLang split FFN-chain OpenCL kernels.")
    ap.add_argument("--m", type=int, default=960)
    ap.add_argument("--n", type=int, default=9216, help="FFN hidden width")
    ap.add_argument("--k", type=int, default=2560)
    ap.add_argument("--n2", type=int, default=2560, help="down output width")
    # Tuned by examples/opencl/tuner/tune_ffn.py; see tuner/out/ffn_cd_mvp/final_config.json.
    ap.add_argument("--gate-bm", type=int, default=64)
    ap.add_argument("--gate-bn", type=int, default=256)
    ap.add_argument("--gate-bk", type=int, default=16)
    ap.add_argument("--gate-threads", type=int, default=256)
    ap.add_argument("--up-bm", type=int, default=64)
    ap.add_argument("--up-bn", type=int, default=256)
    ap.add_argument("--up-bk", type=int, default=64)
    ap.add_argument("--up-threads", type=int, default=256)
    ap.add_argument("--down-bm", type=int, default=32)
    ap.add_argument("--down-bn", type=int, default=128)
    ap.add_argument("--down-bk", type=int, default=64)
    ap.add_argument("--down-threads", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--fused-out", type=Path, default=None, help="optionally also emit fused gate_up reference")
    ap.add_argument("--fused-bm", type=int, default=None)
    ap.add_argument("--fused-bn", type=int, default=None)
    ap.add_argument("--fused-threads", type=int, default=None)
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    stages = [
        ("gate", args.gate_bm, args.gate_bn, args.gate_bk, args.gate_threads, args.k, args.n),
        ("up", args.up_bm, args.up_bn, args.up_bk, args.up_threads, args.k, args.n),
        ("down", args.down_bm, args.down_bn, args.down_bk, args.down_threads, args.n, args.n2),
    ]
    for name, bm, bn, bk, threads, kdim, ndim in stages:
        if bm % 8 or bn % 8 or bk % 4:
            raise SystemExit(f"{name}: grgemm requires bm%8==0, bn%8==0, bk%4==0")
        if threads != (bm // 8) * (bn // 8):
            raise SystemExit(f"{name}: threads == (bm/8)*(bn/8) = {(bm // 8) * (bn // 8)}")
        emit_kernel(
            make_grgemm_kernel(args.m, ndim, kdim, bm, bn, bk, threads, "kn", "float32", 256),
            args.out_dir / f"ffn_{name}.cl",
        )
    emit_kernel(make_silu_mul_kernel(args.m * args.n, 256, 8), args.out_dir / "silu_mul.cl")
    if args.fused_out is not None:
        fbm = args.fused_bm or args.gate_bm
        fbn = args.fused_bn or args.gate_bn
        fthreads = args.fused_threads or ((fbm // 8) * (fbn // 8))
        if fbm % 8 or fbn % 8 or fthreads != (fbm // 8) * (fbn // 8):
            raise SystemExit(f"fused: threads == (bm/8)*(bn/8) = {(fbm // 8) * (fbn // 8)}")
        emit_kernel(make_ffn_gate_up_kernel(args.m, args.n, args.k, fbm, fbn, fthreads), args.fused_out)
    if not args.skip_clang:
        for path in (args.out_dir / "ffn_gate.cl", args.out_dir / "ffn_up.cl", args.out_dir / "silu_mul.cl", args.out_dir / "ffn_down.cl"):
            rc = syntax_check(path)
            if rc not in (0, None):
                raise SystemExit(f"clang syntax failed for {path}: {rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
