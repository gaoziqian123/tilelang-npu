#!/usr/bin/env python3
"""Compile-only GDN prefill variant with arbitrary wscratch names.

This regression intentionally renames the per-worker scratch declarations while
keeping the generated ABI/function shape identical to ``gdn_prefill.py``.  It
verifies that Hexagon wscratch lowering maps buffers by declaration order plus
shape/dtype instead of hard-coded user variable names.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "gdn_prefill_renamed.c"

os.environ["PYTHONPATH"] = str(ROOT)
sys.path.insert(0, str(ROOT))

import tilelang  # noqa: E402
import tilelang.hexagon  # noqa: F401,E402 - registers backend/target
import tilelang.hexagon.language as T  # noqa: E402
from examples.hexagon.gdn_prefill import run_syntax_check  # noqa: E402


@tilelang.jit(out_idx=[6, 7], target="hexagon", execution_backend="aot")
def gdn_prefill_renamed(TOK: int = 1024, Hk: int = 16, Hv: int = 32, D: int = 128, chunk: int = 32, dtype=T.float16):
    @T.prim_func
    def main(
        Q: T.Tensor((Hk, TOK, 128), T.float16),
        K: T.Tensor((Hk, TOK, 128), T.float16),
        V: T.Tensor((Hv, TOK, 128), T.float16),
        G: T.Tensor((Hv, TOK), T.float32),
        B: T.Tensor((Hv, TOK), T.float32),
        S0: T.Tensor((Hv, 128, 128), T.float32),
        O: T.Tensor((Hv, TOK, 128), T.float16),
        S1: T.Tensor((Hv, 128, 128), T.float32),
    ):
        with T.Kernel(Hv, threads=6) as hv:
            mys = T.alloc_wscratch((128, 128), T.float32)
            kk = T.alloc_wscratch((chunk, 128), T.float32)
            qq = T.alloc_wscratch((chunk, 128), T.float32)
            vv = T.alloc_wscratch((chunk, 128), T.float32)
            ww = T.alloc_wscratch((chunk, 128), T.float32)
            oo = T.alloc_wscratch((chunk, 128), T.float32)
            tri_a = T.alloc_wscratch((chunk, chunk), T.float32)
            tri_p = T.alloc_wscratch((chunk, chunk), T.float32)
            kprod = T.alloc_wscratch((128,), T.float32)
            qprod = T.alloc_wscratch((128,), T.float32)
            eg = T.alloc_wscratch((chunk,), T.float32)
            eg_inv = T.alloc_wscratch((chunk,), T.float32)
            bet = T.alloc_wscratch((chunk,), T.float32)
            egc = T.alloc_wscratch((1,), T.float32)

            T.hexagon.load_state128(S0, mys, hv)
            for c in T.serial(TOK // chunk):
                hk = hv % Hk
                t0 = c * chunk

                T.hexagon.load_h2f_rows128(Q, K, V, qq, kk, vv, TOK, hk, hv, t0)

                G_acc = T.alloc_var(T.float32, init=0.0)
                for i in T.serial(chunk):
                    G_acc = G_acc + G[hv, t0 + i]
                    Gc = T.alloc_var(T.float32, init=T.max(G_acc, -60.0))
                    eg[i] = T.hexagon.exp_fp32(Gc)
                    eg_inv[i] = T.hexagon.exp_fp32(-Gc)
                    bet[i] = B[hv, t0 + i]
                egc[0] = T.hexagon.exp_fp32(T.max(G_acc, -60.0))

                for i in T.serial(chunk):
                    for j in T.serial(i + 1):
                        for dk in T.vectorized(128):
                            kprod[dk] = kk[i, dk] * kk[j, dk]
                            qprod[dk] = qq[i, dk] * kk[j, dk]
                        dkk = T.alloc_var(T.float32)
                        dqk = T.alloc_var(T.float32)
                        T.reduce_sum(kprod, dkk)
                        T.reduce_sum(qprod, dqk)
                        dec = T.alloc_var(T.float32, init=eg[i] * eg_inv[j])
                        if j < i:
                            tri_a[i, j] = bet[i] * dec * dkk
                        tri_p[i, j] = dec * dqk

                for i in T.serial(chunk):
                    T.hexagon.state_x2_matvec128(mys, kk, qq, ww, oo, i)
                    T.hexagon.affine_rows128(vv, ww, bet, eg, i)

                for i in T.serial(chunk):
                    T.hexagon.forward_solve32(tri_a, ww, i)

                for i in T.serial(chunk):
                    T.hexagon.output_rows128(oo, ww, tri_p, eg, O, TOK, hv, t0, i)

                T.hexagon.state_decay_rows128(kk, egc, eg_inv)
                T.hexagon.state_update32(mys, kk, ww, egc)
            T.hexagon.store_state128(mys, S1, hv)

    return main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit renamed-wscratch Hexagon GDN prefill regression kernel.")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hk", type=int, default=16)
    parser.add_argument("--hv", type=int, default=32)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-clang", action="store_true", help="Skip hexagon-clang -fsyntax-only.")
    return parser.parse_args()


def emit(args: argparse.Namespace) -> str:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_HEXAGON_EMIT_C"] = str(args.out)
    artifact = tilelang.engine.lower(
        gdn_prefill_renamed.get_tir(TOK=args.tokens, Hk=args.hk, Hv=args.hv, D=args.d, chunk=args.chunk)
        .with_attr("global_symbol", "attnops_tl_gdn_prefill"),
        target="hexagon",
        enable_device_compile=False,
    )
    src = artifact.kernel_source
    args.out.write_text(src, encoding="utf-8")
    return src


def main() -> int:
    args = parse_args()
    src = emit(args)
    print(f"EMIT_OK path={args.out} lines={len(src.splitlines())}")
    return run_syntax_check(args.out, args.skip_clang)


if __name__ == "__main__":
    raise SystemExit(main())
