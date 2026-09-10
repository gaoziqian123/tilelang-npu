#!/usr/bin/env python3
"""Emit a Hexagon intrinsic C GDN prefill kernel with TileLang.

当前示例是显式 scratch buffer + 带操作数叶片的写法:所有中间 buffer
都在 TileLang 层声明,T.hexagon.* 只表示硬件动作叶片,不隐藏数据流。
少数仍保留的参数化复合叶片用于承载 128x128 state matvec、32 行
forward solve、32 行 state update 这类粒度合适的手写 HVX 配方。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "gdn_prefill.c"
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


@tilelang.jit(out_idx=[6, 7], target="hexagon", execution_backend="aot")
def gdn_prefill(TOK: int = 1024, Hk: int = 16, Hv: int = 32, D: int = 128, chunk: int = 32, dtype=T.float16):
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
        # Correct-granularity GDN: every scratch buffer is explicit TileLang
        # state. T.hexagon.* calls below are hardware-action leaves with
        # operands; no zero-arg primitive or hidden slot owns dataflow.
        with T.Kernel(Hv, threads=6) as hv:
            state = T.alloc_wscratch((128, 128), T.float32)
            kf = T.alloc_wscratch((chunk, 128), T.float32)
            qf = T.alloc_wscratch((chunk, 128), T.float32)
            vf = T.alloc_wscratch((chunk, 128), T.float32)
            w = T.alloc_wscratch((chunk, 128), T.float32)
            o_acc = T.alloc_wscratch((chunk, 128), T.float32)
            A = T.alloc_wscratch((chunk, chunk), T.float32)
            P = T.alloc_wscratch((chunk, chunk), T.float32)
            kkprod = T.alloc_wscratch((128,), T.float32)
            qkprod = T.alloc_wscratch((128,), T.float32)
            eG = T.alloc_wscratch((chunk,), T.float32)
            eGinv = T.alloc_wscratch((chunk,), T.float32)
            beta = T.alloc_wscratch((chunk,), T.float32)
            eGC = T.alloc_wscratch((1,), T.float32)

            T.hexagon.load_state128(S0, state, hv)
            for c in T.serial(TOK // chunk):
                hk = hv % Hk
                t0 = c * chunk

                # a) Q/K/V chunk fp16 -> fp32 scratch rows.
                T.hexagon.load_h2f_rows128(Q, K, V, qf, kf, vf, TOK, hk, hv, t0)

                # b) fp32 serial prefix gate scan and exp factors (R8 explicit).
                # Matches attnops_gdn.c:152-162: inclusive G cumsum,
                # clamp at -60, fp32 exp, beta copy, and chunk-final eGC.
                G_acc = T.alloc_var(T.float32, init=0.0)
                for i in T.serial(chunk):
                    G_acc = G_acc + G[hv, t0 + i]
                    Gc = T.alloc_var(T.float32, init=T.max(G_acc, -60.0))
                    eG[i] = T.hexagon.exp_fp32(Gc)
                    eGinv[i] = T.hexagon.exp_fp32(-Gc)
                    beta[i] = B[hv, t0 + i]
                eGC[0] = T.hexagon.exp_fp32(T.max(G_acc, -60.0))

                # c) A/P triangular matrices from user-layer dot128 reductions.
                # A is strictly lower triangular; P includes the diagonal.
                for i in T.serial(chunk):
                    for j in T.serial(i + 1):
                        # A[i,j] = beta[i] * eG[i] * eGinv[j] * dot(k_i,k_j)
                        # P[i,j] =           eG[i] * eGinv[j] * dot(q_i,k_j)
                        for dk in T.vectorized(128):
                            kkprod[dk] = kf[i, dk] * kf[j, dk]
                            qkprod[dk] = qf[i, dk] * kf[j, dk]
                        dkk = T.alloc_var(T.float32)
                        dqk = T.alloc_var(T.float32)
                        T.reduce_sum(kkprod, dkk)
                        T.reduce_sum(qkprod, dqk)
                        dec = T.alloc_var(T.float32, init=eG[i] * eGinv[j])
                        if j < i:
                            A[i, j] = beta[i] * dec * dkk
                        P[i, j] = dec * dqk

                # d/e) S0^T*k, S0^T*q and beta/eG affine w RHS.
                for i in T.serial(chunk):
                    T.hexagon.state_x2_matvec128(state, kf, qf, w, o_acc, i)
                    T.hexagon.affine_rows128(vf, w, beta, eG, i)

                # f) UT forward substitution over 32 rows of w.
                for i in T.serial(chunk):
                    T.hexagon.forward_solve32(A, w, i)

                # g) output row = eG*S0^T*q + tril(P)*w, stored fp16.
                for i in T.serial(chunk):
                    T.hexagon.output_rows128(o_acc, w, P, eG, O, TOK, hv, t0, i)

                # h) state update. Decay is only in e^{-gamma}/e^{gamma}/chunk end;
                # UT matrix above intentionally carries no decay.
                T.hexagon.state_decay_rows128(kf, eGC, eGinv)
                T.hexagon.state_update32(state, kf, w, eGC)
            T.hexagon.store_state128(state, S1, hv)

    return main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit the Hexagon GDN prefill example kernel.")
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
        gdn_prefill.get_tir(TOK=args.tokens, Hk=args.hk, Hv=args.hv, D=args.d, chunk=args.chunk)
        .with_attr("global_symbol", "attnops_tl_gdn_prefill"),
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
