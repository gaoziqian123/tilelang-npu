#!/usr/bin/env python3
"""Emit a standard-construct TileLang Hexagon GDN prefill kernel.

This example is intentionally written only with TileLang language constructs
(T.copy/T.gemm/T.parallel/T.serial/T.vectorized/T.alloc_shared/
T.alloc_fragment/T.exp plus scalar arithmetic).  Unlike
``gdn_prefill.py``, it does not call any user-visible hand-written GDN C
leaf such as ``hrt_tlgdn_*``.

Shape is fixed to the Qwen GDN prefill anchor: T=1024, Hk=16, Hv=32,
D=128, chunk=32.  The generated kernel is a code-generation/structure
example; remaining expressiveness gaps are reported by this script's
``--self-check`` output rather than hidden behind handwritten C snippets.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "gdn_std.c"
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
from tilelang.transform import PassConfigKey, PassContext  # noqa: E402
import tilelang.hexagon  # noqa: F401,E402 - registers backend/target
import tilelang.hexagon.language as T  # noqa: E402


@tilelang.jit(out_idx=[6, 7], target="hexagon", execution_backend="aot")
def gdn_std(TOK: int = 1024, Hk: int = 16, Hv: int = 32, D: int = 128, chunk: int = 32):
    @T.prim_func
    def main(
        Q: T.Tensor((Hk, TOK, D), T.float16),
        K: T.Tensor((Hk, TOK, D), T.float16),
        V: T.Tensor((Hv, TOK, D), T.float16),
        G: T.Tensor((Hv, TOK), T.float32),
        B: T.Tensor((Hv, TOK), T.float32),
        S0: T.Tensor((Hv, D, D), T.float32),
        O: T.Tensor((Hv, TOK, D), T.float16),
        S1: T.Tensor((Hv, D, D), T.float32),
    ):
        with T.Kernel(Hv, threads=6) as hv:
            # Zero-short-path standard GDN.  This is the same chunk algebra as
            # attnops_gdn.c, but all dataflow is expressed in TileLang loops.
            # No hrt_tlgdn_* chunk helper owns a matvec/solve/update step.
            state = T.alloc_wscratch((D, D), T.float32)
            kf = T.alloc_wscratch((chunk, D), T.float32)
            qf = T.alloc_wscratch((chunk, D), T.float32)
            vf = T.alloc_wscratch((chunk, D), T.float32)
            w = T.alloc_wscratch((chunk, D), T.float32)
            o_acc = T.alloc_wscratch((chunk, D), T.float32)
            A = T.alloc_wscratch((chunk, chunk), T.float32)
            P = T.alloc_wscratch((chunk, chunk), T.float32)
            kkprod = T.alloc_wscratch((D,), T.float32)
            qkprod = T.alloc_wscratch((D,), T.float32)
            eG = T.alloc_wscratch((chunk,), T.float32)
            eGinv = T.alloc_wscratch((chunk,), T.float32)
            beta = T.alloc_wscratch((chunk,), T.float32)
            eGC = T.alloc_wscratch((1,), T.float32)

            hk = hv % Hk
            for r0 in T.serial(D):
                for d0 in T.vectorized(D):
                    state[r0, d0] = S0[hv, r0, d0]
            for c in T.serial(TOK // chunk):
                t0 = c * chunk
                for i_load in T.serial(chunk):
                    for d_load in T.vectorized(D):
                        qf[i_load, d_load] = T.Cast("float32", Q[hk, t0 + i_load, d_load])
                        kf[i_load, d_load] = T.Cast("float32", K[hk, t0 + i_load, d_load])
                        vf[i_load, d_load] = T.Cast("float32", V[hv, t0 + i_load, d_load])

                G_acc = T.alloc_var(T.float32, init=0.0)
                for i_gate in T.serial(chunk):
                    G_acc = G_acc + G[hv, t0 + i_gate]
                    Gc = T.alloc_var(T.float32, init=T.max(G_acc, -60.0))
                    eG[i_gate] = T.exp(Gc)
                    eGinv[i_gate] = T.exp(-Gc)
                    beta[i_gate] = B[hv, t0 + i_gate]
                eGC[0] = T.exp(T.max(G_acc, -60.0))

                for i_ap in T.serial(chunk):
                    for j_ap in T.serial(i_ap + 1):
                        for dk0 in T.vectorized(D):
                            kkprod[dk0] = kf[i_ap, dk0] * kf[j_ap, dk0]
                            qkprod[dk0] = qf[i_ap, dk0] * kf[j_ap, dk0]
                        dkk = T.alloc_var(T.float32)
                        dqk = T.alloc_var(T.float32)
                        T.reduce_sum(kkprod, dkk)
                        T.reduce_sum(qkprod, dqk)
                        dec = T.alloc_var(T.float32, init=eG[i_ap] * eGinv[j_ap])
                        if j_ap < i_ap:
                            A[i_ap, j_ap] = beta[i_ap] * dec * dkk
                        P[i_ap, j_ap] = dec * dqk

                for i_mv in T.serial(chunk):
                    for d_init in T.vectorized(D):
                        w[i_mv, d_init] = 0.0
                        o_acc[i_mv, d_init] = 0.0
                    for dk1 in T.serial(D):
                        for d_mv in T.vectorized(D):
                            w[i_mv, d_mv] = w[i_mv, d_mv] + kf[i_mv, dk1] * state[dk1, d_mv]
                            o_acc[i_mv, d_mv] = o_acc[i_mv, d_mv] + qf[i_mv, dk1] * state[dk1, d_mv]
                    for d_aff in T.vectorized(D):
                        w[i_mv, d_aff] = beta[i_mv] * vf[i_mv, d_aff] - beta[i_mv] * eG[i_mv] * w[i_mv, d_aff]

                for i_sol in T.serial(chunk):
                    for j_sol in T.serial(i_sol):
                        for d_sol in T.vectorized(D):
                            w[i_sol, d_sol] = w[i_sol, d_sol] - A[i_sol, j_sol] * w[j_sol, d_sol]

                for i_out in T.serial(chunk):
                    for d_scale in T.vectorized(D):
                        o_acc[i_out, d_scale] = eG[i_out] * o_acc[i_out, d_scale]
                    for j_out in T.serial(i_out + 1):
                        for d_out in T.vectorized(D):
                            o_acc[i_out, d_out] = o_acc[i_out, d_out] + P[i_out, j_out] * w[j_out, d_out]
                    for d_store in T.vectorized(D):
                        O[hv, t0 + i_out, d_store] = T.Cast("float16", o_acc[i_out, d_store])

                for i_dec in T.serial(chunk):
                    for d_dec in T.vectorized(D):
                        kf[i_dec, d_dec] = eGC[0] * eGinv[i_dec] * kf[i_dec, d_dec]
                for dk2 in T.serial(D):
                    for d_state0 in T.vectorized(D):
                        state[dk2, d_state0] = eGC[0] * state[dk2, d_state0]
                    for i_up in T.serial(chunk):
                        for d_state in T.vectorized(D):
                            state[dk2, d_state] = state[dk2, d_state] + kf[i_up, dk2] * w[i_up, d_state]

            for r1 in T.serial(D):
                for d1 in T.vectorized(D):
                    S1[hv, r1, d1] = state[r1, d1]

    return main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit standard-construct Hexagon GDN prefill example.")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hk", type=int, default=16)
    parser.add_argument("--hv", type=int, default=32)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-clang", action="store_true", help="Skip hexagon-clang -fsyntax-only.")
    parser.add_argument("--self-check", action="store_true", help="Run structural checks on generated C.")
    return parser.parse_args()


def emit(args: argparse.Namespace) -> str:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_HEXAGON_EMIT_C"] = str(args.out)
    with PassContext(config={PassConfigKey.TL_HEXAGON_PROF.value: True}):
        artifact = tilelang.engine.lower(
            gdn_std.get_tir(TOK=args.tokens, Hk=args.hk, Hv=args.hv, D=args.d, chunk=args.chunk)
            .with_attr("global_symbol", "attnops_tl_gdn_std"),
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


def structural_check(src: str) -> int:
    checks: list[tuple[str, bool]] = []
    checks.append(("pool workers emitted", "attnops_pool_run_ctx(" in src and "_pool" in src))
    checks.append(("hmx present", "hrt_hmx_mm_f16" in src and "hrt_acc_read_f16" in src))
    worker_blob = src.split("int attnops_tl_gdn_std", 1)[0]
    checks.append(("hmx outside worker functions", "hrt_hmx_mm_f16" not in worker_blob))
    checks.append(("copy recipe staging", "hrt_tlgdn_stage_f32_to_ah" in src or "hrt_tlgdn_stage_f32_to_wh" in src))
    checks.append(("prof counters", "tl.hexagon_prof profile slots" in src and "tl_prof_phase_t0" in src and "prof[5.." in src))
    m = re.search(r"#define TL_GENERIC_VTCM_BYTES \(\(size_t\)(\d+)\)", src)
    vtcm = int(m.group(1)) if m else -1
    checks.append((f"vtcm budget {vtcm} < 8MB", 0 <= vtcm < 8 * 1024 * 1024))
    rc = 0
    for name, ok in checks:
        print(f"CHECK_{'PASS' if ok else 'FAIL'} {name}")
        rc = rc or (0 if ok else 1)
    return rc


def main() -> int:
    args = parse_args()
    src = emit(args)
    print(f"EMIT_OK path={args.out} lines={len(src.splitlines())}")
    rc = structural_check(src) if args.self_check else 0
    rc = run_syntax_check(args.out, args.skip_clang) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
