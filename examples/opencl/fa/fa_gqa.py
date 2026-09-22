#!/usr/bin/env python3
"""Emit a TileLang OpenCL GQA flash-attention prefill kernel.

Structure mirrors the handwritten production kernel backend/gpu/fa/fa.cl
(~25.5ms on this shape), adapted to TileLang while respecting the measured
Adreno compiler facts (DESIGN_rr_gemm section 10):

- One workgroup of 256 threads per (q head, 32-row q tile), kv tiles of 128.
- QK^T: Qs/Ks are staged TRANSPOSED ([k][m] / [k][n]) into shared in BK=16
  dim chunks (scalar transposed stores, vector global loads), then an
  elementwise FMA loop accumulates the (32,128) score fragment in fp32.
  No T.gemm: the GR vector lower requires threads == (M/8)*(N/8), which
  conflicts with the 256-thread workgroup the rest of the pipeline needs.
- Online softmax one thread per row, running m/l/alpha in small shared
  vectors, P written back fp16 in place.
- P@V: persistent fp32 output accumulators in registers (32 per thread,
  (32,256)/256), rescaled by alpha per kv step, acc += p_j * V[j][dv] with V
  read straight from global.  No O tile in shared and no global
  read-modify-write: the Adreno OpenCL compiler fails to build global RMW
  loops nested inside the ks/dc loops (clBuildProgram error with an empty
  "Pass" log), and a 16KB shared O tile caps residency at 1 workgroup/CU.
- Local memory ~14KB, two workgroups stay resident per 32KB Adreno CU.

Fixed anchor shape: S=1024, HQ=16, HKV=4, D=256, causal, fp16 in/out.
Q is [HQ*S, D] and K/V are [HKV*S, D] row-major head-major, matching the
hexagon fa_std ABI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tilelang  # noqa: E402
import tilelang.opencl  # noqa: F401,E402 - registers OpenCL backend
from tilelang import tvm  # noqa: E402
import tilelang.language as T  # noqa: E402

PASS_CFG = {"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}


def make_fa_kernel(S: int, HQ: int, HKV: int, D: int, tile_q: int, tile_kv: int,
                   threads: int, bk: int):
    group = HQ // HKV

    @T.prim_func
    def fa_gqa_kernel(
        Q: T.Tensor((HQ * S, D), "float16"),
        K: T.Tensor((HKV * S, D), "float16"),
        V: T.Tensor((HKV * S, D), "float16"),
        O: T.Tensor((HQ * S, D), "float16"),
    ):
        T.func_attr({"tl.opencl.assume_inbounds": 1})
        with T.Kernel(T.ceildiv(S, tile_q), HQ, threads=threads) as (bx, by):
            HKV  # keep in the closure: the K/V tensor annotations need it
            hq = by
            g = hq // group
            # Transposed staging for the QK^T FMA loop, re-staged every bk
            # dims (the fa.cl pattern: tiny local buffers, many barriers).
            Qs = T.alloc_shared((bk, tile_q), "float16")   # [k][m]
            Ks = T.alloc_shared((bk, tile_kv), "float16")  # [k][n]
            # Scores then probabilities, fp16 (magnitudes small, rounding at
            # input level - same lesson as fa.cl).
            Sh = T.alloc_shared((tile_q, tile_kv), "float16")
            mrun = T.alloc_shared((tile_q,), "float32")
            lrun = T.alloc_shared((tile_q,), "float32")
            alpha = T.alloc_shared((tile_q,), "float32")
            # Score fragment (per kv tile) and the persistent output
            # accumulator: (tile_q, D) fp32 spread over all threads.
            acc_s = T.alloc_fragment((tile_q, tile_kv), "float32")
            acc_o = T.alloc_fragment((tile_q, D), "float32")

            for i in T.Parallel(tile_q):
                mrun[i] = -32768.0
                lrun[i] = 0.0
                alpha[i] = 1.0
            for i, dv in T.Parallel(tile_q, D):
                acc_o[i, dv] = 0.0
            T.sync_threads()

            q0 = bx * tile_q
            # Causal: row q0+i sees keys [0, q0+i]; loop over tile_kv-wide
            # visible steps (over-reach is masked below).
            for ks in T.serial(T.ceildiv(q0 + tile_q, tile_kv)):
                kv0 = ks * tile_kv

                # ---- Phase 1: Sh = Q_tile @ K_tile^T, bk-dim chunks ----
                for i, j in T.Parallel(tile_q, tile_kv):
                    acc_s[i, j] = 0.0
                for kb in T.serial(0, D, bk):
                    # Stage transposed: consecutive c is stride-tile_q /
                    # stride-tile_kv in the store (scalar shared stores,
                    # contiguous vector global loads along D).
                    for m, c in T.Parallel(tile_q, bk):
                        Qs[c, m] = Q[hq * S + q0 + m, kb + c]
                    for n, c in T.Parallel(tile_kv, bk):
                        Ks[c, n] = K[g * S + kv0 + n, kb + c]
                    T.sync_threads()
                    for k in T.serial(bk):
                        for i, j in T.Parallel(tile_q, tile_kv):
                            acc_s[i, j] = (
                                acc_s[i, j]
                                + T.Cast("float32", Qs[k, i])
                                * T.Cast("float32", Ks[k, j])
                            )
                    T.sync_threads()
                # Causal mask + scale, store fp16 in place.
                for i, j in T.Parallel(tile_q, tile_kv):
                    Sh[i, j] = T.Cast(
                        "float16",
                        T.if_then_else(
                            kv0 + j > q0 + i,
                            -32768.0,
                            acc_s[i, j] * 0.0625,  # 1/sqrt(256)
                        ),
                    )
                T.sync_threads()

                # ---- Phase 2: online softmax, one thread per row ----
                for r in T.Parallel(tile_q):
                    m_old = T.alloc_var("float32")
                    m_old = mrun[r]
                    m_new = T.alloc_var("float32")
                    m_new = m_old
                    for c in T.serial(tile_kv):
                        m_new = T.max(m_new, T.Cast("float32", Sh[r, c]))
                    alpha[r] = T.exp(m_old - m_new)  # m_old=-inf -> 0
                    l_t = T.alloc_var("float32")
                    l_t = 0.0
                    for c in T.serial(tile_kv):
                        p = T.exp(T.Cast("float32", Sh[r, c]) - m_new)
                        Sh[r, c] = T.Cast("float16", p)
                        l_t = l_t + p
                    lrun[r] = lrun[r] * alpha[r] + l_t
                    mrun[r] = m_new
                T.sync_threads()

                # ---- Phase 3: acc_o = acc_o*alpha + P @ V tile ----
                # V streams straight from global (hot in L2 across q tiles);
                # P comes from shared as a per-row scalar broadcast.
                for i, dv in T.Parallel(tile_q, D):
                    acc_o[i, dv] = acc_o[i, dv] * alpha[i]
                for j in T.serial(tile_kv):
                    for i, dv in T.Parallel(tile_q, D):
                        acc_o[i, dv] = (
                            acc_o[i, dv]
                            + T.Cast("float32", Sh[i, j])
                            * T.Cast("float32", V[g * S + kv0 + j, dv])
                        )
                T.sync_threads()

            # Epilogue: o = acc / l, fp16 store.
            for i, dv in T.Parallel(tile_q, D):
                O[hq * S + q0 + i, dv] = T.Cast("float16", acc_o[i, dv] / lrun[i])

    return fa_gqa_kernel


def syntax_check(path: Path) -> int | None:
    clang = subprocess.run(
        ["bash", "-lc", "command -v clang"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if clang.returncode != 0:
        print("CLANG_SYNTAX_SKIP no clang in PATH (set CLANG_BIN)")
        return None
    res = subprocess.run(
        ["bash", "-lc", f"clang -fsyntax-only -x cl -cl-std=CL3.0 {path}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(res.stdout, end="")
    if res.returncode:
        print(f"CLANG_SYNTAX_FAIL rc={res.returncode}")
    return res.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit the TileLang OpenCL GQA flash-attention prefill kernel.")
    ap.add_argument("--s", type=int, default=1024)
    ap.add_argument("--hq", type=int, default=16)
    ap.add_argument("--hkv", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--tile-q", type=int, default=32)
    ap.add_argument("--tile-kv", type=int, default=128)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--bk", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "fa_gqa.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    if args.hq % args.hkv:
        raise SystemExit("hq must be a multiple of hkv")
    if args.s % args.tile_q or args.d % args.bk:
        raise SystemExit("need s%tile_q==0 and d%bk==0")

    with tvm.target.Target("opencl"), tvm.transform.PassContext(config=PASS_CFG):
        artifact = tilelang.lower(
            make_fa_kernel(args.s, args.hq, args.hkv, args.d, args.tile_q,
                           args.tile_kv, args.threads, args.bk),
            target="opencl",
            enable_device_compile=False,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(artifact.kernel_source, encoding="utf-8")
    print("SELECTED_PATH fa_gqa")
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")
    if not args.skip_clang:
        rc = syntax_check(args.out)
        if rc not in (0, None):
            raise SystemExit(f"clang syntax failed: {rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
