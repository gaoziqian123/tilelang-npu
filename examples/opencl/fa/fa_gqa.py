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
                   threads: int, bk: int, ablate: str = "none"):
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
            # accumulator.  Shaped as (block_i, block_j, 4, 4) / (block_i,
            # block_v, 2, 16) so every fragment access is unit-scale in the
            # loop variables: layout inference's InverseAffineIterMap rejects
            # scaled indices (acc[bi*4+ii] -> CanProveEqual(scale,1) fails),
            # so the block factor lives in the fragment shape instead.
            acc_s = T.alloc_fragment((tile_q // 4, tile_kv // 4, 4, 4), "float32")
            # NOTE: a 4D-blocked acc_o (e.g. (tile_q//2, D//16, 2, 16), the
            # fa.cl PV thread mapping) makes layout inference replicate 4x
            # work per thread (128 accs) -> certain spills, ~7.5s.  Instead
            # keep one fragment dim per index (scale-1 everywhere) and let
            # the last dim be the vector lane: 8-wide V loads via
            # T.vectorized(8).
            acc_o = T.alloc_fragment((tile_q, D // 8, 8), "float32")
            # Per-thread half4 staging for the QK splat operands: one deref
            # vector load per k, then free .sN component splats (the fa.cl
            # form) instead of a scalar load + convert + (float4)(x,x,x,x)
            # constructor per FMA row.
            q4 = T.alloc_local((4,), "float16")
            k4 = T.alloc_local((4,), "float16")
            qtmp = T.alloc_local((4,), "float16")
            ktmp = T.alloc_local((4,), "float16")

            for i in T.Parallel(tile_q):
                mrun[i] = -32768.0
                lrun[i] = 0.0
                alpha[i] = 1.0
            # All fragment accesses use the handwritten fa.cl block-threaded
            # form: the Parallel loop maps threads to 4x4 (QK) / 2x16 (PV)
            # blocks one-to-one, serial loops inside are per-thread.  Layout
            # inference then derives the fa.cl thread mapping naturally
            # (Qs/Ks vector loads + splat FMAs); mixing flat fragment loops
            # with blocked ones breaks InverseAffineIterMap, so EVERY
            # acc_s/acc_o access below is block-indexed.
            for i, dv8 in T.Parallel(tile_q, D // 8):
                for dv in T.serial(8):
                    acc_o[i, dv8, dv] = 0.0
            T.sync_threads()

            q0 = bx * tile_q
            # Causal: row q0+i sees keys [0, q0+i]; loop over tile_kv-wide
            # visible steps (over-reach is masked below).
            for ks in T.serial(T.ceildiv(q0 + tile_q, tile_kv)):
                kv0 = ks * tile_kv

                # ---- Phase 1: Sh = Q_tile @ K_tile^T, bk-dim chunks ----
                for bi, bj in T.Parallel(tile_q // 4, tile_kv // 4):
                    for ii in T.serial(4):
                        for jj in T.serial(4):
                            acc_s[bi, bj, ii, jj] = 0.0
                for kb in T.serial(0, D, bk):
                    # Stage transposed: half4 vector loads along D into a
                    # per-thread local, then scalar strided shared stores
                    # (the fa.cl staging form).
                    for m, c4 in T.Parallel(tile_q if ablate not in ("stage", "all") else 1, bk // 4) if ablate not in ("stage", "all") else T.Parallel(1, 1):
                        for cc in T.vectorized(4):
                            qtmp[cc] = Q[hq * S + q0 + m, kb + c4 * 4 + cc]
                        for cc in T.serial(4):
                            Qs[c4 * 4 + cc, m] = qtmp[cc]
                    for n, c4 in T.Parallel(tile_kv if ablate != "skel" else 1, bk // 4) if ablate != "skel" else T.Parallel(1, 1):
                        for cc in T.vectorized(4):
                            ktmp[cc] = K[g * S + kv0 + n, kb + c4 * 4 + cc]
                        for cc in T.serial(4):
                            Ks[c4 * 4 + cc, n] = ktmp[cc]
                    T.sync_threads()
                    if ablate not in ("qk", "all"):
                        for bi, bj in T.Parallel(tile_q // 4, tile_kv // 4):
                            for k in T.serial(bk):
                                for ii in T.vectorized(4):
                                    q4[ii] = Qs[k, bi * 4 + ii]
                                # Hoist the ii-invariant Ks vector load out
                                # of the ii loop: the vectorizer otherwise
                                # emits one vload4 per ii (4x redundant
                                # loads the device compiler does not CSE).
                                for jj in T.vectorized(4):
                                    k4[jj] = Ks[k, bj * 4 + jj]
                                for ii in T.serial(4):
                                    for jj in T.serial(4):
                                        acc_s[bi, bj, ii, jj] = (
                                            acc_s[bi, bj, ii, jj]
                                            + T.Cast("float32", q4[ii])
                                            * T.Cast("float32", k4[jj])
                                        )
                    T.sync_threads()
                # Causal mask + scale, store fp16 in place.
                if ablate not in ("mask", "all"):
                    for bi, bj in T.Parallel(tile_q // 4, tile_kv // 4):
                        for ii in T.serial(4):
                            for jj in T.serial(4):
                                Sh[bi * 4 + ii, bj * 4 + jj] = T.Cast(
                                    "float16",
                                    T.if_then_else(
                                        kv0 + bj * 4 + jj > q0 + bi * 4 + ii,
                                        -32768.0,
                                        acc_s[bi, bj, ii, jj] * 0.0625,  # 1/sqrt(256)
                                    ),
                                )
                elif ablate == "mask":
                    for bi, bj in T.Parallel(tile_q // 4, tile_kv // 4):
                        for ii in T.serial(4):
                            for jj in T.serial(4):
                                Sh[bi * 4 + ii, bj * 4 + jj] = T.Cast(
                                    "float16", acc_s[bi, bj, ii, jj] * 0.0625
                                )
                T.sync_threads()

                # ---- Phase 2: online softmax, one thread per row ----
                for r in T.Parallel(tile_q) if ablate not in ("softmax", "all") else T.Parallel(1):
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
                for i, dv8 in T.Parallel(tile_q if ablate != "skel" else 1, D // 8 if ablate != "skel" else 2) if ablate != "skel" else T.Parallel(1, 1):
                    for dv in T.serial(8):
                        acc_o[i, dv8, dv] = acc_o[i, dv8, dv] * alpha[i]
                if ablate not in ("pv", "all"):
                    for j in T.serial(tile_kv):
                        for i, dv8 in T.Parallel(tile_q, D // 8):
                            for dv in T.vectorized(8):
                                acc_o[i, dv8, dv] = (
                                    acc_o[i, dv8, dv]
                                    + T.Cast("float32", Sh[i, j])
                                    * T.Cast("float32", V[g * S + kv0 + j, dv8 * 8 + dv])
                                )
                T.sync_threads()

            # Epilogue: o = acc / l, fp16 store.
            for i, dv8 in T.Parallel(tile_q, D // 8):
                for dv in T.serial(8):
                    O[hq * S + q0 + i, dv8 * 8 + dv] = T.Cast(
                        "float16", acc_o[i, dv8, dv] / lrun[i]
                    )

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
    ap.add_argument("--ablate", choices=("none", "qk", "pv", "softmax", "stage", "mask", "all", "skel"), default="none")
    args = ap.parse_args()

    if args.hq % args.hkv:
        raise SystemExit("hq must be a multiple of hkv")
    if args.s % args.tile_q or args.d % args.bk:
        raise SystemExit("need s%tile_q==0 and d%bk==0")
    if args.tile_q % 4 or args.tile_kv % 4 or args.d % 16:
        raise SystemExit("need tile_q%4==0, tile_kv%4==0, d%16==0 (block-threaded loops)")
    if (args.tile_q // 4) * (args.tile_kv // 4) != args.threads:
        raise SystemExit("threads must equal (tile_q/4)*(tile_kv/4)")
    if (args.tile_q // 2) * (args.d // 16) != args.threads:
        raise SystemExit("threads must equal (tile_q/2)*(d/16)")

    with tvm.target.Target("opencl"), tvm.transform.PassContext(config=PASS_CFG):
        artifact = tilelang.lower(
            make_fa_kernel(args.s, args.hq, args.hkv, args.d, args.tile_q,
                           args.tile_kv, args.threads, args.bk, args.ablate),
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
