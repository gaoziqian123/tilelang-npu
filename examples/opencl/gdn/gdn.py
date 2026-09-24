#!/usr/bin/env python3
"""Emit a TileLang OpenCL GDN prefill kernel pair.

This is the GPU-side TileLang version of ``backend/gpu/gdn/gdn.cl`` for the
Qwen3.5 Gated DeltaNet anchor shape (T=1024/960, Hk=16, Hv=32, D=128,
chunk=32).  The algorithm and workgroup mapping intentionally mirror the
handwritten kernel:

* ``gdn_prep_kernel``: one 128-thread workgroup per (chunk, value-head).  It
  builds chunk-local gate scans, the strict-lower UT matrix (without decay
  folded into K^TQ), forward-solves U/W, stores kd and A2 scratch.
* ``gdn_seq_kernel``: one workgroup per (32-column dv block, value-head).  It
  scans chunks serially, computes inter + intra output, and updates fp32 state.

Only standard TileLang OpenCL constructs are used; no backend peepholes are
required beyond the existing FA/FFN/GEMM infrastructure.
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

import tilelang  # noqa: E402
import tilelang.opencl  # noqa: F401,E402
from tilelang import tvm  # noqa: E402
import tilelang.language as T  # noqa: E402

PASS_CFG = {"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}


def make_gdn_prep_kernel(TOK: int, Hk: int, Hv: int, D: int, chunk: int, threads: int):
    @T.prim_func
    def gdn_prep_kernel(
        Q: T.Tensor((Hk, TOK, D), "float16"),
        K: T.Tensor((Hk, TOK, D), "float16"),
        V: T.Tensor((Hv, TOK, D), "float16"),
        G: T.Tensor((Hv, TOK), "float32"),
        B: T.Tensor((Hv, TOK), "float32"),
        Ubuf: T.Tensor((Hv, TOK // chunk, chunk, D), "float16"),
        Wbuf: T.Tensor((Hv, TOK // chunk, chunk, D), "float16"),
        KDbuf: T.Tensor((Hv, TOK // chunk, chunk, D), "float16"),
        A2buf: T.Tensor((Hv, TOK // chunk, chunk, chunk), "float16"),
        EgcBuf: T.Tensor((Hv, TOK // chunk, chunk), "float32"),
        EglBuf: T.Tensor((Hv, TOK // chunk), "float32"),
    ):
        T.func_attr({"tl.opencl.assume_inbounds": 1})
        with T.Kernel(TOK // chunk, Hv, threads=threads) as (c, h):
            hk = h % Hk
            gf = T.alloc_shared((chunk,), "float32")
            bf = T.alloc_shared((chunk,), "float32")
            gc = T.alloc_shared((chunk,), "float32")
            egc = T.alloc_shared((chunk,), "float32")
            L = T.alloc_shared((chunk, chunk), "float32")
            UW = T.alloc_shared((chunk, D), "float16")

            for i in T.Parallel(chunk):
                gf[i] = G[h, c * chunk + i]
                bf[i] = B[h, c * chunk + i]
            T.sync_threads()

            for i in T.Parallel(chunk):
                s = T.alloc_var("float32")
                s = 0.0
                for r in T.serial(chunk):
                    if r <= i:
                        s = s + gf[r]
                gc[i] = s
                egc[i] = T.exp(s)
                EgcBuf[h, c, i] = T.exp(s)
            T.sync_threads()
            EglBuf[h, c] = T.exp(gc[chunk - 1])

            # L = beta_i * (k_i dot k_j) * exp(gc_i-gc_j), strict lower.
            # A2 = (q_i dot k_j) * exp(gc_i-gc_j), lower including diag.
            for p in T.Parallel(chunk * chunk):
                i = p // chunk
                j = p % chunk
                kk = T.alloc_var("float32")
                qk = T.alloc_var("float32")
                kk = 0.0
                qk = 0.0
                for kd in T.serial(D):
                    kj = T.Cast("float32", K[hk, c * chunk + j, kd])
                    kk = kk + T.Cast("float32", K[hk, c * chunk + i, kd]) * kj
                    qk = qk + T.Cast("float32", Q[hk, c * chunk + i, kd]) * kj
                decay = T.exp(gc[i] - gc[j])
                L[i, j] = T.if_then_else(j < i, bf[i] * kk * decay, 0.0)
                A2buf[h, c, i, j] = T.Cast("float16", T.if_then_else(j <= i, qk * decay, 0.0))
            T.sync_threads()

            # Forward solve U, then W.  Thread d owns one vector column.
            for d in T.Parallel(D):
                for i in T.serial(chunk):
                    au = T.alloc_var("float32")
                    au = bf[i] * T.Cast("float32", V[h, c * chunk + i, d])
                    for m in T.serial(chunk):
                        if m < i:
                            au = au - L[i, m] * T.Cast("float32", UW[m, d])
                    UW[i, d] = T.Cast("float16", au)
                for i in T.serial(chunk):
                    Ubuf[h, c, i, d] = UW[i, d]
            T.sync_threads()
            for d in T.Parallel(D):
                for i in T.serial(chunk):
                    aw = T.alloc_var("float32")
                    aw = bf[i] * T.Cast("float32", K[hk, c * chunk + i, d]) * egc[i]
                    for m in T.serial(chunk):
                        if m < i:
                            aw = aw - L[i, m] * T.Cast("float32", UW[m, d])
                    UW[i, d] = T.Cast("float16", aw)
                    T.sync_threads()
                for i in T.serial(chunk):
                    Wbuf[h, c, i, d] = UW[i, d]
                    KDbuf[h, c, i, d] = T.Cast("float16", T.Cast("float32", K[hk, c * chunk + i, d]) * T.exp(gc[chunk - 1] - gc[i]))

    return gdn_prep_kernel


def make_gdn_seq_kernel(TOK: int, Hk: int, Hv: int, D: int, chunk: int, threads: int):
    @T.prim_func
    def gdn_seq_kernel(
        Q: T.Tensor((Hk, TOK, D), "float16"),
        Ubuf: T.Tensor((Hv, TOK // chunk, chunk, D), "float16"),
        Wbuf: T.Tensor((Hv, TOK // chunk, chunk, D), "float16"),
        KDbuf: T.Tensor((Hv, TOK // chunk, chunk, D), "float16"),
        A2buf: T.Tensor((Hv, TOK // chunk, chunk, chunk), "float16"),
        EgcBuf: T.Tensor((Hv, TOK // chunk, chunk), "float32"),
        EglBuf: T.Tensor((Hv, TOK // chunk), "float32"),
        S: T.Tensor((Hv, D, D), "float32"),
        O: T.Tensor((Hv, TOK, D), "float16"),
    ):
        T.func_attr({"tl.opencl.assume_inbounds": 1})
        with T.Kernel(D // 32, Hv, threads=threads) as (dvb, h):
            hk = h % Hk
            kdl = T.alloc_shared((chunk, D), "float16")
            vn = T.alloc_shared((chunk, 32), "float16")

            for c in T.serial(TOK // chunk):
                for idx in T.Parallel(chunk * D):
                    kdl[idx // D, idx % D] = KDbuf[h, c, idx // D, idx % D]
                T.sync_threads()

                # pass1 + output: 32 token/column-vector threads, 4 lanes per token.
                for tid in T.Parallel(threads):
                    t = tid // 4
                    dvg = tid % 4
                    dv0 = dvb * 32 + dvg * 8
                    acc = T.alloc_local((8,), "float32")
                    acco = T.alloc_local((8,), "float32")
                    for lane in T.vectorized(8):
                        acc[lane] = 0.0
                        acco[lane] = 0.0
                    for dk in T.serial(D):
                        wv = T.Cast("float32", Wbuf[h, c, t, dk])
                        qv = T.Cast("float32", Q[hk, c * chunk + t, dk])
                        for lane in T.vectorized(8):
                            sv = S[h, dk, dv0 + lane]
                            acc[lane] = acc[lane] + wv * sv
                            acco[lane] = acco[lane] + qv * sv
                    for lane in T.vectorized(8):
                        vn[t, dvg * 8 + lane] = T.Cast("float16", T.Cast("float32", Ubuf[h, c, t, dv0 + lane]) - acc[lane])
                T.sync_threads()

                for tid2 in T.Parallel(threads):
                    t2 = tid2 // 4
                    dvg2 = tid2 % 4
                    dv02 = dvb * 32 + dvg2 * 8
                    out8 = T.alloc_local((8,), "float32")
                    for lane in T.vectorized(8):
                        out8[lane] = 0.0
                    for j in T.serial(chunk):
                        if j <= t2:
                            a2 = T.Cast("float32", A2buf[h, c, t2, j])
                            for lane in T.vectorized(8):
                                out8[lane] = out8[lane] + a2 * T.Cast("float32", vn[j, dvg2 * 8 + lane])
                    # Recompute inter output here to keep the first parallel loop simple.
                    for dk in T.serial(D):
                        qv = T.Cast("float32", Q[hk, c * chunk + t2, dk])
                        for lane in T.vectorized(8):
                            out8[lane] = out8[lane] + EgcBuf[h, c, t2] * qv * S[h, dk, dv02 + lane]
                    for lane in T.vectorized(8):
                        O[h, c * chunk + t2, dv02 + lane] = T.Cast("float16", out8[lane])
                T.sync_threads()

                # pass2: update S.  Same mapping as gdn.cl, four 8-dk groups per dv column.
                for tid3 in T.Parallel(threads):
                    dk8 = tid3 // 32
                    dvl = tid3 % 32
                    dv = dvb * 32 + dvl
                    for sb in T.serial(4):
                        dk0 = sb * 32 + dk8 * 8
                        m8 = T.alloc_local((8,), "float32")
                        for lane in T.vectorized(8):
                            m8[lane] = 0.0
                        for tt in T.serial(chunk):
                            vnv = T.Cast("float32", vn[tt, dvl])
                            for lane in T.vectorized(8):
                                m8[lane] = m8[lane] + T.Cast("float32", kdl[tt, dk0 + lane]) * vnv
                        for lane in T.vectorized(8):
                            S[h, dk0 + lane, dv] = EglBuf[h, c] * S[h, dk0 + lane, dv] + m8[lane]
                T.sync_threads()

    return gdn_seq_kernel


def syntax_check(path: Path) -> int | None:
    clang_bin = os.environ.get("CLANG_BIN", "/usr/lib/llvm-13/bin/clang")
    if not Path(clang_bin).exists():
        clang = subprocess.run(["bash", "-lc", "command -v clang"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        clang_bin = clang.stdout.strip() if clang.returncode == 0 else ""
    if not clang_bin:
        print("CLANG_SYNTAX_SKIP no clang")
        return None
    cmd = [clang_bin, "-x", "cl", "-cl-std=CL3.0", "-fsyntax-only"]
    headers = Path("/root/project/backend/gpu/OpenCL-Headers")
    if headers.exists():
        cmd += ["-I", str(headers)]
    cmd.append(str(path))
    res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(res.stdout, end="")
    print(f"CLANG_SYNTAX_RC {res.returncode}")
    return res.returncode


def emit_one(func, out: Path) -> None:
    with tvm.target.Target("opencl"), tvm.transform.PassContext(config=PASS_CFG):
        artifact = tilelang.lower(func, target="opencl", enable_device_compile=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(artifact.kernel_source, encoding="utf-8")
    print(f"SOURCE_PATH {out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit TileLang OpenCL GDN prefill kernels.")
    ap.add_argument("--t", type=int, default=1024)
    ap.add_argument("--hk", type=int, default=16)
    ap.add_argument("--hv", type=int, default=32)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()
    if args.t % args.chunk or args.threads != 128 or args.d != 128 or args.chunk != 32:
        raise SystemExit("current mapping requires T%chunk==0, threads=128, D=128, chunk=32")
    outs = [args.out_dir / "gdn_prep.cl", args.out_dir / "gdn_seq.cl"]
    emit_one(make_gdn_prep_kernel(args.t, args.hk, args.hv, args.d, args.chunk, args.threads), outs[0])
    emit_one(make_gdn_seq_kernel(args.t, args.hk, args.hv, args.d, args.chunk, args.threads), outs[1])
    if not args.skip_clang:
        for p in outs:
            rc = syntax_check(p)
            if rc not in (0, None):
                raise SystemExit(f"clang syntax failed for {p}: {rc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
