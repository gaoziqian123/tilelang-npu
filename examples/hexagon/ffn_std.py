#!/usr/bin/env python3
"""Emit a standard-construct TileLang Hexagon fused SwiGLU FFN kernel.

This example follows ``gdn_std.py``: it is written with TileLang language
constructs only (T.copy/T.gemm/T.parallel/T.serial/T.vectorized/T.exp and scalar
arithmetic).  It does not embed handwritten C strings and does not call
user-visible T.hexagon leaf intrinsics.

Fixed anchor shape: M=1024, K=2560, FF=9216.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "ffn_std.c"
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


@tilelang.jit(out_idx=[4], target="hexagon", execution_backend="aot")
def ffn_std(M: int = 1024, K: int = 2560, FF: int = 9216, ff_panel: int = 256, k_panel: int = 128):
    @T.prim_func
    def main(
        X: T.Tensor((M, K), T.float16),
        Wg: T.Tensor((FF, K), T.float16),
        Wu: T.Tensor((FF, K), T.float16),
        Wd: T.Tensor((K, FF), T.float16),
        Slab: T.Tensor((2 * M, FF), T.float16),
    ):
        with T.Kernel(1, threads=1):
            # Phase-1 resident panels: one activation row block and one gate/up
            # weight panel pair.  HMX calls stay in the caller thread.
            X_a = T.alloc_shared((32, K), T.float16, layout="ah")
            Wg_b = T.alloc_shared((ff_panel, K), T.float16, layout="wh")
            Wu_b = T.alloc_shared((ff_panel, K), T.float16, layout="wh")
            Gate = T.alloc_shared((32, ff_panel), T.float16)
            Up = T.alloc_shared((32, ff_panel), T.float16)
            Gate_acc = T.alloc_fragment((32, ff_panel), T.float32)
            Up_acc = T.alloc_fragment((32, ff_panel), T.float32)

            # Phase-2 resident buffer.  H is stored row-major in the slab so it
            # can be staged to AH by the standard global-rm -> VTCM-AH recipe.
            H_a = T.alloc_shared((32, FF), T.float16, layout="ah")
            Wd_b = T.alloc_shared((k_panel, FF), T.float16, layout="wh")
            Ytile = T.alloc_shared((32, k_panel), T.float16)
            Y_acc = T.alloc_fragment((32, k_panel), T.float32)

            # Phase 1: h = silu(x @ Wg^T) * (x @ Wu^T), stored in global
            # row-major scratch Slab[0:M, 0:FF].
            for p in T.serial(FF // ff_panel):
                # Stage each weight panel once, before the row-block sweep.  Use
                # explicit regions so each pool job stages a disjoint 32-row WH
                # slice instead of a degenerate job-0 full-panel copy.
                for wg_job in T.parallel(ff_panel // 32):
                    T.copy(Wg[p * ff_panel + wg_job * 32 : p * ff_panel + (wg_job + 1) * 32, 0:K], Wg_b[wg_job * 32 : (wg_job + 1) * 32, 0:K], layout=("rm", "wh"))
                for wu_job in T.parallel(ff_panel // 32):
                    T.copy(Wu[p * ff_panel + wu_job * 32 : p * ff_panel + (wu_job + 1) * 32, 0:K], Wu_b[wu_job * 32 : (wu_job + 1) * 32, 0:K], layout=("rm", "wh"))
                for mb in T.Pipelined(M // 32, num_stages=2, order=[0, 1, 2, 3, 4, 5], stage=[0, 1, 1, 1, 1, 1]):
                    # Activation staging is per row block.  Keep it in a pool
                    # phase so the caller thread only runs HMX chains.
                    for xjob in T.parallel(8):
                        T.copy(X[mb * 32 : (mb + 1) * 32, xjob * (K // 8) : (xjob + 1) * (K // 8)], X_a[0:32, xjob * (K // 8) : (xjob + 1) * (K // 8)], layout=("rm", "ah"))
                    T.gemm(X_a, Wg_b, Gate_acc, transpose_B=True, clear_accum=True)
                    T.copy(Gate_acc, Gate, layout=("ah", "rm"))
                    T.gemm(X_a, Wu_b, Up_acc, transpose_B=True, clear_accum=True)
                    T.copy(Up_acc, Up, layout=("ah", "rm"))
                    for r in T.parallel(32):
                        for c in T.vectorized(ff_panel):
                            Slab[mb * 32 + r, p * ff_panel + c] = T.Cast(
                                "float16",
                                (T.Cast("float32", Gate[r, c]) / (T.Cast("float32", 1.0) + T.exp(-T.Cast("float32", Gate[r, c]))))
                                * T.Cast("float32", Up[r, c]),
                            )

            # Phase 2: y = h @ Wd^T.  The output y lives after h in the same
            # slab at Slab[M:M+M, 0:K]; columns [K:FF) are padding kept only to
            # preserve a 2-D standard-copy-friendly slab.
            for kp in T.serial(K // k_panel):
                # Wd is independent of the row-block loop; keep it resident for
                # all M/32 row blocks for this output-channel panel.
                for wd_job in T.parallel(k_panel // 32):
                    T.copy(Wd[kp * k_panel + wd_job * 32 : kp * k_panel + (wd_job + 1) * 32, 0:FF], Wd_b[wd_job * 32 : (wd_job + 1) * 32, 0:FF], layout=("rm", "wh"))
                for rb2 in T.Pipelined(M // 32, num_stages=2, order=[0, 1, 2, 3], stage=[0, 1, 1, 1]):
                    for hjob in T.parallel(36):
                        T.copy(Slab[rb2 * 32 : (rb2 + 1) * 32, hjob * (FF // 36) : (hjob + 1) * (FF // 36)], H_a[0:32, hjob * (FF // 36) : (hjob + 1) * (FF // 36)], layout=("rm", "ah"))
                    T.gemm(H_a, Wd_b, Y_acc, transpose_B=True, clear_accum=True)
                    T.copy(Y_acc, Ytile, layout=("ah", "rm"))
                    # Slab is intentionally the single out buffer.
                    for ry in T.parallel(32):
                        for cy in T.vectorized(k_panel):
                            Slab[M + rb2 * 32 + ry, kp * k_panel + cy] = Ytile[ry, cy]

    return main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit standard-construct Hexagon fused SwiGLU FFN example.")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=2560)
    parser.add_argument("--ff", type=int, default=9216)
    parser.add_argument("--ff-panel", type=int, default=256)
    parser.add_argument("--k-panel", type=int, default=128)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-clang", action="store_true", help="Skip hexagon-clang -fsyntax-only.")
    parser.add_argument("--self-check", action="store_true", help="Run structural checks on generated C.")
    return parser.parse_args()


def emit(args: argparse.Namespace) -> str:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_HEXAGON_EMIT_C"] = str(args.out)
    with PassContext(config={PassConfigKey.TL_HEXAGON_PROF.value: True}):
        artifact = tilelang.engine.lower(
            ffn_std.get_tir(M=args.m, K=args.k, FF=args.ff, ff_panel=args.ff_panel, k_panel=args.k_panel)
            .with_attr("global_symbol", "attnops_tl_ffn_std"),
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
    checks.append(("async pool starts emitted", src.count("attnops_pool_start_ctx(") >= 2))
    checks.append(("async pool joins emitted", src.count("attnops_pool_join();") >= 2))
    checks.append(("sync prologue/epilogue pools emitted", "attnops_pool_run_ctx(" in src))
    checks.append(("hmx present", "hrt_hmx_mm_f16" in src and "hrt_acc_read_f16" in src))
    worker_blob = src.split("int attnops_tl_ffn_std", 1)[0]
    checks.append(("hmx outside worker functions", "hrt_hmx_mm_f16" not in worker_blob))
    checks.append(("standard silu lowering", "expf(" in src or "hrt_exp" in src))
    checks.append(("copy recipe staging", "hrt_tl" in src and ("copy" in src or "stage" in src)))
    checks.append(("prof counters", "tl.hexagon_prof profile slots" in src and "tl_prof_phase_t0" in src and "prof[5.." in src))
    checks.append(("no handwritten ffn leaf", "attnops_ffn(" not in src and "fn_silu_ah_worker" not in src))
    checks.append(("rotated pipeline buffers", "floormod" in src or "% 2" in src))
    async_start_pos = src.find("attnops_pool_start_ctx(")
    async_join_pos = src.find("attnops_pool_join();", async_start_pos)
    sync_after_join_pos = src.find("attnops_pool_run_ctx(", async_join_pos)
    checks.append(("join before following sync pool", async_start_pos >= 0 and async_join_pos > async_start_pos and sync_after_join_pos > async_join_pos))
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
    if args.ff_panel not in (256, 512):
        print("FF_PANEL_FAIL expected 256 or 512")
        return 1
    src = emit(args)
    print(f"EMIT_OK path={args.out} lines={len(src.splitlines())}")
    rc = structural_check(src) if args.self_check else 0
    rc = run_syntax_check(args.out, args.skip_clang) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
