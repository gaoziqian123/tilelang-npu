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


@tilelang.jit(out_idx=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], target="hexagon", execution_backend="aot")
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
        APg: T.Tensor((Hv, TOK // chunk, 64, chunk), T.float16),
        WUg: T.Tensor((Hv, TOK // chunk, 64, D), T.float16),
        Otmp: T.Tensor((Hv, TOK // chunk, chunk, D), T.float16),
        DSg: T.Tensor((Hv, D, D), T.float16),
        Wg: T.Tensor((Hv, TOK // chunk, chunk, D), T.float32),
        Pg: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32),
        Ag: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32),
        eGg: T.Tensor((Hv, TOK // chunk, chunk), T.float32),
        eGivg: T.Tensor((Hv, TOK // chunk, chunk), T.float32),
        betag: T.Tensor((Hv, TOK // chunk, chunk), T.float32),
        eGCg: T.Tensor((Hv, TOK // chunk), T.float32),
    ):
        with T.Kernel(1, threads=1):
            # Full standard-construct GDN route-b.  HMX GEMMs are kept on the
            # caller thread; scalar triangular recurrences use global scratch so
            # they never scalar-load VTCM (R2).  Persistent fp32 state and its
            # fp16 AH shadow live in VTCM across chunks.
            A_a = T.alloc_shared((32, 128), T.float16, layout="ah")
            B_a = T.alloc_shared((32, 128), T.float16, layout="wh")
            KH_a = T.alloc_shared((Hv, 32, 128), T.float16, layout="ah")
            QH_a = T.alloc_shared((Hv, 32, 128), T.float16, layout="ah")
            KL_b = T.alloc_shared((Hv, 32, 128), T.float16, layout="wh")
            P_a = T.alloc_shared((Hv, 32, 32), T.float16, layout="ah")
            Wt_b = T.alloc_shared((Hv, 128, 32), T.float16, layout="wh")
            Kt_b = T.alloc_shared((128, 32), T.float16, layout="ah")
            state = T.alloc_shared((Hv, D, D), T.float32)
            state_wh = T.alloc_shared((Hv, D, D), T.float16, layout="wh")
            KH = T.alloc_shared((Hv, chunk, D), T.float32)
            QH = T.alloc_shared((Hv, chunk, D), T.float32)
            KL = T.alloc_shared((Hv, chunk, D), T.float32)
            Khat = T.alloc_shared((Hv, chunk, D), T.float32)
            AP_acc = T.alloc_fragment((32, 32), T.float32)
            WU_acc = T.alloc_fragment((32, 128), T.float32)
            O_acc = T.alloc_fragment((32, 128), T.float32)
            DS_acc = T.alloc_fragment((128, 128), T.float32)

            # Phase 0: initialize fp32 state and AH shadow once.
            for hinit in T.serial(Hv):
                for r in T.serial(D):
                    for d in T.vectorized(D):
                        state[hinit, r, d] = S0[hinit, r, d]
                T.copy(S0[hinit, 0, 0], state_wh[hinit, 0, 0], layout=("rm", "wh"))

            for c in T.serial(TOK // chunk):
                t0 = c * chunk

                # Phase 1: per-head cumsum/exp factors, row-scaled K/Q tiles,
                # and k_hat fold for the chunk-end state GEMM.
                for hvp in T.parallel(Hv):
                    hk_p = hvp % Hk
                    eGCg[hvp, c] = 0.0
                    for i in T.serial(chunk):
                        eGCg[hvp, c] = eGCg[hvp, c] + G[hvp, t0 + i]
                        eGg[hvp, c, i] = T.exp(T.max(eGCg[hvp, c], -60.0))
                        eGivg[hvp, c, i] = T.exp(-T.max(eGCg[hvp, c], -60.0))
                        betag[hvp, c, i] = B[hvp, t0 + i]
                    eGCg[hvp, c] = T.exp(T.max(eGCg[hvp, c], -60.0))
                    for i in T.serial(chunk):
                        for d in T.vectorized(D):
                            KH[hvp, i, d] = betag[hvp, c, i] * eGg[hvp, c, i] * T.Cast("float32", K[hk_p, t0 + i, d])
                            KL[hvp, i, d] = eGivg[hvp, c, i] * T.Cast("float32", K[hk_p, t0 + i, d])
                            QH[hvp, i, d] = eGg[hvp, c, i] * T.Cast("float32", Q[hk_p, t0 + i, d])
                            Khat[hvp, i, d] = eGCg[hvp, c] * eGivg[hvp, c, i] * T.Cast("float32", K[hk_p, t0 + i, d])

                # Phase 2 AP staging is pure HVX and can run on pool workers.
                for hv_ap_stage in T.parallel(Hv):
                    T.copy(KL[hv_ap_stage, 0, 0], KL_b[hv_ap_stage, 0, 0], layout=("rm", "wh"), annotations={"hexagon.copy.trans": 1})
                    T.copy(KH[hv_ap_stage, 0, 0], KH_a[hv_ap_stage, 0, 0], layout=("rm", "ah"))
                    T.copy(QH[hv_ap_stage, 0, 0], QH_a[hv_ap_stage, 0, 0], layout=("rm", "ah"))

                for hv in T.serial(Hv):
                    hk = hv % Hk
                    # Phase 2: AP=[Kh;Qh]@Kl^T and WU=[K;Q]@S^T.  AP/WU are
                    # committed to global fp16 scratch because phase 3 needs
                    # scalar triangular coefficients without VTCM scalar reads.
                    T.gemm(KH_a[hv, 0, 0], KL_b[hv, 0, 0], AP_acc, transpose_B=True, clear_accum=True)
                    T.copy(AP_acc, APg[hv, c, 0, 0], layout=("ah", "rm"))
                    T.gemm(QH_a[hv, 0, 0], KL_b[hv, 0, 0], AP_acc, transpose_B=True, clear_accum=True)
                    T.copy(AP_acc, APg[hv, c, chunk, 0], layout=("ah", "rm"))


                # Reuse KH_a/QH_a storage for raw K/Q staging used by WU.
                for hv_wu_stage in T.parallel(Hv):
                    hk_wu_stage = hv_wu_stage % Hk
                    T.copy(K[hk_wu_stage, t0, 0], KH_a[hv_wu_stage, 0, 0], layout=("rm", "ah"))
                    T.copy(Q[hk_wu_stage, t0, 0], QH_a[hv_wu_stage, 0, 0], layout=("rm", "ah"))

                for hv_wu in T.serial(Hv):
                    T.gemm(KH_a[hv_wu, 0, 0], state_wh[hv_wu, 0, 0], WU_acc, transpose_B=True, clear_accum=True)
                    T.copy(WU_acc, WUg[hv_wu, c, 0, 0], layout=("ah", "rm"))
                    T.gemm(QH_a[hv_wu, 0, 0], state_wh[hv_wu, 0, 0], WU_acc, transpose_B=True, clear_accum=True)
                    T.copy(WU_acc, WUg[hv_wu, c, chunk, 0], layout=("ah", "rm"))

                # Phase 3: triangular masks, affine RHS, and serial forward solve.
                # Masks run as a 1024-lane vector pass over the (32,32) plane
                # while indexing the logical matrix dimensions directly; lane
                # index l maps to row l//32 / col l%32.
                for hvp3 in T.parallel(Hv):
                    for l in T.vectorized(chunk * chunk):
                        Pg[hvp3, c, l // chunk, l % chunk] = T.if_then_else(
                            l % chunk <= l // chunk,
                            T.Cast("float32", APg[hvp3, c, chunk + l // chunk, l % chunk]),
                            T.Cast("float32", 0.0))
                        Ag[hvp3, c, l // chunk, l % chunk] = T.if_then_else(
                            l % chunk < l // chunk,
                            T.Cast("float32", APg[hvp3, c, l // chunk, l % chunk]),
                            T.Cast("float32", 0.0))
                    for i in T.serial(chunk):
                        for d in T.vectorized(D):
                            Wg[hvp3, c, i, d] = betag[hvp3, c, i] * T.Cast("float32", V[hvp3, t0 + i, d]) - betag[hvp3, c, i] * eGg[hvp3, c, i] * T.Cast("float32", WUg[hvp3, c, i, d])
                    for i in T.serial(chunk):
                        for j in T.serial(i):
                            for d in T.vectorized(D):
                                Wg[hvp3, c, i, d] = Wg[hvp3, c, i, d] - Ag[hvp3, c, i, j] * Wg[hvp3, c, j, d]

                # Phase 4 staging is pure HVX.  Wt_b is staged here and reused
                # by phase 6; only Khat needs an extra staging pool below.
                for hv4_stage in T.parallel(Hv):
                    T.copy(Pg[hv4_stage, c, 0, 0], P_a[hv4_stage, 0, 0], layout=("rm", "ah"))
                    T.copy(Wg[hv4_stage, c, 0, 0], Wt_b[hv4_stage, 0, 0], layout=("rm", "wh"))

                for hv4 in T.serial(Hv):
                    # Phase 4: O' = tril(P) @ w.
                    T.gemm(P_a[hv4, 0, 0], Wt_b[hv4, 0, 0], O_acc, transpose_B=True, clear_accum=True)
                    T.copy(O_acc, Otmp[hv4, c, 0, 0], layout=("ah", "rm"))

                for hv6 in T.serial(Hv):
                    # Phase 6: DeltaS = k_hat^T @ w.
                    T.copy(Khat[hv6, 0, 0], Kt_b, layout=("rm", "ah"), annotations={"hexagon.copy.trans": 1})
                    T.gemm(Kt_b, Wt_b[hv6, 0, 0], DS_acc, transpose_B=True, clear_accum=True)
                    T.copy(DS_acc, DSg[hv6, 0, 0], layout=("ah", "rm"))

                # Phases 5 and 7: output combine, fp32 state update, AH shadow
                # refresh for the next chunk, and final state store on c==last.
                for hvp7 in T.parallel(Hv):
                    for i in T.serial(chunk):
                        for d in T.vectorized(D):
                            O[hvp7, t0 + i, d] = T.Cast("float16", eGg[hvp7, c, i] * T.Cast("float32", WUg[hvp7, c, chunk + i, d]) + T.Cast("float32", Otmp[hvp7, c, i, d]))
                    for r in T.serial(D):
                        for d in T.vectorized(D):
                            state[hvp7, r, d] = eGCg[hvp7, c] * state[hvp7, r, d] + T.Cast("float32", DSg[hvp7, r, d])
                    if c == TOK // chunk - 1:
                        for r2 in T.serial(D):
                            for d2 in T.vectorized(D):
                                S1[hvp7, r2, d2] = state[hvp7, r2, d2]
                for hv7 in T.serial(Hv):
                    T.copy(state[hv7, 0, 0], state_wh[hv7, 0, 0], layout=("rm", "wh"))

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
