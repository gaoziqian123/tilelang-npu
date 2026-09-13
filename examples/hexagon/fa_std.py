#!/usr/bin/env python3
"""Emit a standard-construct TileLang Hexagon GQA flash-attention prefill.

This is the pure-standard companion to the handwritten ``attnops_fa256.c``
kernel: it uses only TileLang constructs (T.copy/T.gemm/T.parallel/T.serial/
T.vectorized/T.reduce_max/T.reduce_sum/T.exp and scalar arithmetic) and lets the
Hexagon backend lower staging, reductions, HMX GEMMs and worker-pool phases.

Fixed v1 anchor shape: S=1024, HQ=16, HKV=4, D=256, q/kv tile=32.  KV head g
serves q heads 4g..4g+3.  Inputs and output are logical row-major fp16 in the
single generic slab ABI; staging to AH/WH happens inside this kernel.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "fa_std.c"
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


@tilelang.jit(out_idx=[3, 4, 5, 6, 7, 8, 9, 10, 11], target="hexagon", execution_backend="aot")
def fa_std(S: int = 1024, HQ: int = 16, HKV: int = 4, D: int = 256, tile: int = 32):
    @T.prim_func
    def main(
        Q: T.Tensor((HQ * S, D), T.float16),
        K: T.Tensor((HKV * S, D), T.float16),
        V: T.Tensor((HKV * S, D), T.float16),
        O: T.Tensor((HQ * S, D), T.float16),
        Pbuf: T.Tensor((S // tile, tile, 4 * tile), T.float16),
        Mus: T.Tensor((S // tile, tile), T.float32),
        Mfin: T.Tensor((tile,), T.float32),
        Lbuf: T.Tensor((tile,), T.float32),
        ScoreRow: T.Tensor((tile,), T.float16),
        Mtmp: T.Tensor((1,), T.float32),
        Obuf: T.Tensor((tile, D), T.float16),
        Vt: T.Tensor((HKV, D, S), T.float16),
        Oacc: T.Tensor((tile, D), T.float32),
    ):
        with T.Kernel(1, threads=1):
            # Per-q-tile resident operands.  All global inputs are logical
            # row-major and are staged with standard T.copy recipes.
            Q_a = T.alloc_shared((tile, D), T.float16, layout="ah")
            # K/V are staged once per KV group.  K_all is logical [S,D] in WH
            # layout; kt sweeps pass point views K_all[kt*32, 0].  V_all is
            # logical V^T [D,S] in WH layout; P@V passes point views
            # V_all[0, kt*32].
            K_all = T.alloc_shared((S, D), T.float16, layout="wh")
            # P@V is padded to K=128 because the standard f16 rm->AH/WH staging
            # recipe is 64-column granular.  P columns [32,128) and V rows
            # [32,128) are zero, so the math is unchanged.  HMX has no
            # non-transposed B mode (WH always encodes an [N,K] source), so
            # V is staged pre-transposed: Vpad[d, kv] and transpose_B=True.
            V_all = T.alloc_shared((D, S), T.float16, layout="wh")
            P_a = T.alloc_shared((tile, 4 * tile), T.float16, layout="ah")
            Score = T.alloc_fragment((tile, tile), T.float32)
            Out = T.alloc_fragment((tile, D), T.float32)

            # Four KV groups, each serving four query heads (GQA 16/4).
            for g in T.serial(HKV):
                # Phase G0: stage the full K[g] head into VTCM WH once.  Each
                # worker owns one 32-row tile and one 64-column slice so the
                # lowering uses the rm->WH sliced multi-row-block recipe.
                for k_all_job in T.parallel((S // tile) * (D // 64)):
                    kt_all = k_all_job // (D // 64)
                    kc_all = k_all_job % (D // 64)
                    T.copy(
                        K[g * S + kt_all * tile : g * S + (kt_all + 1) * tile,
                          kc_all * 64 : (kc_all + 1) * 64],
                        K_all[kt_all * tile : (kt_all + 1) * tile,
                              kc_all * 64 : (kc_all + 1) * 64],
                        layout=("rm", "wh"),
                    )

                # Phase G1: build full-head V^T in global scratch once per KV
                # group.  This is still scalar, but runs four times total
                # instead of once per (query tile, value tile) pair.
                for vt_d in T.parallel(D):
                    for vt_s in T.serial(S):
                        Vt[g, vt_d, vt_s] = V[g * S + vt_s, vt_d]

                # Phase G2: stage full V^T into VTCM WH once.  Use 64-column
                # slices so every point view later is 128B-aligned.
                for v_all_stage in T.parallel(S // 64):
                    T.copy(
                        Vt[g, 0:D, v_all_stage * 64 : (v_all_stage + 1) * 64],
                        V_all[0:D, v_all_stage * 64 : (v_all_stage + 1) * 64],
                        layout=("rm", "wh"),
                    )

                for hqg in T.serial(HQ // HKV):
                    hq = g * (HQ // HKV) + hqg
                    for qt in T.Pipelined(S // tile, num_stages=2, order=[0, 1, 2, 3, 4, 5, 6], stage=[0, 1, 1, 1, 1, 1, 1]):
                        q0 = qt * tile

                        # Phase 0: Q staging.  v1 applies the 1/sqrt(256)
                        # scale immediately after score readback instead of
                        # materializing a scaled VTCM-RM -> VTCM-AH copy route
                        # (that route is intentionally not exposed by the
                        # current standard Hexagon copy lowering).
                        for q_stage in T.parallel(D // 64):
                            T.copy(
                                Q[hq * S + q0 : hq * S + q0 + tile, q_stage * 64 : (q_stage + 1) * 64],
                                Q_a[0:tile, q_stage * 64 : (q_stage + 1) * 64],
                                layout=("rm", "ah"),
                            )

                        # Initialize online softmax state.
                        # Keep this scalar so the qt+1 Q async stage in the
                        # pipelined steady loop is not immediately joined by a
                        # following worker-pool phase.
                        for ri in T.serial(tile):
                            Mfin[ri] = -32768.0
                            Lbuf[ri] = 0.0

                        # Phase A: score every visible KV tile, causal-mask the
                        # diagonal tile, update running row max and store
                        # P_kt = exp(score - m_used_kt) in global scratch.
                        for kt in T.serial(qt + 1):
                            T.gemm(
                                Q_a,
                                K_all[kt * tile : (kt + 1) * tile, 0:D],
                                Score,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.copy(Score, Pbuf[kt, 0, 0], layout=("ah", "rm"))

                            # Mask+scale, running max, exp and pad-zeroing are
                            # row-parallel pool jobs (one row per job): scalar
                            # global ops are unchanged, but six workers hide
                            # ~6x of the latency.  Job r exclusively owns
                            # Mfin[r]/Lbuf[r]/Pbuf[kt, r, :], so no races.
                            # The pad columns [32,128) are zeroed here only for
                            # kt slots first touched this qt; a slot reused from
                            # an earlier qt already has zero pad (nothing else
                            # writes cols >=32).
                            for rmask in T.parallel(tile):
                                for cmask in T.serial(tile):
                                    if kt == qt and cmask > rmask:
                                        Pbuf[kt, rmask, cmask] = T.Cast("float16", -32768.0)
                                    else:
                                        Pbuf[kt, rmask, cmask] = T.Cast("float16", T.Cast("float32", Pbuf[kt, rmask, cmask]) / 16.0)
                                # Row max stays scalar (T.reduce_max on a
                                # sliced region trips a known generic
                                # LowerTileOp slice-substitution bug), but runs
                                # inside the row-parallel pool job; Mus[kt, r]
                                # is job-private so it doubles as the
                                # accumulator (Mtmp is shared, do not use).
                                Mus[kt, rmask] = T.Cast("float32", -32768.0)
                                for cmax in T.serial(tile):
                                    Mus[kt, rmask] = T.max(Mus[kt, rmask], T.Cast("float32", Pbuf[kt, rmask, cmax]))
                                Mfin[rmask] = T.max(Mfin[rmask], Mus[kt, rmask])
                                Mus[kt, rmask] = Mfin[rmask]
                                for cprob in T.serial(tile):
                                    Pbuf[kt, rmask, cprob] = T.Cast(
                                        "float16",
                                        T.exp(T.Cast("float32", Pbuf[kt, rmask, cprob]) - Mfin[rmask]),
                                    )
                                # Pbuf rows have stride 128 fp16.  The live score span
                                # is columns [0,32), while the padded P@V K dimension
                                # is [0,128).  Do not use HVX vector stores starting at
                                # column 32: that address is only 64B-aligned, and HVX
                                # stores silently align down to 128B, clobbering the
                                # live score row.  Scalar global stores are slower but
                                # keep the producer/consumer region intact.
                                # Pad columns [32,128) are zeroed on the first
                                # qt that touches this slot (qt == kt); from
                                # then on no phase writes cols >=32, so the pad
                                # stays zero for all later qt.
                                if kt == qt:
                                    for czero in T.serial(3 * tile):
                                        Pbuf[kt, rmask, tile + czero] = T.Cast("float16", 0.0)

                        # Phase B: online rescale to the final row max and row
                        # normalizer l.  v1 uses a scalar row sum here because
                        # Hexagon's current standard reduce_sum helper is 128
                        # lanes only; P stays fp16 for the HMX P@V pass.
                        for kt2 in T.serial(qt + 1):
                            for rsum in T.parallel(tile):
                                alpha = T.exp(Mus[kt2, rsum] - Mfin[rsum])
                                for csum in T.serial(tile):
                                    Pbuf[kt2, rsum, csum] = T.Cast(
                                        "float16", T.Cast("float32", Pbuf[kt2, rsum, csum]) * alpha)
                                for cadd in T.serial(tile):
                                    Lbuf[rsum] = Lbuf[rsum] + T.Cast("float32", Pbuf[kt2, rsum, cadd])

                        # Phase C: O = sum_kt P_kt @ V_kt.  v1 copies each HMX
                        # product to global scratch and accumulates there,
                        # because the standard Hexagon lowering currently
                        # requires every T.gemm to be immediately followed by a
                        # T.copy(acc, ...).
                        # O accumulates in fp32 scratch: per-tile partials are
                        # fp16 (hexkl only has acc_read_f16), but the running
                        # sum must not round-trip through fp16 across kt3.
                        for zrow in T.parallel(tile):
                            for zd in T.serial(D):
                                Oacc[zrow, zd] = 0.0
                        for kt3 in T.serial(qt + 1):
                            for p_stage in T.parallel(2):
                                T.copy(
                                    Pbuf[kt3, 0:tile, p_stage * 64 : (p_stage + 1) * 64],
                                    P_a[0:tile, p_stage * 64 : (p_stage + 1) * 64],
                                    layout=("rm", "ah"),
                                )
                            T.gemm(
                                P_a,
                                V_all[0:D, kt3 * tile : kt3 * tile + 4 * tile],
                                Out,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.copy(Out, Obuf, layout=("ah", "rm"))
                            for arow in T.parallel(tile):
                                for ad in T.serial(D):
                                    Oacc[arow, ad] = Oacc[arow, ad] + T.Cast("float32", Obuf[arow, ad])

                        # Phase D: normalize and write row-major output.
                        for rout in T.parallel(tile):
                            inv_l = 1.0 / Lbuf[rout]
                            for dout in T.serial(D):
                                O[hq * S + q0 + rout, dout] = T.Cast("float16", Oacc[rout, dout] * inv_l)

    return main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit standard-construct Hexagon FA prefill example.")
    parser.add_argument("--s", type=int, default=1024)
    parser.add_argument("--hq", type=int, default=16)
    parser.add_argument("--hkv", type=int, default=4)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-clang", action="store_true", help="Skip hexagon-clang -fsyntax-only.")
    parser.add_argument("--self-check", action="store_true", help="Run structural checks on generated C.")
    return parser.parse_args()


def emit(args: argparse.Namespace) -> str:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_HEXAGON_EMIT_C"] = str(args.out)
    with PassContext(config={PassConfigKey.TL_HEXAGON_PROF.value: True}):
        artifact = tilelang.engine.lower(
            fa_std.get_tir(S=args.s, HQ=args.hq, HKV=args.hkv, D=args.d, tile=args.tile)
            .with_attr("global_symbol", "attnops_tl_fa_std"),
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
    worker_blob = src.split("int attnops_tl_fa_std", 1)[0]
    checks.append(("hmx outside worker functions", "hrt_hmx_mm_f16" not in worker_blob))
    checks.append(("staging recipes present", "hrt_stage_act_hvx_direct" in src and "hrt_stage_f16_rm_to_wh_nt" in src))
    checks.append(("async qt pipeline emitted", "attnops_pool_start_ctx(" in src and "attnops_pool_join();" in src))
    checks.append(("full-head KV staging", "const int k_all_job = job" in src and "const int v_all_stage = job" in src and "const int vt_d = job" in src))
    checks.append(("no per-kt K/V staging", "for (int kt = 0; kt < (qt + 1); kt++)" in src and "const int k_stage = job" not in src and "Vpad" not in src))
    checks.append(("row-parallel softmax pool phases", "const int rmask = job" in src and "const int rsum = job" in src and "const int arow = job" in src and "const int rout = job" in src))
    checks.append(("fp32 exp lowering", "expf(" in src or "hrt_exp" in src))
    checks.append(("causal mask present", "-32768" in src and "rmask < cmask" in src))
    checks.append(("prof counters", "tl.hexagon_prof profile slots" in src and "tl_prof_phase_t0" in src and "prof[5.." in src))
    checks.append(("pure standard: no handwritten fa leaf", "attnops_fa256" not in src and "fa2_" not in src))
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
    if args.s % args.tile or args.tile != 32 or args.d != 256 or args.hq != 16 or args.hkv != 4:
        print("FA_STD_SHAPE_FAIL expected S%32==0, tile=32, HQ=16, HKV=4, D=256")
        return 1
    src = emit(args)
    print(f"EMIT_OK path={args.out} lines={len(src.splitlines())}")
    rc = structural_check(src) if args.self_check else 0
    rc = run_syntax_check(args.out, args.skip_clang) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
