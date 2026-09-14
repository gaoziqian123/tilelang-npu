#!/usr/bin/env python3
"""Emit a standard-construct TileLang Hexagon GQA flash-attention prefill.

v4 is a streaming single-pass flash-attention kernel: for each q tile it walks
visible k/v tiles once, keeps online softmax state and the running output
accumulator resident in VTCM, rescales the VTCM accumulator on max changes, and
normalizes only at writeout.  It deliberately mirrors the official CUDA
TileLang FA shape while staying within standard TileLang constructs
(T.copy/T.gemm/T.parallel/T.serial/T.vectorized/T.Pipelined/T.alloc_var/
T.reduce_max/T.reduce_sum/T.exp/T.if_then_else/T.Cast).

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


@tilelang.jit(out_idx=[3], target="hexagon", execution_backend="aot")
def fa_std(S: int = 1024, HQ: int = 16, HKV: int = 4, D: int = 256, tile: int = 32):
    @T.prim_func
    def main(
        Q: T.Tensor((HQ * S, D), T.float16),
        K: T.Tensor((HKV * S, D), T.float16),
        V: T.Tensor((HKV, D, S), T.float16),
        O: T.Tensor((HQ * S, D), T.float16),
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
            ScoreB = T.alloc_shared((tile, 4 * tile), T.float16)
            OutB = T.alloc_shared((tile, D), T.float16)
            Oacc = T.alloc_shared((tile, D), T.float32)
            # Online softmax state is cross-phase state in VTCM, not global
            # scratch.  Pool workers are non-persistent, so the only legal
            # state across phases is memory; each row owns one 128B f32 slot.
            # Mslot replicates m across all lanes so reduce_max returns m;
            # Lslot keeps l in lane 0 and zeros elsewhere so reduce_sum
            # returns l.  This avoids scalar VTCM loads/stores.
            Mslot = T.alloc_shared((tile, tile), T.float32)
            Lslot = T.alloc_shared((tile, tile), T.float32)
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

                # Phase G1: stage full V^T input into VTCM WH once.  The host
                # supplies V as logical [HKV,D,S] row-major; source ld is S
                # (1024), so the WH sliced _s route is selected.
                for v_all_stage in T.parallel(S // 64):
                    T.copy(
                        V[g, 0:D, v_all_stage * 64 : (v_all_stage + 1) * 64],
                        V_all[0:D, v_all_stage * 64 : (v_all_stage + 1) * 64],
                        layout=("rm", "wh"),
                    )

                for hqg in T.serial(HQ // HKV):
                    hq = g * (HQ // HKV) + hqg
                    for qt in T.Pipelined(S // tile, num_stages=2, order=[0, 1, 2], stage=[0, 1, 1]):
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

                        # Streaming phase: score one visible KV tile, update
                        # online softmax state, rescale the VTCM output
                        # accumulator, then immediately consume P@V.
                        for kt in T.serial(qt + 1):
                            T.gemm(
                                Q_a,
                                K_all[kt * tile : (kt + 1) * tile, 0:D],
                                Score,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.copy(Score, ScoreB[0:tile, 0:tile], layout=("ah", "rm"))

                            # Mask+scale, online max/sum and padded P materialize
                            # in the same VTCM row-major ScoreB buffer.  The pad
                            # lanes [32,128) are biased during exp so they
                            # underflow to zero for the padded P@V K=128 slot.
                            for rmask in T.parallel(tile):
                                if kt == qt:
                                    # Keep the proven scalar diagonal mask: the
                                    # current Hexagon fp16 vector select uses a
                                    # word-lane predicate, so integer lane
                                    # compares would mask f16 column pairs.
                                    for cmask in T.serial(tile):
                                        if cmask > rmask:
                                            ScoreB[rmask, cmask] = T.Cast("float16", -32768.0)
                                        else:
                                            ScoreB[rmask, cmask] = T.Cast("float16", T.Cast("float32", ScoreB[rmask, cmask]) / 16.0)
                                else:
                                    for cmask in T.vectorized(tile):
                                        ScoreB[rmask, cmask] = T.Cast("float16", T.Cast("float32", ScoreB[rmask, cmask]) / 16.0)
                                m_old = T.alloc_var(T.float32)
                                l_old = T.alloc_var(T.float32)
                                if kt == 0:
                                    # qt-local state initialization belongs to
                                    # the softmax row job: pool jobs are not
                                    # persistent and m/l cannot live in thread
                                    # registers across phases.
                                    m_old = -32768.0
                                    l_old = 0.0
                                else:
                                    T.reduce_max(Mslot[rmask, 0], m_old)
                                    T.reduce_sum(Lslot[rmask, 0], l_old)
                                m_row = T.alloc_var(T.float32)
                                T.reduce_max(ScoreB[rmask, 0], m_row)
                                m_new = T.max(m_old, m_row)
                                alpha = T.exp(m_old - m_new)
                                for cprob in T.vectorized(128):
                                    ScoreB[rmask, cprob] = T.Cast(
                                        "float16",
                                        T.exp(T.Cast("float32", ScoreB[rmask, cprob]) - T.if_then_else(cprob >= tile, m_new + 32768.0, m_new)),
                                    )
                                l_acc = T.alloc_var(T.float32)
                                T.reduce_sum(ScoreB[rmask, 0], l_acc)
                                l_new = l_old * alpha + l_acc
                                if kt > 0:
                                    for dscale in T.vectorized(D):
                                        Oacc[rmask, dscale] = Oacc[rmask, dscale] * alpha
                                for sstate in T.vectorized(tile):
                                    Mslot[rmask, sstate] = m_new
                                    Lslot[rmask, sstate] = T.if_then_else(sstate == 0, l_new, 0.0)

                            for p_stage in T.parallel(2):
                                T.copy(
                                    ScoreB[0:tile, p_stage * 64 : (p_stage + 1) * 64],
                                    P_a[0:tile, p_stage * 64 : (p_stage + 1) * 64],
                                    layout=("rm", "ah"),
                                )
                            T.gemm(
                                P_a,
                                V_all[0:D, kt * tile : kt * tile + 4 * tile],
                                Out,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.copy(Out, OutB, layout=("ah", "rm"))
                            for arow in T.parallel(tile):
                                for ad in T.vectorized(D):
                                    if kt == 0:
                                        Oacc[arow, ad] = T.Cast("float32", OutB[arow, ad])
                                    else:
                                        Oacc[arow, ad] = Oacc[arow, ad] + T.Cast("float32", OutB[arow, ad])

                        # Phase D: normalize and write row-major output.
                        for rout in T.parallel(tile):
                            l_final = T.alloc_var(T.float32)
                            T.reduce_sum(Lslot[rout, 0], l_final)
                            inv_l = 1.0 / l_final
                            for dout in T.vectorized(D):
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
    checks.append(("full-head KV staging", "const int k_all_job = job" in src and "const int v_all_stage = job" in src and "const int vt_d = job" not in src))
    checks.append(("no per-kt K/V staging", "for (int kt = 0; kt < (qt + 1); kt++)" in src and "const int k_stage = job" not in src and "Vpad" not in src))
    checks.append(("streaming row-parallel pool phases", "const int rmask = job" in src and "const int p_stage = job" in src and "const int arow = job" in src and "const int rout = job" in src and "const int rscale = job" not in src))
    softmax_blob = src[src.find("static void attnops_tl_fa_std_pool4_worker"):src.find("static void attnops_tl_fa_std_pool5_worker")]
    checks.append(("anti-pattern: no vectorized f16 integer-predicate select for causal mask", "hrt_splat_h(0xF800)" not in softmax_blob))
    checks.append(("vtcm ScoreB to P_a staging", "hrt_stage_act_hvx_direct_strided((const uint8_t *)(((f16 *)(V +" in src))
    checks.append(("vtcm 32-lane reduce helpers", "hrt_reduce_max_f16_32_vtcm" in src and "hrt_reduce_sum_f16_32_vtcm" in src))
    checks.append(("state in VTCM slots", "Mslot" not in src and "Lslot" not in src and "Mfin" not in src and "Lbuf" not in src and "Alpha" not in src))
    checks.append(("fp32 exp lowering", "expf(" in src or "hrt_exp" in src))
    checks.append(("causal mask present", ("-32768" in src or "0xF800" in src) and "cmask" in src and "rmask" in src))
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
