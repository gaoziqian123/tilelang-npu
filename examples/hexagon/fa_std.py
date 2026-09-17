#!/usr/bin/env python3
"""Emit a standard-construct TileLang Hexagon GQA flash-attention prefill.

v6 is v5 with host-prepacked K/V: the host packs K as [HKV*S, D] WH and V^T as
[HKV*D, S] WH before the call, so the kernel stages each head with ONE big
wh->wh block copy (auto-partitioned into pool jobs by HexagonCopyPartition)
instead of many small rm->wh conversion slices.  The math is unchanged: for
each q tile it materializes the entire visible score row block in VTCM, runs
one row-parallel f32-domain softmax phase, stores the probability matrix
directly in AH layout, and consumes it with one chained HMX P@V GEMM.  There
is deliberately no cross-query-tile online state: no Alpha, Mfin/Lbuf slots,
per-kt P staging, or VTCM output accumulator.  It stays within standard
TileLang constructs (T.copy/T.gemm/T.parallel/T.serial/T.vectorized/T.alloc_var/
T.reduce_max/T.reduce_sum/T.exp/T.if_then_else/T.Cast).

Fixed v1 anchor shape: S=1024, HQ=16, HKV=4, D=256, q/kv tile=32.  KV head g
serves q heads 4g..4g+3.  Inputs and output are logical row-major fp16 in the
single generic slab ABI; staging to AH/WH happens inside this kernel.
"""

from __future__ import annotations

import argparse
import builtins
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
        K: T.Tensor((HKV * S, D), T.float16, layout="wh"),
        V: T.Tensor((HKV * D, S), T.float16, layout="wh"),
        O: T.Tensor((HQ * S, D), T.float16),
    ):
        with T.Kernel(1, threads=1):
            # Per-q-tile resident operands.  Q is logical row-major; K and V^T
            # arrive host-prepacked in WH tile layout (see fa_std_test.c
            # pack_rm_to_wh), so staging is a pure block copy.
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
            # v5 materializes the full visible score/probability row block for
            # one query tile.  ScoreFull is row-major fp16 readout from HMX;
            # P_ah stores softmax probabilities directly in AH placement and is
            # consumed by one P@V GEMM with dynamic K=(qt+1)*32.
            ScoreFull = T.alloc_shared((tile, S), T.float16)
            P_ah = T.alloc_shared((tile, S), T.float16, layout="ah")
            ZeroTile = T.alloc_shared((tile, tile), T.float16)
            OutB = T.alloc_shared((tile, D), T.float16)
            RowSum = T.alloc_shared((tile, tile), T.float32)
            Score = T.alloc_fragment((tile, tile), T.float32)
            Out = T.alloc_fragment((tile, D), T.float32)

            for zinit in T.vectorized(tile * tile):
                ZeroTile[zinit // tile, zinit % tile] = T.Cast("float16", 0.0)

            # Four KV groups, each serving four query heads (GQA 16/4).
            for g in T.serial(HKV):
                # Phase G0/G1: K/V arrive host-prepacked in WH layout, so
                # staging is one big wh->wh block copy per head.  There is no
                # manual T.parallel slicing: HexagonCopyPartition rewrites
                # each copy into a worker-pool phase before emission.
                T.copy(
                    K[g * S : (g + 1) * S, 0:D],
                    K_all[0:S, 0:D],
                    layout=("wh", "wh"),
                )
                T.copy(
                    V[g * D : (g + 1) * D, 0:S],
                    V_all[0:D, 0:S],
                    layout=("wh", "wh"),
                )

                for hqg in T.serial(HQ // HKV):
                    hq = g * (HQ // HKV) + hqg
                    for qt in T.unroll(S // tile):
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

                        # Phase 1: materialize all visible score tiles for this
                        # query tile.  There is no cross-kt softmax state.
                        for kt in T.serial(qt + 1):
                            T.gemm(
                                Q_a,
                                K_all[kt * tile : (kt + 1) * tile, 0:D],
                                Score,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.copy(Score, ScoreFull[0:tile, kt * tile : (kt + 1) * tile], layout=("ah", "rm"))

                        # Phase 2: one pool phase for full-row softmax.  Each
                        # job owns one row and runs two whole-row sweeps with a
                        # static S-wide extent.  The f16 vector lowering always
                        # covers a full 128B (64-element) HVX vector, so the
                        # earlier per-32-column sweeps spilled into the live
                        # neighbour tile; S=1024 is an exact multiple of 64.
                        # The causal predicate uses the global column
                        # (qt*tile + rmask), which also biases every invisible
                        # lane to -inf, so pad handling is free.  Reductions
                        # stay 32-lane over the visible tiles only.
                        for rmask in T.parallel(tile):
                            m_row = T.alloc_var(T.float32)
                            m_tile = T.alloc_var(T.float32)
                            # Vector-predicate whitelist has no Mul: keep the
                            # causal threshold in a scalar var (local.var lowers
                            # to a plain C scalar, same as the f32 vars below).
                            qrow = T.alloc_var(T.int32)
                            qrow = qt * tile + rmask
                            for cmask in T.vectorized(S):
                                ScoreFull[rmask, cmask] = T.Cast(
                                    "float16",
                                    T.if_then_else(
                                        cmask > qrow,
                                        -32768.0,
                                        T.Cast("float32", ScoreFull[rmask, cmask]) / 16.0,
                                    ),
                                )
                            m_row = -32768.0
                            for ktm in T.serial(qt + 1):
                                T.reduce_max(ScoreFull[rmask, ktm * tile], m_tile)
                                m_row = T.max(m_row, m_tile)

                            m_new = T.max(m_row, -32768.0)
                            l_row = T.alloc_var(T.float32)
                            l_tile = T.alloc_var(T.float32)
                            for cprob in T.vectorized(S):
                                ScoreFull[rmask, cprob] = T.Cast(
                                    "float16",
                                    T.exp(T.Cast("float32", ScoreFull[rmask, cprob]) - m_new),
                                )
                            l_row = 0.0
                            for kte in T.serial(qt + 1):
                                T.reduce_sum(ScoreFull[rmask, kte * tile], l_tile)
                                l_row = l_row + l_tile
                            l_new = l_row + 0.0
                            for sstate in T.vectorized(tile):
                                RowSum[rmask, sstate] = T.if_then_else(sstate == 0, l_new, 0.0)

                        # Phase 2b: stage the full visible probability tiles
                        # from RM ScoreFull to AH P_ah using the proven 32x32
                        # VTCM-source copy route.  One job owns one visible
                        # 32-column tile; this avoids 64B one-row copies and
                        # keeps phase count per qt small.
                        for p_stage in T.parallel(S // tile):
                            if p_stage > qt:
                                T.copy(ZeroTile[0:tile, 0:tile], P_ah[0:tile, p_stage * tile : (p_stage + 1) * tile], layout=("rm", "ah"))
                            else:
                                T.copy(
                                    ScoreFull[0:tile, p_stage * tile : (p_stage + 1) * tile],
                                    P_ah[0:tile, p_stage * tile : (p_stage + 1) * tile],
                                    layout=("rm", "ah"),
                                )

                        # Phase 3: consume the whole probability row block with
                        # one statically-sized HMX GEMM and read out once per
                        # query tile.  The Python-unrolled qt loop keeps
                        # K=(qt+1)*32 static for TileLang.
                        T.gemm(
                            P_ah[0:tile, 0:S],
                            V_all[0:D, 0:S],
                            Out,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.copy(Out, OutB, layout=("ah", "rm"))

                        # Phase 4: normalize by rowsum and write row-major O.
                        for rout in T.parallel(tile):
                            l_final = T.alloc_var(T.float32)
                            T.reduce_sum(RowSum[rout, 0], l_final)
                            inv_l = 1.0 / l_final
                            for dout in T.vectorized(D):
                                O[hq * S + q0 + rout, dout] = T.Cast("float16", T.Cast("float32", OutB[rout, dout]) * inv_l)

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
    checks.append(("staging recipes present", "hrt_stage_act_hvx_direct" in src and "hrt_copy_wh_block" in src))
    checks.append(("qt-local schedule emitted", "for (int qt = 0; qt <" in src and "for (int kt = 0; kt < (qt + 1); kt++)" in src))
    checks.append(("full-head KV block staging", "const int cpy_job = job" in src and "const int k_all_job = job" not in src and "const int v_all_stage = job" not in src and "hrt_stage_f16_rm_to_wh_nt" not in src))
    checks.append(("full-row score sweep", "for (int kt = 0; kt < (qt + 1); kt++)" in src and "ScoreFull" not in src and "const int k_stage = job" not in src and "Vpad" not in src))
    checks.append(("v5 row-parallel full softmax", "const int rmask = job" in src and "const int rout = job" in src and "const int p_stage = job" in src and "const int arow = job" not in src))
    softmax_blob = src[src.find("static void attnops_tl_fa_std_pool4_worker"):src.find("static void attnops_tl_fa_std_pool5_worker")]
    checks.append(("anti-pattern: no vectorized f16 integer-predicate select for causal mask", "hrt_splat_h(0xF800)" not in softmax_blob))
    checks.append(("per-qt 32x32 P staging", "const int p_stage = job" in src and "hrt_stage_act_hvx_direct_strided((const uint8_t *)(((f16 *)(V +" in src and "p_stage * 32" in src))
    checks.append(("vtcm 32-lane reduce helpers", "hrt_reduce_max_f16_32_vtcm" in src and "hrt_reduce_sum_f16_32_vtcm" in src))
    checks.append(("no online state", "Mslot" not in src and "Lslot" not in src and "Mfin" not in src and "Lbuf" not in src and "Alpha" not in src and "Oacc" not in src))
    checks.append(("fp32 exp lowering", "expf(" in src or "hrt_exp" in src))
    checks.append(("causal mask present", ("-32768" in src or "0xF800" in src) and "cmask" in src and "rmask" in src))
    checks.append(("prof counters", "tl.hexagon_prof profile slots" in src and "tl_prof_phase_t0" in src and "prof[5.." in src))
    checks.append(("pure standard: no handwritten fa leaf", "attnops_fa256" not in src and "fa2_" not in src))
    m = re.search(r"#define TL_GENERIC_VTCM_BYTES \(\(size_t\)(\d+)\)", src)
    vtcm = int(m.group(1)) if m else -1
    checks.append((f"vtcm budget {vtcm} < 8MB", 0 <= vtcm < 8 * 1024 * 1024))
    for budget_s in (1024, 4096):
        approx = 2 * budget_s * 256 * 2 + 2 * 32 * budget_s * 2 + 32 * 256 * 2 + 32 * 32 * 4 + 32 * 256 * 2
        print(f"FA_STD_V5_VTCM_BUDGET S={budget_s} approx_bytes={approx} mib={approx / (1024*1024):.3f}")
        checks.append((f"approx vtcm S={budget_s} < 8MB", approx < 8 * 1024 * 1024))
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
