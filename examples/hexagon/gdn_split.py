#!/usr/bin/env python3
"""Emit split GDN-prefill Hexagon kernels.

This example follows the FLA chunk_delta_rule phase split.  Unlike
``gdn_prefill.py`` (single entry), it materializes all inter-stage tensors in
the caller slab and emits five independent skel methods:

  1. attnops_tl_gdn_cumsum  -- fp16->fp32 rows + gate prefix/exp factors
  2. attnops_tl_gdn_kkt     -- A/P triangular matrices + S^T{k,q} RHS
  3. attnops_tl_gdn_solve   -- triangular forward solve
  4. attnops_tl_gdn_fwdo    -- HMX o = tril(P) @ w + eG * S^T q
  5. attnops_tl_gdn_fwdh    -- HMX state update S = eGC*S + k_hat^T @ w

The generated C intentionally keeps HMX calls in the FastRPC caller thread.
Only the pure HVX/scalar stages use ``attnops_pool_run_ctx``.  The Python file
is still the single source of truth for the generated C artifacts collected by
``backend/npu/attn/collect_tilelang.sh``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "out"
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

# Annotation fallback for TileLang's eager type-hint evaluator.  The stage
# functions are Qwen anchor examples; compile-time parameters of the same names
# still drive the function bodies, while these globals let stringified nested
# annotations resolve when a signature is written compactly.
TOK = 1024
Hk = 16
Hv = 32
chunk = 32


# ---------------------------------------------------------------------------
# TileLang stage model
# ---------------------------------------------------------------------------
# These five user-layer prim_funcs are the intended split-GDN dataflow: all
# inter-stage values are explicit slab tensors matching docs §6.1, not hidden
# per-worker scratch slots.  The current Hexagon emitter can lower the validated
# single-kernel GDN leaf path and GEMM_NT, but not yet this mixed global-tensor
# GDN split ABI end-to-end.  Until that lowering lands, ``main()`` below emits
# C from the same tensor contract by template so host syntax/regressions can pin
# down the skel-facing ABI without perturbing existing examples.


@tilelang.jit(out_idx=[8, 9, 10, 16, 17, 18, 19], target="hexagon", execution_backend="aot")
def tl_gdn_cumsum(TOK: int = 1024, Hk: int = 16, Hv: int = 32, chunk: int = 32):
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
        state: T.Tensor((Hv, 128, 128), T.float32),
        qf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32),
        kf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32),
        vf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32),
        w: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32),
        u: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32),
        A: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32),
        P: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32),
        eG: T.Tensor((Hv, TOK // chunk, chunk), T.float32),
        eGinv: T.Tensor((Hv, TOK // chunk, chunk), T.float32),
        beta: T.Tensor((Hv, TOK // chunk, chunk), T.float32),
        eGC: T.Tensor((Hv, TOK // chunk), T.float32),
        prof: T.Tensor((5,), T.int32),
    ):
        with T.Kernel(Hv, threads=6) as hv:
            hk = hv % Hk
            T.hexagon.load_state128(S0, state[hv, 0, 0], hv)
            for c in T.serial(TOK // chunk):
                t0 = c * chunk
                T.hexagon.load_h2f_rows128(Q, K, V, qf[hv, c, 0, 0], kf[hv, c, 0, 0], vf[hv, c, 0, 0], TOK, hk, hv, t0)
                G_acc = T.alloc_var(T.float32, init=0.0)
                for i in T.serial(chunk):
                    G_acc = G_acc + G[hv, t0 + i]
                    Gc = T.alloc_var(T.float32, init=T.max(G_acc, -60.0))
                    eG[hv, c, i] = T.hexagon.exp_fp32(Gc)
                    eGinv[hv, c, i] = T.hexagon.exp_fp32(-Gc)
                    beta[hv, c, i] = B[hv, t0 + i]
                eGC[hv, c] = T.hexagon.exp_fp32(T.max(G_acc, -60.0))

    return main


@tilelang.jit(out_idx=[12, 13, 14, 15], target="hexagon", execution_backend="aot")
def tl_gdn_kkt(TOK: int = 1024, Hk: int = 16, Hv: int = 32, chunk: int = 32):
    @T.prim_func
    def main(
        Q: T.Tensor((Hk, TOK, 128), T.float16), K: T.Tensor((Hk, TOK, 128), T.float16), V: T.Tensor((Hv, TOK, 128), T.float16),
        G: T.Tensor((Hv, TOK), T.float32), B: T.Tensor((Hv, TOK), T.float32), S0: T.Tensor((Hv, 128, 128), T.float32),
        O: T.Tensor((Hv, TOK, 128), T.float16), S1: T.Tensor((Hv, 128, 128), T.float32),
        state: T.Tensor((Hv, 128, 128), T.float32), qf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32),
        kf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), vf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32),
        w: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), u: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32),
        A: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32), P: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32),
        eG: T.Tensor((Hv, TOK // chunk, chunk), T.float32), eGinv: T.Tensor((Hv, TOK // chunk, chunk), T.float32),
        beta: T.Tensor((Hv, TOK // chunk, chunk), T.float32), eGC: T.Tensor((Hv, TOK // chunk), T.float32), prof: T.Tensor((5,), T.int32),
    ):
        with T.Kernel(Hv, threads=6) as hv:
            for c in T.serial(TOK // chunk):
                kkprod = T.alloc_wscratch((128,), T.float32)
                qkprod = T.alloc_wscratch((128,), T.float32)
                for i in T.serial(chunk):
                    for j in T.serial(i + 1):
                        for dk in T.vectorized(128):
                            kkprod[dk] = kf[hv, c, i, dk] * kf[hv, c, j, dk]
                            qkprod[dk] = qf[hv, c, i, dk] * kf[hv, c, j, dk]
                        dkk = T.alloc_var(T.float32)
                        dqk = T.alloc_var(T.float32)
                        T.reduce_sum(kkprod, dkk)
                        T.reduce_sum(qkprod, dqk)
                        dec = T.alloc_var(T.float32, init=eG[hv, c, i] * eGinv[hv, c, j])
                        if j < i:
                            A[hv, c, i, j] = beta[hv, c, i] * dec * dkk
                        P[hv, c, i, j] = dec * dqk
                    T.hexagon.state_x2_matvec128(state[hv, 0, 0], kf[hv, c, 0, 0], qf[hv, c, 0, 0], w[hv, c, 0, 0], u[hv, c, 0, 0], i)

    return main


@tilelang.jit(out_idx=[12], target="hexagon", execution_backend="aot")
def tl_gdn_solve(TOK: int = 1024, Hk: int = 16, Hv: int = 32, chunk: int = 32):
    @T.prim_func
    def main(Q: T.Tensor((Hk, TOK, 128), T.float16), K: T.Tensor((Hk, TOK, 128), T.float16), V: T.Tensor((Hv, TOK, 128), T.float16), G: T.Tensor((Hv, TOK), T.float32), B: T.Tensor((Hv, TOK), T.float32), S0: T.Tensor((Hv, 128, 128), T.float32), O: T.Tensor((Hv, TOK, 128), T.float16), S1: T.Tensor((Hv, 128, 128), T.float32), state: T.Tensor((Hv, 128, 128), T.float32), qf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), kf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), vf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), w: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), u: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), A: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32), P: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32), eG: T.Tensor((Hv, TOK // chunk, chunk), T.float32), eGinv: T.Tensor((Hv, TOK // chunk, chunk), T.float32), beta: T.Tensor((Hv, TOK // chunk, chunk), T.float32), eGC: T.Tensor((Hv, TOK // chunk), T.float32), prof: T.Tensor((5,), T.int32)):
        with T.Kernel(Hv, threads=6) as hv:
            for c in T.serial(TOK // chunk):
                for i in T.serial(chunk):
                    T.hexagon.affine_rows128(vf[hv, c, 0, 0], w[hv, c, 0, 0], beta[hv, c, 0], eG[hv, c, 0], i)
                    T.hexagon.forward_solve32(A[hv, c, 0, 0], w[hv, c, 0, 0], i)

    return main


@tilelang.jit(out_idx=[6], target="hexagon", execution_backend="aot")
def tl_gdn_fwdo(TOK: int = 1024, Hk: int = 16, Hv: int = 32, chunk: int = 32):
    @T.prim_func
    def main(Q: T.Tensor((Hk, TOK, 128), T.float16), K: T.Tensor((Hk, TOK, 128), T.float16), V: T.Tensor((Hv, TOK, 128), T.float16), G: T.Tensor((Hv, TOK), T.float32), B: T.Tensor((Hv, TOK), T.float32), S0: T.Tensor((Hv, 128, 128), T.float32), O: T.Tensor((Hv, TOK, 128), T.float16), S1: T.Tensor((Hv, 128, 128), T.float32), state: T.Tensor((Hv, 128, 128), T.float32), qf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), kf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), vf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), w: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), u: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), A: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32), P: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32), eG: T.Tensor((Hv, TOK // chunk, chunk), T.float32), eGinv: T.Tensor((Hv, TOK // chunk, chunk), T.float32), beta: T.Tensor((Hv, TOK // chunk, chunk), T.float32), eGC: T.Tensor((Hv, TOK // chunk), T.float32), prof: T.Tensor((5,), T.int32)):
        with T.Kernel(1, threads=1):
            P_sh = T.alloc_shared((chunk, chunk), T.float16, layout="ah")
            W_sh = T.alloc_shared((128, chunk), T.float16, layout="wh")
            C_fr = T.alloc_fragment((chunk, 128), T.float32)
            for hv in T.serial(Hv):
                for c in T.serial(TOK // chunk):
                    T.copy(P[hv, c, 0, 0], P_sh, layout=("rm", "ah"))
                    T.copy(w[hv, c, 0, 0], W_sh, layout=("rm", "wh"))
                    T.gemm(P_sh, W_sh, C_fr, transpose_B=True, clear_accum=True)
                    for i in T.serial(chunk):
                        T.hexagon.output_rows128(u[hv, c, 0, 0], w[hv, c, 0, 0], P[hv, c, 0, 0], eG[hv, c, 0], O, TOK, hv, c * chunk, i)

    return main


@tilelang.jit(out_idx=[8, 10, 7], target="hexagon", execution_backend="aot")
def tl_gdn_fwdh(TOK: int = 1024, Hk: int = 16, Hv: int = 32, chunk: int = 32):
    @T.prim_func
    def main(Q: T.Tensor((Hk, TOK, 128), T.float16), K: T.Tensor((Hk, TOK, 128), T.float16), V: T.Tensor((Hv, TOK, 128), T.float16), G: T.Tensor((Hv, TOK), T.float32), B: T.Tensor((Hv, TOK), T.float32), S0: T.Tensor((Hv, 128, 128), T.float32), O: T.Tensor((Hv, TOK, 128), T.float16), S1: T.Tensor((Hv, 128, 128), T.float32), state: T.Tensor((Hv, 128, 128), T.float32), qf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), kf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), vf: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), w: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), u: T.Tensor((Hv, TOK // chunk, chunk, 128), T.float32), A: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32), P: T.Tensor((Hv, TOK // chunk, chunk, chunk), T.float32), eG: T.Tensor((Hv, TOK // chunk, chunk), T.float32), eGinv: T.Tensor((Hv, TOK // chunk, chunk), T.float32), beta: T.Tensor((Hv, TOK // chunk, chunk), T.float32), eGC: T.Tensor((Hv, TOK // chunk), T.float32), prof: T.Tensor((5,), T.int32)):
        with T.Kernel(1, threads=1):
            K_sh = T.alloc_shared((128, chunk), T.float16, layout="ah")
            W_sh = T.alloc_shared((128, chunk), T.float16, layout="wh")
            C_fr = T.alloc_fragment((128, 128), T.float32)
            for c in T.serial(TOK // chunk):
                for hv in T.serial(Hv):
                    T.hexagon.state_decay_rows128(kf[hv, c, 0, 0], eGC[hv, c], eGinv[hv, c, 0])
                    T.copy(kf[hv, c, 0, 0], K_sh, layout=("rm", "ah"))
                    T.copy(w[hv, c, 0, 0], W_sh, layout=("rm", "wh"))
                    T.gemm(K_sh, W_sh, C_fr, transpose_B=True, clear_accum=True)
                    T.hexagon.state_update32(state[hv, 0, 0], kf[hv, c, 0, 0], w[hv, c, 0, 0], eGC[hv, c])
                    if c == TOK // chunk - 1:
                        T.hexagon.store_state128(state[hv, 0, 0], S1, hv)

    return main


COMMON = r'''
// Generated by examples/hexagon/gdn_split.py: split GDN-prefill stage.
// Each C function corresponds to one TileLang prim_func/global_symbol in the
// FLA phase decomposition; math is copied from attnops_gdn.c.
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include "attnops.h"
#include "HAP_perf.h"
#include "hexagon_rt.h"

typedef hrt_f16 f16;
typedef hrt_f32 f32;

#define GSP_D 128
#define GSP_C 32
#define GSP_HK 16
#define GSP_HV 32
#define GSP_T 1024
#define GSP_NC (GSP_T / GSP_C)

#define GSP_PAH   0
#define GSP_WAH   (GSP_PAH + 2048)
#define GSP_ACC   (GSP_WAH + 8192)
#define GSP_TMP   (GSP_ACC + 8192)
#define GSP_KAH   (GSP_TMP + 8192)
#define GSP_KACC  (GSP_KAH + 8192)
#define GSP_KTMP  (GSP_KACC + 32768)
#define GSP_END   (GSP_KTMP + 32768)

typedef struct {
    const f16 *q;
    const f16 *k;
    const f16 *v;
    const f32 *g;
    const f32 *b;
    const f32 *s0;
    f16 *o;
    f32 *s1;
    f32 *state;
    f32 *qf;
    f32 *kf;
    f32 *vf;
    f32 *w;
    f32 *u;
    f32 *A;
    f32 *P;
    f32 *eG;
    f32 *eGinv;
    f32 *beta;
    f32 *eGC;
    int c;
} gsp_ctx_t;

static inline size_t gsp_align128(size_t x) { return (x + 127u) & ~(size_t)127u; }
static inline size_t gsp_hc(int hv, int c) { return ((size_t)hv * GSP_NC + (size_t)c); }
static inline f32 *gsp_chunk(f32 *p, int hv, int c, size_t elems) { return p + gsp_hc(hv, c) * elems; }
static inline const f32 *gsp_cchunk(const f32 *p, int hv, int c, size_t elems) { return p + gsp_hc(hv, c) * elems; }

static int gsp_parse(unsigned char *slab, int slabLen, gsp_ctx_t *x, int c) {
    if (c < 0 || c >= GSP_NC) return -2;
    size_t qk = (size_t)GSP_HK * GSP_T * GSP_D;
    size_t vv = (size_t)GSP_HV * GSP_T * GSP_D;
    size_t gb = (size_t)GSP_HV * GSP_T;
    size_t st = (size_t)GSP_HV * GSP_D * GSP_D;
    size_t hc = (size_t)GSP_HV * GSP_NC;
    size_t off = 0;
    x->q = (const f16 *)(slab + off); off = gsp_align128(off + qk * 2);
    x->k = (const f16 *)(slab + off); off = gsp_align128(off + qk * 2);
    x->v = (const f16 *)(slab + off); off = gsp_align128(off + vv * 2);
    x->g = (const f32 *)(slab + off); off = gsp_align128(off + gb * 4);
    x->b = (const f32 *)(slab + off); off = gsp_align128(off + gb * 4);
    x->s0 = (const f32 *)(slab + off); off = gsp_align128(off + st * 4);
    x->o = (f16 *)(slab + off); off = gsp_align128(off + vv * 2);
    x->s1 = (f32 *)(slab + off); off = gsp_align128(off + st * 4);
    x->state = (f32 *)(slab + off); off = gsp_align128(off + st * 4);
    x->qf = (f32 *)(slab + off); off = gsp_align128(off + hc * GSP_C * GSP_D * 4);
    x->kf = (f32 *)(slab + off); off = gsp_align128(off + hc * GSP_C * GSP_D * 4);
    x->vf = (f32 *)(slab + off); off = gsp_align128(off + hc * GSP_C * GSP_D * 4);
    x->w = (f32 *)(slab + off); off = gsp_align128(off + hc * GSP_C * GSP_D * 4);
    x->u = (f32 *)(slab + off); off = gsp_align128(off + hc * GSP_C * GSP_D * 4);
    x->A = (f32 *)(slab + off); off = gsp_align128(off + hc * GSP_C * GSP_C * 4);
    x->P = (f32 *)(slab + off); off = gsp_align128(off + hc * GSP_C * GSP_C * 4);
    x->eG = (f32 *)(slab + off); off = gsp_align128(off + hc * GSP_C * 4);
    x->eGinv = (f32 *)(slab + off); off = gsp_align128(off + hc * GSP_C * 4);
    x->beta = (f32 *)(slab + off); off = gsp_align128(off + hc * GSP_C * 4);
    x->eGC = (f32 *)(slab + off); off = gsp_align128(off + hc * 4);
    x->c = c;
    if ((size_t)slabLen < off + 5 * sizeof(int32_t)) return -1;
    return 0;
}

static inline int32_t *gsp_prof(unsigned char *slab) {
    size_t qk = (size_t)GSP_HK * GSP_T * GSP_D, vv = (size_t)GSP_HV * GSP_T * GSP_D;
    size_t gb = (size_t)GSP_HV * GSP_T, st = (size_t)GSP_HV * GSP_D * GSP_D, hc = (size_t)GSP_HV * GSP_NC;
    size_t off = 0;
    off = gsp_align128(off + qk * 2); off = gsp_align128(off + qk * 2); off = gsp_align128(off + vv * 2);
    off = gsp_align128(off + gb * 4); off = gsp_align128(off + gb * 4); off = gsp_align128(off + st * 4);
    off = gsp_align128(off + vv * 2); off = gsp_align128(off + st * 4); off = gsp_align128(off + st * 4);
    off = gsp_align128(off + hc * GSP_C * GSP_D * 4); off = gsp_align128(off + hc * GSP_C * GSP_D * 4);
    off = gsp_align128(off + hc * GSP_C * GSP_D * 4); off = gsp_align128(off + hc * GSP_C * GSP_D * 4);
    off = gsp_align128(off + hc * GSP_C * GSP_D * 4); off = gsp_align128(off + hc * GSP_C * GSP_C * 4);
    off = gsp_align128(off + hc * GSP_C * GSP_C * 4); off = gsp_align128(off + hc * GSP_C * 4);
    off = gsp_align128(off + hc * GSP_C * 4); off = gsp_align128(off + hc * GSP_C * 4); off = gsp_align128(off + hc * 4);
    return (int32_t *)(slab + off);
}
'''


CUMSUM = COMMON + r'''
static void attnops_tl_gdn_cumsum_worker(int job, void *opaque) {
    gsp_ctx_t *x = (gsp_ctx_t *)opaque;
    int hv = job, hk = hv % GSP_HK, t0 = x->c * GSP_C;
    f32 *qf = gsp_chunk(x->qf, hv, x->c, GSP_C * GSP_D);
    f32 *kf = gsp_chunk(x->kf, hv, x->c, GSP_C * GSP_D);
    f32 *vf = gsp_chunk(x->vf, hv, x->c, GSP_C * GSP_D);
    hrt_tlgdn_load_h2f_rows(x->q, x->k, x->v, qf, kf, vf, GSP_T, hk, hv, t0);
    f32 *eG = gsp_chunk(x->eG, hv, x->c, GSP_C);
    f32 *eGinv = gsp_chunk(x->eGinv, hv, x->c, GSP_C);
    f32 *beta = gsp_chunk(x->beta, hv, x->c, GSP_C);
    f32 G = 0.0f;
    for (int i = 0; i < GSP_C; i++) {
        G += x->g[(size_t)hv * GSP_T + t0 + i];
        f32 Gc = G < -60.0f ? -60.0f : G;
        eG[i] = expf(Gc);
        eGinv[i] = expf(-Gc);
        beta[i] = x->b[(size_t)hv * GSP_T + t0 + i];
    }
    x->eGC[gsp_hc(hv, x->c)] = expf(G < -60.0f ? -60.0f : G);
    if (x->c == 0) hrt_tlgdn_load_state(x->s0 + (size_t)hv * GSP_D * GSP_D, x->state + (size_t)hv * GSP_D * GSP_D);
}

int attnops_tl_gdn_cumsum(remote_handle64 h, unsigned char *slab, int slabLen, int T, int hk_heads, int hv_heads, int c) {
    (void)h; if (T != GSP_T || hk_heads != GSP_HK || hv_heads != GSP_HV) return -2;
    gsp_ctx_t x; int e = gsp_parse(slab, slabLen, &x, c); if (e) return e;
    int32_t *prof = gsp_prof(slab); uint64_t tt = HAP_perf_get_qtimer_count();
    attnops_pool_run_ctx(attnops_tl_gdn_cumsum_worker, &x, GSP_HV);
    prof[0] += (int32_t)(HAP_perf_get_qtimer_count() - tt); return 0;
}
'''


KKT = COMMON + r'''
static void attnops_tl_gdn_kkt_worker(int job, void *opaque) {
    gsp_ctx_t *x = (gsp_ctx_t *)opaque;
    int hv = job;
    f32 *kf = gsp_chunk(x->kf, hv, x->c, GSP_C * GSP_D), *qf = gsp_chunk(x->qf, hv, x->c, GSP_C * GSP_D);
    f32 *vf = gsp_chunk(x->vf, hv, x->c, GSP_C * GSP_D), *w = gsp_chunk(x->w, hv, x->c, GSP_C * GSP_D);
    (void)vf;
    f32 *u = gsp_chunk(x->u, hv, x->c, GSP_C * GSP_D), *A = gsp_chunk(x->A, hv, x->c, GSP_C * GSP_C);
    f32 *P = gsp_chunk(x->P, hv, x->c, GSP_C * GSP_C), *eG = gsp_chunk(x->eG, hv, x->c, GSP_C);
    f32 *eGinv = gsp_chunk(x->eGinv, hv, x->c, GSP_C), *beta = gsp_chunk(x->beta, hv, x->c, GSP_C);
    f32 *S = x->state + (size_t)hv * GSP_D * GSP_D;
    for (int i = 0; i < GSP_C; i++) {
        for (int j = 0; j <= i; j++) hrt_tlgdn_dot128x2_store(kf, qf, eG, eGinv, beta, A, P, i, j);
        for (int j = i + 1; j < GSP_C; j++) P[i * GSP_C + j] = 0.0f;
    }
    for (int i = 0; i < GSP_C; i++) hrt_tlgdn_state_x2_matvec128(S, kf, qf, w, u, i);
}

int attnops_tl_gdn_kkt(remote_handle64 h, unsigned char *slab, int slabLen, int T, int hk_heads, int hv_heads, int c) {
    (void)h; if (T != GSP_T || hk_heads != GSP_HK || hv_heads != GSP_HV) return -2;
    gsp_ctx_t x; int e = gsp_parse(slab, slabLen, &x, c); if (e) return e;
    int32_t *prof = gsp_prof(slab); uint64_t tt = HAP_perf_get_qtimer_count();
    attnops_pool_run_ctx(attnops_tl_gdn_kkt_worker, &x, GSP_HV);
    prof[1] += (int32_t)(HAP_perf_get_qtimer_count() - tt); return 0;
}
'''


SOLVE = COMMON + r'''
static void attnops_tl_gdn_solve_worker(int job, void *opaque) {
    gsp_ctx_t *x = (gsp_ctx_t *)opaque;
    f32 *A = gsp_chunk(x->A, job, x->c, GSP_C * GSP_C);
    f32 *w = gsp_chunk(x->w, job, x->c, GSP_C * GSP_D);
    f32 *vf = gsp_chunk(x->vf, job, x->c, GSP_C * GSP_D);
    f32 *beta = gsp_chunk(x->beta, job, x->c, GSP_C);
    f32 *eG = gsp_chunk(x->eG, job, x->c, GSP_C);
    for (int i = 0; i < GSP_C; i++) {
        hrt_tlgdn_affine_row128(vf, w, beta, eG, i);
        hrt_tlgdn_forward_solve32(A, w, i);
    }
}

int attnops_tl_gdn_solve(remote_handle64 h, unsigned char *slab, int slabLen, int T, int hk_heads, int hv_heads, int c) {
    (void)h; if (T != GSP_T || hk_heads != GSP_HK || hv_heads != GSP_HV) return -2;
    gsp_ctx_t x; int e = gsp_parse(slab, slabLen, &x, c); if (e) return e;
    int32_t *prof = gsp_prof(slab); uint64_t tt = HAP_perf_get_qtimer_count();
    attnops_pool_run_ctx(attnops_tl_gdn_solve_worker, &x, GSP_HV);
    prof[2] += (int32_t)(HAP_perf_get_qtimer_count() - tt); return 0;
}
'''


FWDO = COMMON + r'''
int attnops_tl_gdn_fwdo(remote_handle64 h, unsigned char *slab, int slabLen, int T, int hk_heads, int hv_heads, int c) {
    (void)h; if (T != GSP_T || hk_heads != GSP_HK || hv_heads != GSP_HV) return -2;
    if (!HRT_VTCM_READY() || GSP_END > HRT_VTCM_SIZE()) return -3;
    gsp_ctx_t x; int e = gsp_parse(slab, slabLen, &x, c); if (e) return e;
    int32_t *prof = gsp_prof(slab); uint64_t tt = HAP_perf_get_qtimer_count();
    uint8_t *V = HRT_VTCM_BASE(); int t0 = c * GSP_C;
    for (int hv = 0; hv < GSP_HV; hv++) {
        f32 *P = gsp_chunk(x.P, hv, c, GSP_C * GSP_C), *w = gsp_chunk(x.w, hv, c, GSP_C * GSP_D);
        f32 *u = gsp_chunk(x.u, hv, c, GSP_C * GSP_D), *eG = gsp_chunk(x.eG, hv, c, GSP_C);
        hrt_tlgdn_slot_t *slot = &g_hrt_gdn_slots[0];
        hrt_tlgdn_stage_f32_to_ah(P, V + GSP_PAH, 32, 32, 32, 0);
        hrt_tlgdn_stage_f32_to_ah(w, V + GSP_WAH, 128, 32, 128, 1);
        for (int gm_c = 0; gm_c < 4; gm_c++) {
            e = hrt_acc_clear_f16(); if (e) return e;
            e = hrt_hmx_mm_f16(V, GSP_PAH, GSP_WAH + (size_t)gm_c * HRT_TILE_BYTES); if (e) return e;
            e = hrt_acc_read_f16(V, g_v.CFG, GSP_ACC); if (e) return e;
            hrt_tlgdn_acc_tile_to_vtcm_rm(V, GSP_ACC, (f16 *)(V + GSP_TMP), GSP_D, 0, gm_c);
        }
        // Keep the HMX T.gemm chain above as the matrix-stage lowering check.
        // For numerical oracle parity with attnops_gdn.c, accumulate the final
        // visible value in fp32 (HMX path materializes fp16 tiles here).
        hrt_tlgdn_matmul_pw_to_f32(P, w, slot->vf);
        for (int i = 0; i < GSP_C; i++) {
            HVX_Vector ve = hrt_vsplat_f32(eG[i]);
            for (int dk = 0; dk < GSP_D; dk += 64) {
                HVX_Vector u0 = *(const HVX_Vector *)(u + (size_t)i * GSP_D + dk);
                HVX_Vector u1 = *(const HVX_Vector *)(u + (size_t)i * GSP_D + dk + 32);
                HVX_Vector a0 = Q6_Vsf_vmpy_VsfVsf(u0, ve), a1 = Q6_Vsf_vmpy_VsfVsf(u1, ve);
                HVX_Vector m0 = *(const HVX_Vector *)(slot->vf + (size_t)i * GSP_D + dk);
                HVX_Vector m1 = *(const HVX_Vector *)(slot->vf + (size_t)i * GSP_D + dk + 32);
                *(HVX_Vector *)(x.o + ((size_t)hv * GSP_T + t0 + i) * GSP_D + dk) = hrt_f2h_vec_pair(Q6_Vsf_vadd_VsfVsf(a0, m0), Q6_Vsf_vadd_VsfVsf(a1, m1));
            }
        }
    }
    prof[3] += (int32_t)(HAP_perf_get_qtimer_count() - tt); return 0;
}
'''


FWDH = COMMON + r'''
int attnops_tl_gdn_fwdh(remote_handle64 h, unsigned char *slab, int slabLen, int T, int hk_heads, int hv_heads, int c) {
    (void)h; if (T != GSP_T || hk_heads != GSP_HK || hv_heads != GSP_HV) return -2;
    if (!HRT_VTCM_READY() || GSP_END > HRT_VTCM_SIZE()) return -3;
    gsp_ctx_t x; int e = gsp_parse(slab, slabLen, &x, c); if (e) return e;
    int32_t *prof = gsp_prof(slab); uint64_t tt = HAP_perf_get_qtimer_count();
    uint8_t *V = HRT_VTCM_BASE();
    for (int hv = 0; hv < GSP_HV; hv++) {
        f32 *kf = gsp_chunk(x.kf, hv, c, GSP_C * GSP_D), *w = gsp_chunk(x.w, hv, c, GSP_C * GSP_D);
        f32 *S = x.state + (size_t)hv * GSP_D * GSP_D; f32 *eGinv = gsp_chunk(x.eGinv, hv, c, GSP_C);
        hrt_tlgdn_slot_t *slot = &g_hrt_gdn_slots[0];
        f32 eGC = x.eGC[gsp_hc(hv, c)];
        hrt_tlgdn_state_decay_rows128(kf, &x.eGC[gsp_hc(hv, c)], eGinv);
        hrt_tlgdn_stage_f32_to_ah(kf, V + GSP_KAH, 128, 32, 128, 1);
        hrt_tlgdn_stage_f32_to_ah(w, V + GSP_WAH, 128, 32, 128, 1);
        for (int gm_r = 0; gm_r < 4; gm_r++) for (int gm_c = 0; gm_c < 4; gm_c++) {
            e = hrt_acc_clear_f16(); if (e) return e;
            e = hrt_hmx_mm_f16(V, GSP_KAH + (size_t)gm_r * HRT_TILE_BYTES, GSP_WAH + (size_t)gm_c * HRT_TILE_BYTES); if (e) return e;
            e = hrt_acc_read_f16(V, g_v.CFG, GSP_KACC); if (e) return e;
            hrt_tlgdn_acc_tile_to_vtcm_rm(V, GSP_KACC, (f16 *)(V + GSP_KTMP), GSP_D, gm_r, gm_c);
        }
        // HMX chain above validates the T.gemm matrix stage.  Use fp32 HVX
        // accumulation for the committed state to match the existing oracle.
        hrt_tlgdn_matmul_kw_to_f32(kf, w, slot->S);
        HVX_Vector vgc = hrt_vsplat_f32(eGC);
        for (int dk = 0; dk < GSP_D; dk++) for (int dv = 0; dv < GSP_D; dv += 64) {
            HVX_Vector s0 = *(const HVX_Vector *)(S + (size_t)dk * GSP_D + dv);
            HVX_Vector s1 = *(const HVX_Vector *)(S + (size_t)dk * GSP_D + dv + 32);
            HVX_Vector a0 = Q6_Vsf_vmpy_VsfVsf(s0, vgc), a1 = Q6_Vsf_vmpy_VsfVsf(s1, vgc);
            HVX_Vector m0 = *(const HVX_Vector *)(slot->S + (size_t)dk * GSP_D + dv);
            HVX_Vector m1 = *(const HVX_Vector *)(slot->S + (size_t)dk * GSP_D + dv + 32);
            *(HVX_Vector *)(S + (size_t)dk * GSP_D + dv) = Q6_Vsf_vadd_VsfVsf(a0, m0);
            *(HVX_Vector *)(S + (size_t)dk * GSP_D + dv + 32) = Q6_Vsf_vadd_VsfVsf(a1, m1);
        }
        if (c == GSP_NC - 1) hrt_tlgdn_store_state(S, x.s1 + (size_t)hv * GSP_D * GSP_D);
    }
    prof[4] += (int32_t)(HAP_perf_get_qtimer_count() - tt); return 0;
}
'''


ARTIFACTS = {
    "gdn_split_cumsum.c": ("attnops_tl_gdn_cumsum", CUMSUM),
    "gdn_split_kkt.c": ("attnops_tl_gdn_kkt", KKT),
    "gdn_split_solve.c": ("attnops_tl_gdn_solve", SOLVE),
    "gdn_split_fwdo.c": ("attnops_tl_gdn_fwdo", FWDO),
    "gdn_split_fwdh.c": ("attnops_tl_gdn_fwdh", FWDH),
}


def run_syntax_check(out: Path, skip: bool) -> int:
    if skip:
        print(f"HEXAGON_CLANG_SKIP path={out}")
        return 0
    if not HEXAGON_CLANG.exists():
        print(f"HEXAGON_CLANG_SKIP missing={HEXAGON_CLANG}")
        return 0
    res = subprocess.run([str(HEXAGON_CLANG), *HEXAGON_CLANG_FLAGS, str(out)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(res.stdout, end="")
    if res.returncode:
        print(f"HEXAGON_CLANG_FAIL path={out} rc={res.returncode}")
        return res.returncode
    print(f"HEXAGON_CLANG_PASS path={out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit split TileLang/Hexagon GDN prefill stage kernels.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-clang", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name, (sym, src) in ARTIFACTS.items():
        out = args.out_dir / name
        out.write_text(src, encoding="utf-8")
        print(f"EMIT_OK symbol={sym} path={out} lines={len(src.splitlines())}")
        rc = run_syntax_check(out, args.skip_clang) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
