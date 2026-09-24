// silu_mul_test.c - host validation for TileLang-emitted SwiGLU silu_mul.
// Shape: M=960, FF=9216.  Slab ABI matches attnops_tl_silu_generic:
//   G[M*FF] fp16 | U[M*FF] fp16 | O[M*FF] fp16, each 128B-aligned.
// Checks generic-expression and VTCM-staging O = silu(G) * U against fp64
// reference with R12 rms-scaled rel error, then compares the two outputs directly.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "sdkl.h"
#include "attnops.h"

#define M 960
#define FF 9216

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static size_t al128(size_t v) { return (v + 127) & ~(size_t)127; }

static uint32_t rng_state = 0x2468ace1u;
static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng_state = x;
}
static _Float16 rand_h(float scale) {
    int v = (int)(xorshift32() % 2001u) - 1000;
    return (_Float16)((float)v * scale / 1000.0f);
}

typedef struct { size_t g_off, u_off, o_off, prof_off, slab_sz, elems; } layout_t;

static layout_t make_layout(void) {
    layout_t l;
    l.elems = (size_t)M * FF;
    size_t off = 0;
    l.g_off = off; off += l.elems * 2;
    l.u_off = al128(off); off = l.u_off + l.elems * 2;
    l.o_off = al128(off); off = l.o_off + l.elems * 2;
    l.prof_off = al128(off);
    l.slab_sz = l.prof_off + 8192;
    return l;
}

static void fill_inputs(unsigned char *slab, const layout_t *l) {
    _Float16 *G = (_Float16 *)(slab + l->g_off);
    _Float16 *U = (_Float16 *)(slab + l->u_off);
    _Float16 *O = (_Float16 *)(slab + l->o_off);
    rng_state = 0x13572468u;
    for (size_t i = 0; i < l->elems; i++) G[i] = rand_h(6.0f);
    for (size_t i = 0; i < l->elems; i++) U[i] = rand_h(1.0f);
    memset(O, 0, l->elems * 2);
    memset(slab + l->prof_off, 0, 8192);
}

static void clear_outputs(unsigned char *slab, const layout_t *l) {
    memset(slab + l->o_off, 0, l->elems * 2);
    memset(slab + l->prof_off, 0, 8192);
}

static void copy_output(_Float16 *dst, const unsigned char *slab, const layout_t *l) {
    memcpy(dst, slab + l->o_off, l->elems * 2);
}

static double ref_silu_mul(double g, double u) {
    return (g / (1.0 + exp(-g))) * u;
}

static int check_ref(const unsigned char *slab, const layout_t *l, double *max_rel_out) {
    const _Float16 *G = (const _Float16 *)(slab + l->g_off);
    const _Float16 *U = (const _Float16 *)(slab + l->u_off);
    const _Float16 *O = (const _Float16 *)(slab + l->o_off);
    double ss = 0.0;
    for (size_t i = 0; i < l->elems; i++) {
        double ref = ref_silu_mul((double)(float)G[i], (double)(float)U[i]);
        ss += ref * ref;
    }
    double rms = sqrt(ss / (double)l->elems);
    double mr = 0.0, bgot = 0.0, bref = 0.0;
    size_t bi = 0;
    int bad = 0;
    for (size_t i = 0; i < l->elems; i++) {
        double ref = ref_silu_mul((double)(float)G[i], (double)(float)U[i]);
        double got = (double)(float)O[i];
        double rel = fabs(got - ref) / (fabs(ref) + 0.02 * rms);
        if (rel > mr) { mr = rel; bi = i; bgot = got; bref = ref; }
        if (rel >= 0.1) bad++;
    }
    *max_rel_out = mr;
    if (bad) {
        printf("  worst i=%zu got %.8f ref %.8f rms %.8f max_rel %.6f bad %d/%zu\n",
               bi, bgot, bref, rms, mr, bad, l->elems);
        return 1;
    }
    return 0;
}

static double compare_outputs(const _Float16 *a, const _Float16 *b, size_t n) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) {
        double av = (double)(float)a[i];
        ss += av * av;
    }
    double rms = sqrt(ss / (double)n);
    double mr = 0.0;
    for (size_t i = 0; i < n; i++) {
        double av = (double)(float)a[i];
        double bv = (double)(float)b[i];
        double rel = fabs(av - bv) / (fabs(av) + 0.02 * rms);
        if (rel > mr) mr = rel;
    }
    return mr;
}

typedef int (*silu_fn_t)(remote_handle64, unsigned char *, int, int, int, int);

typedef struct {
    const char *name;
    double max_rel;
    double ms;
    double gbps;
    int ret;
    int fail;
    int prof_cycles;
} result_t;

static result_t run_kernel(const char *name, silu_fn_t fn, remote_handle64 ah,
                           unsigned char *slab, const layout_t *l, int iters, int abl,
                           _Float16 *out_copy) {
    result_t r;
    memset(&r, 0, sizeof(r));
    r.name = name;
    r.max_rel = 999.0;
    clear_outputs(slab, l);
    int e = fn(ah, slab, (int)l->slab_sz, M, FF, abl);
    if (e) {
        r.ret = e;
        r.fail = 1;
        return r;
    }
    double t0 = now_s();
    for (int it = 0; it < iters; it++) {
        clear_outputs(slab, l);
        e = fn(ah, slab, (int)l->slab_sz, M, FF, abl);
    }
    r.ms = (now_s() - t0) * 1e3 / iters;
    r.ret = e;
    if (e) {
        r.fail = 1;
        return r;
    }
    r.fail = check_ref(slab, l, &r.max_rel);
    double gb = 3.0 * (double)l->elems * 2.0 / 1e9;
    r.gbps = gb / (r.ms * 1e-3);
    int *prof = (int *)(slab + l->prof_off);
    r.prof_cycles = prof[0];
    if (out_copy) copy_output(out_copy, slab, l);
    return r;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 10;
    layout_t l = make_layout();
    unsigned char *slab = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, l.slab_sz);
    if (!slab) { printf("alloc fail slab_sz=%zu\n", l.slab_sz); return 1; }
    fill_inputs(slab, &l);
    remote_register_buf_attr2(slab, l.slab_sz, rpcmem_to_fd(slab),
        FASTRPC_ATTR_COHERENT | FASTRPC_ATTR_KEEP_MAP | FASTRPC_ATTR_TRY_MAP_STATIC);

    struct remote_rpc_control_unsigned_module umod = { .domain = CDSP_DOMAIN_ID, .enable = 1 };
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &umod, sizeof umod);
    char uri[256];
    snprintf(uri, sizeof uri, "%s&_dom=cdsp", attnops_URI);
    remote_handle64 ah = -1;
    if (attnops_open(uri, &ah)) { printf("open fail\n"); return 1; }

    _Float16 *out_gen = (_Float16 *)malloc(l.elems * 2);
    _Float16 *out_vtcm = (_Float16 *)malloc(l.elems * 2);
    if (!out_gen || !out_vtcm) { printf("malloc output copies fail\n"); return 1; }

    int abl = getenv("ABL") ? atoi(getenv("ABL")) : 0;
    result_t gen = run_kernel("generic", attnops_tl_silu_generic, ah, slab, &l, iters, abl, out_gen);
    result_t vtcm = run_kernel("vtcm", attnops_tl_silu_vtcm, ah, slab, &l, iters, abl, out_vtcm);
    double vtcm_pair_rel = compare_outputs(out_gen, out_vtcm, l.elems);
    double vtcm_perf_delta = gen.ms > 0.0 ? (vtcm.ms / gen.ms - 1.0) * 100.0 : 0.0;

    printf("TL_SILU_COMPARE M=%d FF=%d elems=%zu iters=%d\n", M, FF, l.elems, iters);
    printf("  kernel      max_rel     ms      GBps   prof_cycles ret status\n");
    printf("  %-10s %.6f  %.3f  %.2f  %d %d %s\n",
           gen.name, gen.max_rel, gen.ms, gen.gbps, gen.prof_cycles, gen.ret,
           gen.fail ? "FAIL" : "OK");
    printf("  %-10s %.6f  %.3f  %.2f  %d %d %s\n",
           vtcm.name, vtcm.max_rel, vtcm.ms, vtcm.gbps, vtcm.prof_cycles, vtcm.ret,
           vtcm.fail ? "FAIL" : "OK");
    printf("  vtcm_minus_generic max_rel_delta=%.6f ms_delta=%.3f ms perf_delta=%.1f%% out_pair_max_rel=%.6f\n",
           vtcm.max_rel - gen.max_rel, vtcm.ms - gen.ms, vtcm_perf_delta, vtcm_pair_rel);
    attnops_close(ah);
    free(out_gen);
    free(out_vtcm);
    rpcmem_free(slab);
    int fail = gen.fail || vtcm.fail || gen.ret || vtcm.ret;
    printf("TL_SILU_ALL_%s\n", fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}
