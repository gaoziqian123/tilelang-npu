// gdn_prefill_test.c - host validation for TileLang-emitted GDN prefill.
// Anchor: T=1024,Hk=16,Hv=32,D=128,chunk=32.  Compares O(fp16) and S1(fp32)
// against a fp64 per-token recurrence using R12 rms-scaled relative error.
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

#define T 1024
#define HK 16
#define HV 32
#define D 128

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static size_t al128(size_t v) { return (v + 127) & ~(size_t)127; }

static uint32_t rng_state = 0x7654321u;
static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return rng_state = x;
}
static _Float16 rand_h(float scale) {
    int v = (int)(xorshift32() % 2001u) - 1000;
    return (_Float16)((float)v * scale / 1000.0f);
}
static float rand_f(float scale) {
    int v = (int)(xorshift32() % 2001u) - 1000;
    return (float)v * scale / 1000.0f;
}

typedef struct {
    size_t q_off, k_off, v_off, g_off, b_off, s0_off, o_off, s1_off, prof_off, slab_sz;
} layout_t;

static layout_t make_layout(void) {
    size_t qk = (size_t)HK * T * D * 2;
    size_t vv = (size_t)HV * T * D * 2;
    size_t gb = (size_t)HV * T * 4;
    size_t st = (size_t)HV * D * D * 4;
    layout_t l;
    size_t off = 0;
    l.q_off = off; off += qk;
    l.k_off = off; off += qk;
    l.v_off = off; off += vv;
    l.g_off = off; off += gb;
    l.b_off = off; off += gb;
    l.s0_off = off; off += st;
    l.o_off = off; off += vv;
    l.s1_off = off; off += st;
    l.prof_off = al128(off);
    l.slab_sz = l.prof_off + 8192;
    return l;
}

static void fill_inputs(unsigned char *slab, const layout_t *l) {
    _Float16 *Q = (_Float16 *)(slab + l->q_off);
    _Float16 *K = (_Float16 *)(slab + l->k_off);
    _Float16 *V = (_Float16 *)(slab + l->v_off);
    float *G = (float *)(slab + l->g_off);
    float *B = (float *)(slab + l->b_off);
    float *S0 = (float *)(slab + l->s0_off);
    _Float16 *O = (_Float16 *)(slab + l->o_off);
    float *S1 = (float *)(slab + l->s1_off);
    rng_state = 0x13579bdfu;
    for (size_t i = 0; i < (size_t)HK * T * D; i++) Q[i] = rand_h(0.18f);
    for (size_t i = 0; i < (size_t)HK * T * D; i++) K[i] = rand_h(0.18f);
    for (size_t i = 0; i < (size_t)HV * T * D; i++) V[i] = rand_h(0.20f);
    for (size_t i = 0; i < (size_t)HV * T; i++) {
        G[i] = -0.001f - fabsf(rand_f(0.020f));       // small negative log-decay
        B[i] = 0.02f + fabsf(rand_f(0.20f));          // post-sigmoid beta-like
    }
    for (size_t i = 0; i < (size_t)HV * D * D; i++) S0[i] = rand_f(0.010f);
    memset(O, 0, (size_t)HV * T * D * 2);
    memset(S1, 0, (size_t)HV * D * D * 4);
    memset(slab + l->prof_off, 0, 8192);
}

static void clear_outputs(unsigned char *slab, const layout_t *l) {
    memset(slab + l->o_off, 0, (size_t)HV * T * D * 2);
    memset(slab + l->s1_off, 0, (size_t)HV * D * D * 4);
    memset(slab + l->prof_off, 0, 8192);
}

static int check_fp64_ref(const unsigned char *slab, const layout_t *l, double *max_o, double *max_s) {
    const _Float16 *Q = (const _Float16 *)(slab + l->q_off);
    const _Float16 *K = (const _Float16 *)(slab + l->k_off);
    const _Float16 *V = (const _Float16 *)(slab + l->v_off);
    const float *G = (const float *)(slab + l->g_off);
    const float *B = (const float *)(slab + l->b_off);
    const float *S0 = (const float *)(slab + l->s0_off);
    const _Float16 *O = (const _Float16 *)(slab + l->o_off);
    const float *S1 = (const float *)(slab + l->s1_off);
    double *S = (double *)malloc((size_t)D * D * sizeof(double));
    double *U = (double *)malloc((size_t)D * sizeof(double));
    if (!S || !U) { printf("ref alloc fail\n"); return 1; }
    double ss_o = 0.0, ss_s = 0.0, mr_o = 0.0, mr_s = 0.0;
    size_t no = 0, ns = 0;
    for (int hv = 0; hv < HV; hv++) {
        int hk = hv % HK;
        for (int dk = 0; dk < D; dk++)
            for (int dv = 0; dv < D; dv++)
                S[(size_t)dk * D + dv] = S0[((size_t)hv * D + dk) * D + dv];
        for (int t = 0; t < T; t++) {
            const _Float16 *q = Q + ((size_t)hk * T + t) * D;
            const _Float16 *k = K + ((size_t)hk * T + t) * D;
            const _Float16 *v = V + ((size_t)hv * T + t) * D;
            double a = exp((double)G[(size_t)hv * T + t]);
            double beta = (double)B[(size_t)hv * T + t];
            for (int dv = 0; dv < D; dv++) {
                double sk = 0.0;
                for (int dk = 0; dk < D; dk++) sk += S[(size_t)dk * D + dv] * (double)(float)k[dk];
                U[dv] = beta * ((double)(float)v[dv] - a * sk);
            }
            for (int dk = 0; dk < D; dk++) {
                double kk = (double)(float)k[dk];
                for (int dv = 0; dv < D; dv++) S[(size_t)dk * D + dv] = a * S[(size_t)dk * D + dv] + kk * U[dv];
            }
            for (int dv = 0; dv < D; dv++) {
                double ref = 0.0;
                for (int dk = 0; dk < D; dk++) ref += (double)(float)q[dk] * S[(size_t)dk * D + dv];
                double got = (double)(float)O[((size_t)hv * T + t) * D + dv];
                ss_o += ref * ref; no++;
                double rel = fabs(got - ref); // denominator after rms known; stash max later by second pass avoided via conservative below
                double denom = fabs(ref) + 1e-30;
                (void)denom;
            }
        }
        for (int i = 0; i < D * D; i++) { ss_s += S[i] * S[i]; ns++; }
    }
    double rms_o = sqrt(ss_o / (double)no), rms_s = sqrt(ss_s / (double)ns);
    int bad_o = 0, bad_s = 0;
    int bo_h=-1, bo_t=-1, bo_d=-1, bs_h=-1, bs_i=-1;
    double bo_g=0, bo_r=0, bs_g=0, bs_r=0;
    for (int hv = 0; hv < HV; hv++) {
        int hk = hv % HK;
        for (int dk = 0; dk < D; dk++)
            for (int dv = 0; dv < D; dv++)
                S[(size_t)dk * D + dv] = S0[((size_t)hv * D + dk) * D + dv];
        for (int t = 0; t < T; t++) {
            const _Float16 *q = Q + ((size_t)hk * T + t) * D;
            const _Float16 *k = K + ((size_t)hk * T + t) * D;
            const _Float16 *v = V + ((size_t)hv * T + t) * D;
            double a = exp((double)G[(size_t)hv * T + t]);
            double beta = (double)B[(size_t)hv * T + t];
            for (int dv = 0; dv < D; dv++) {
                double sk = 0.0;
                for (int dk = 0; dk < D; dk++) sk += S[(size_t)dk * D + dv] * (double)(float)k[dk];
                U[dv] = beta * ((double)(float)v[dv] - a * sk);
            }
            for (int dk = 0; dk < D; dk++) {
                double kk = (double)(float)k[dk];
                for (int dv = 0; dv < D; dv++) S[(size_t)dk * D + dv] = a * S[(size_t)dk * D + dv] + kk * U[dv];
            }
            for (int dv = 0; dv < D; dv++) {
                double ref = 0.0;
                for (int dk = 0; dk < D; dk++) ref += (double)(float)q[dk] * S[(size_t)dk * D + dv];
                double got = (double)(float)O[((size_t)hv * T + t) * D + dv];
                double rel = fabs(got - ref) / (fabs(ref) + 0.02 * rms_o);
                if (rel > mr_o) { mr_o = rel; bo_h=hv; bo_t=t; bo_d=dv; bo_g=got; bo_r=ref; }
                if (rel >= 0.1) bad_o++;
            }
        }
        for (int i = 0; i < D * D; i++) {
            double ref = S[i];
            double got = (double)S1[(size_t)hv * D * D + i];
            double rel = fabs(got - ref) / (fabs(ref) + 0.02 * rms_s);
            if (rel > mr_s) { mr_s = rel; bs_h=hv; bs_i=i; bs_g=got; bs_r=ref; }
            if (rel >= 0.1) bad_s++;
        }
    }
    *max_o = mr_o; *max_s = mr_s;
    if (bad_o || bad_s) {
        printf("  worst O h=%d t=%d d=%d got %.8f ref %.8f rms %.8f max_rel %.6f bad %d/%zu\n",
               bo_h, bo_t, bo_d, bo_g, bo_r, rms_o, mr_o, bad_o, no);
        printf("  worst S1 h=%d i=%d got %.8f ref %.8f rms %.8f max_rel %.6f bad %d/%zu\n",
               bs_h, bs_i, bs_g, bs_r, rms_s, mr_s, bad_s, ns);
    }
    free(S); free(U);
    return (bad_o || bad_s) ? 1 : 0;
}

static int cmp_tl_ref(const unsigned char *tl, const unsigned char *ref, const layout_t *l, double *max_o, double *max_s) {
    const _Float16 *Ot = (const _Float16 *)(tl + l->o_off), *Or = (const _Float16 *)(ref + l->o_off);
    const float *St = (const float *)(tl + l->s1_off), *Sr = (const float *)(ref + l->s1_off);
    double ss_o=0.0, ss_s=0.0, mr_o=0.0, mr_s=0.0;
    for (size_t i=0;i<(size_t)HV*T*D;i++) { double r=(double)(float)Or[i]; ss_o += r*r; }
    for (size_t i=0;i<(size_t)HV*D*D;i++) { double r=(double)Sr[i]; ss_s += r*r; }
    double rms_o=sqrt(ss_o/((double)HV*T*D)), rms_s=sqrt(ss_s/((double)HV*D*D));
    int bad=0;
    for (size_t i=0;i<(size_t)HV*T*D;i++) {
        double g=(double)(float)Ot[i], r=(double)(float)Or[i];
        double rel=fabs(g-r)/(fabs(r)+0.02*rms_o);
        if (rel>mr_o) mr_o=rel;
        if (rel>=0.1) bad++;
    }
    for (size_t i=0;i<(size_t)HV*D*D;i++) {
        double g=(double)St[i], r=(double)Sr[i];
        double rel=fabs(g-r)/(fabs(r)+0.02*rms_s);
        if (rel>mr_s) mr_s=rel;
        if (rel>=0.1) bad++;
    }
    *max_o=mr_o; *max_s=mr_s;
    return bad ? 1 : 0;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 3;
    layout_t l = make_layout();
    unsigned char *slab_ref = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, l.slab_sz);
    unsigned char *slab_tl = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, l.slab_sz);
    if (!slab_ref || !slab_tl) { printf("alloc fail slab_sz=%zu\n", l.slab_sz); return 1; }
    fill_inputs(slab_ref, &l);
    memcpy(slab_tl, slab_ref, l.slab_sz);
    remote_register_buf_attr2(slab_ref, l.slab_sz, rpcmem_to_fd(slab_ref),
        FASTRPC_ATTR_COHERENT | FASTRPC_ATTR_KEEP_MAP | FASTRPC_ATTR_TRY_MAP_STATIC);
    remote_register_buf_attr2(slab_tl, l.slab_sz, rpcmem_to_fd(slab_tl),
        FASTRPC_ATTR_COHERENT | FASTRPC_ATTR_KEEP_MAP | FASTRPC_ATTR_TRY_MAP_STATIC);
    struct remote_rpc_control_unsigned_module umod = { .domain = CDSP_DOMAIN_ID, .enable = 1 };
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &umod, sizeof umod);
    char uri[256]; snprintf(uri, sizeof uri, "%s&_dom=cdsp", attnops_URI);
    remote_handle64 ah = -1;
    if (attnops_open(uri, &ah)) { printf("open fail\n"); return 1; }

    int e = attnops_gdn(ah, slab_ref, (int)l.slab_sz, T, HK, HV);
    if (e) { printf("REF_GDN warmup ret=%d\n", e); return 1; }
    double t0 = now_s();
    for (int it=0; it<iters; it++) { clear_outputs(slab_ref, &l); e = attnops_gdn(ah, slab_ref, (int)l.slab_sz, T, HK, HV); }
    double ref_ms = (now_s() - t0) * 1e3 / iters;
    if (e) { printf("REF_GDN ret=%d\n", e); return 1; }

    e = attnops_tl_gdn_prefill(ah, slab_tl, (int)l.slab_sz, T, HK, HV);
    if (e) { printf("TL_GDN warmup ret=%d\n", e); return 1; }
    t0 = now_s();
    for (int it=0; it<iters; it++) { clear_outputs(slab_tl, &l); e = attnops_tl_gdn_prefill(ah, slab_tl, (int)l.slab_sz, T, HK, HV); }
    double tl_ms = (now_s() - t0) * 1e3 / iters;
    if (e) { printf("TL_GDN ret=%d\n", e); return 1; }

    double max_o=999, max_s=999, cmp_o=999, cmp_s=999;
    int fail = check_fp64_ref(slab_tl, &l, &max_o, &max_s);
    int cmp_fail = cmp_tl_ref(slab_tl, slab_ref, &l, &cmp_o, &cmp_s);
    printf("TL_GDN T=%d Hk=%d Hv=%d max_rel_o=%.6f max_rel_s1=%.6f ms=%.3f ref_ms=%.3f tl_vs_ref_o=%.6f tl_vs_ref_s1=%.6f ret=%d %s\n",
           T, HK, HV, max_o, max_s, tl_ms, ref_ms, cmp_o, cmp_s, e, (fail || cmp_fail) ? "FAIL" : "OK");
    attnops_close(ah);
    rpcmem_free(slab_ref); rpcmem_free(slab_tl);
    printf("TL_GDN_ALL_%s\n", (fail || cmp_fail) ? "FAIL" : "OK");
    return (fail || cmp_fail) ? 1 : 0;
}
