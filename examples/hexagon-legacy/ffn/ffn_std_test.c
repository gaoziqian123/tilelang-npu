// ffn_std_test.c - host validation for TileLang standard-construct fused FFN.
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

#define M 1024
#define K 2560
#define FF 9216
#define PROF_BYTES 8192

static size_t al128(size_t v) { return (v + 127) & ~(size_t)127; }
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static uint32_t rs = 0x9e3779b9u;
static uint32_t lcg(void) { rs = rs * 1664525u + 1013904223u; return rs; }
static _Float16 rh(float s) { return (_Float16)(((int)(lcg() % 2001) - 1000) * s / 1000.0f); }

typedef struct { size_t x,wg,wu,wd,h,y,prof,sz; } lay_t;
static _Float16 *g_wg_rm, *g_wu_rm, *g_wd_rm;

static lay_t layout_std(void) {
    lay_t l; size_t off = 0;
    l.x = off; off = al128(off + (size_t)M * K * 2);
    l.wg = off; off = al128(off + (size_t)FF * K * 2);
    l.wu = off; off = al128(off + (size_t)FF * K * 2);
    l.wd = off; off = al128(off + (size_t)K * FF * 2);
    l.h = off; off = al128(off + (size_t)M * FF * 2);
    l.y = off; off = al128(off + (size_t)M * FF * 2); // padded row stride FF; valid cols [0,K)
    l.prof = off; l.sz = off + PROF_BYTES;
    return l;
}

static void pack_rm_to_wh(const _Float16 *src, int rows, int cols, _Float16 *dst) {
    for (int rt = 0; rt < rows / 32; rt++)
        for (int ct = 0; ct < cols / 32; ct++) {
            _Float16 *tile = dst + ((size_t)rt * (cols / 32) + ct) * 1024;
            for (int r = 0; r < 32; r++)
                for (int c = 0; c < 32; c++)
                    tile[(r / 2) * 64 + c * 2 + (r & 1)] = src[(size_t)(rt * 32 + c) * cols + ct * 32 + r];
        }
}

static int alloc_logical_weights(void) {
    g_wg_rm = (_Float16 *)malloc((size_t)FF * K * sizeof(_Float16));
    g_wu_rm = (_Float16 *)malloc((size_t)FF * K * sizeof(_Float16));
    g_wd_rm = (_Float16 *)malloc((size_t)K * FF * sizeof(_Float16));
    if (!g_wg_rm || !g_wu_rm || !g_wd_rm) {
        printf("logical weight alloc fail wg=%p wu=%p wd=%p\n", (void *)g_wg_rm, (void *)g_wu_rm, (void *)g_wd_rm);
        return 1;
    }
    return 0;
}

static void free_logical_weights(void) { free(g_wg_rm); free(g_wu_rm); free(g_wd_rm); }

static void fill_inputs(unsigned char *s, const lay_t *l) {
    memset(s, 0, l->sz);
    rs = 0x13579bdfu;
    _Float16 *X = (_Float16 *)(s + l->x), *Wg = (_Float16 *)(s + l->wg);
    _Float16 *Wu = (_Float16 *)(s + l->wu), *Wd = (_Float16 *)(s + l->wd);
    for (size_t i = 0; i < (size_t)M * K; i++) X[i] = rh(0.025f);
    for (size_t i = 0; i < (size_t)FF * K; i++) g_wg_rm[i] = rh(0.025f);
    for (size_t i = 0; i < (size_t)FF * K; i++) g_wu_rm[i] = rh(0.025f);
    for (size_t i = 0; i < (size_t)K * FF; i++) g_wd_rm[i] = rh(0.012f);
    pack_rm_to_wh(g_wg_rm, FF, K, Wg);
    pack_rm_to_wh(g_wu_rm, FF, K, Wu);
    pack_rm_to_wh(g_wd_rm, K, FF, Wd);
    memset(s + l->h, 0, l->sz - l->h);
}

static void clear_outputs(unsigned char *s, const lay_t *l) {
    memset(s + l->h, 0, l->sz - l->h);
}

typedef struct { double score, got, ref, tol; int m, k; } worst_t;
static void push_worst(worst_t *w, worst_t x) { for (int i = 0; i < 10; i++) if (x.score > w[i].score) { for (int j = 9; j > i; j--) w[j] = w[j-1]; w[i] = x; break; } }

static void phase_debug_h(const unsigned char *slab, const lay_t *l) {
    const _Float16 *X = (const _Float16 *)(slab + l->x), *Wg = g_wg_rm;
    const _Float16 *Wu = g_wu_rm, *Hgot = (const _Float16 *)(slab + l->h);
    double rr = 0.0, gg = 0.0, dot = 0.0, max_abs_got = 0.0, max_abs_ref = 0.0, max_abs_err = 0.0;
    double samples_got[4] = {0}, samples_ref[4] = {0};
    const int sm[4] = {0, 0, 1, 100};
    const int sf[4] = {0, 1, 0, 5000};
    for (int m = 0; m < M; m++) {
        const _Float16 *x = X + (size_t)m * K;
        for (int f = 0; f < FF; f++) {
            const _Float16 *wg = Wg + (size_t)f * K, *wu = Wu + (size_t)f * K;
            double g = 0.0, u = 0.0;
            for (int k = 0; k < K; k++) { double xv = (double)(float)x[k]; g += xv * (double)(float)wg[k]; u += xv * (double)(float)wu[k]; }
            double ref = (g / (1.0 + exp(-g))) * u;
            double got = (double)(float)Hgot[(size_t)m * FF + f];
            rr += ref * ref; gg += got * got; dot += ref * got;
            double ag = fabs(got), ar = fabs(ref), ae = fabs(got - ref);
            if (ag > max_abs_got) max_abs_got = ag;
            if (ar > max_abs_ref) max_abs_ref = ar;
            if (ae > max_abs_err) max_abs_err = ae;
            for (int i = 0; i < 4; i++) if (m == sm[i] && f == sf[i]) { samples_got[i] = got; samples_ref[i] = ref; }
        }
    }
    double rms_ref = sqrt(rr / ((double)M * FF));
    double rms_got = sqrt(gg / ((double)M * FF));
    double cos_h = dot / (sqrt(gg * rr) + 1e-300);
    printf("PHASE_DEBUG_H rms_ref=%.9g rms_got=%.9g cos_h=%.9f max_abs_ref=%.9g max_abs_got=%.9g max_abs_err=%.9g\n",
           rms_ref, rms_got, cos_h, max_abs_ref, max_abs_got, max_abs_err);
    for (int i = 0; i < 4; i++)
        printf("PHASE_DEBUG_H_SAMPLE m=%d f=%d got %.9g ref %.9g err %.9g\n", sm[i], sf[i], samples_got[i], samples_ref[i], samples_got[i] - samples_ref[i]);
}

static int check_fp64_ref(const unsigned char *slab, const lay_t *l) {
    const _Float16 *X = (const _Float16 *)(slab + l->x), *Wg = g_wg_rm;
    const _Float16 *Wu = g_wu_rm, *Wd = g_wd_rm;
    const _Float16 *Y = (const _Float16 *)(slab + l->y);
    double *H = (double *)malloc((size_t)M * FF * sizeof(double));
    double *Yref = (double *)malloc((size_t)M * K * sizeof(double));
    if (!H || !Yref) { printf("ref alloc fail H=%p Yref=%p\n", (void *)H, (void *)Yref); free(H); free(Yref); return 1; }
    for (int m = 0; m < M; m++) {
        const _Float16 *x = X + (size_t)m * K;
        for (int f = 0; f < FF; f++) {
            const _Float16 *wg = Wg + (size_t)f * K, *wu = Wu + (size_t)f * K;
            double g = 0.0, u = 0.0;
            for (int k = 0; k < K; k++) { double xv = (double)(float)x[k]; g += xv * (double)(float)wg[k]; u += xv * (double)(float)wu[k]; }
            H[(size_t)m * FF + f] = (g / (1.0 + exp(-g))) * u;
        }
    }
    double rr = 0.0;
    for (int m = 0; m < M; m++) {
        const double *h = H + (size_t)m * FF;
        for (int k = 0; k < K; k++) {
            const _Float16 *wd = Wd + (size_t)k * FF;
            double y = 0.0;
            for (int f = 0; f < FF; f++) y += h[f] * (double)(float)wd[f];
            Yref[(size_t)m * K + k] = y;
            rr += y * y;
        }
    }
    double rms = sqrt(rr / ((double)M * K));
    double dot = 0.0, gg = 0.0; worst_t worst[10] = {0}; int fail = 0;
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) {
        double ref = Yref[(size_t)m * K + k], got = (double)(float)Y[(size_t)m * FF + k];
        dot += got * ref; gg += got * got;
        double tol = fmax(0.1 * (fabs(ref) + 0.02 * rms), 2.5e-4);
        double score = fabs(got - ref) / (tol + 1e-300);
        if (score > 1.0) { fail++; push_worst(worst, (worst_t){score, got, ref, tol, m, k}); }
    }
    double cos_y = dot / (sqrt(gg * rr) + 1e-300);
    printf("FINAL_CRITERION cos_y=%.9f final_fail_y=%d/%d rms=%.9g %s\n", cos_y, fail, M*K, rms, (cos_y >= 0.999999 && fail == 0) ? "PASS" : "FAIL");
    if (cos_y < 0.999999 || fail) {
        printf("FINAL_WORST_Y\n");
        for (int i = 0; i < 10 && worst[i].score > 0; i++)
            printf("  #%d score=%.6f m=%d k=%d got %.8f ref %.8f tol %.8g\n", i, worst[i].score, worst[i].m, worst[i].k, worst[i].got, worst[i].ref, worst[i].tol);
    }
    free(H); free(Yref); return (cos_y >= 0.999999 && fail == 0) ? 0 : 1;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 1;
    int abl = argc > 2 ? atoi(argv[2]) : 0;
    int phase_debug = argc > 3 ? atoi(argv[3]) : 0;
    lay_t l = layout_std();
    printf("FFN_STD_LAYOUT weights=WH_PREPACKED x=%zu wg=%zu wu=%zu wd=%zu h=%zu y=%zu prof=%zu sz=%zu\n", l.x,l.wg,l.wu,l.wd,l.h,l.y,l.prof,l.sz);
    if (alloc_logical_weights()) return 1;
    unsigned char *slab = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, l.sz);
    if (!slab) { printf("alloc fail slab_sz=%zu\n", l.sz); free_logical_weights(); return 1; }
    fill_inputs(slab, &l);
    remote_register_buf_attr2(slab, l.sz, rpcmem_to_fd(slab), FASTRPC_ATTR_COHERENT|FASTRPC_ATTR_KEEP_MAP|FASTRPC_ATTR_TRY_MAP_STATIC);
    struct remote_rpc_control_unsigned_module umod = {.domain = CDSP_DOMAIN_ID, .enable = 1}; remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &umod, sizeof umod);
    char uri[256]; snprintf(uri, sizeof uri, "%s&_dom=cdsp", attnops_URI); remote_handle64 ah = -1; if (attnops_open(uri, &ah)) { printf("open fail\n"); return 1; }
    int e = attnops_tl_ffn_std(ah, slab, (int)l.sz, abl); if (e) { printf("TL_FFN_STD warmup ret=%d slab_sz=%zu\n", e, l.sz); return 1; }
    double t0 = now_s();
    for (int it = 0; it < iters; it++) { clear_outputs(slab, &l); e = attnops_tl_ffn_std(ah, slab, (int)l.sz, abl); if (e) break; }
    double ms = (now_s() - t0) * 1e3 / (iters ? iters : 1);
    if (!e && phase_debug) phase_debug_h(slab, &l);
    int fail = e ? 1 : check_fp64_ref(slab, &l);
    int32_t *p = (int32_t *)(slab + l.prof);
    printf("FFN_STD M=%d K=%d FF=%d ret=%d ms=%.3f prof_ticks total=%d stage=%d mm=%d unperm=%d wall=%d %s\n", M,K,FF,e,ms,p[0],p[1],p[2],p[3],p[4],fail?"FAIL":"OK");
    printf("FFN_STD_EMIT_PROF ticks legacy0=%d stage=%d mm=%d unperm=%d wall=%d pool_silu=%d pool_y=%d\n", p[0],p[1],p[2],p[3],p[4],p[5],p[6]);
    printf("FFN_POOL p5=%d p6=%d p7=%d p8=%d p9=%d p10=%d p11=%d p12=%d p13=%d p14=%d p15=%d p16=%d p17=%d\n",
           p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17]);
    attnops_close(ah); rpcmem_free(slab); free_logical_weights(); return fail ? 1 : 0;
}
