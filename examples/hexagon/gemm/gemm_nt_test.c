// gemm_nt_test.c - host validation for TileLang-emitted attnops_tl_gemm_nt.
// Runs Qwen3.5 GEMM_NT anchor shapes and compares against fp64 reference using
// R12 rms-scaled relative error: |got-ref| / (|ref| + 0.02*rms(ref)).
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "sdkl.h"
#include "attnops.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static size_t al128(size_t v) { return (v + 127) & ~(size_t)127; }

static uint32_t rng_state = 0x1234567u;
static uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return rng_state = x;
}
static _Float16 rand_h(void) {
    int v = (int)(xorshift32() % 2001u) - 1000;
    return (_Float16)((float)v * 0.0005f);
}

// B[N,K] row-major (NT contract) -> WH image of W[K,N], tile order [cb][kb].
static void wh_convert_nt(const _Float16 *B, int N, int K, _Float16 *dst) {
    for (int cb = 0; cb < N / 32; cb++)
        for (int kb = 0; kb < K / 32; kb++) {
            _Float16 *t = dst + ((size_t)cb * (K / 32) + kb) * 1024;
            for (int r = 0; r < 32; r++)
                for (int c = 0; c < 32; c++)
                    t[(r / 2) * 64 + c * 2 + (r % 2)] = B[(size_t)(cb * 32 + c) * K + kb * 32 + r];
        }
}

static int check_shape(const _Float16 *A, const _Float16 *B, const _Float16 *C,
                       int M, int N, int K, double *max_rel_out) {
    const int nsamp = 512;
    double refs[512];
    int is[512], js[512];
    uint32_t save = rng_state;
    rng_state = 0x9e3779b9u ^ (uint32_t)(M * 131 + N * 17 + K);
    double ss = 0.0;
    for (int t = 0; t < nsamp; t++) {
        int i = (int)(xorshift32() % (uint32_t)M);
        int j = (int)(xorshift32() % (uint32_t)N);
        if (t < 16) { i = t % M; j = (t * 97) % N; }
        double acc = 0.0;
        for (int k = 0; k < K; k++)
            acc += (double)(float)A[(size_t)i * K + k] * (double)(float)B[(size_t)j * K + k];
        refs[t] = acc;
        is[t] = i;
        js[t] = j;
        ss += acc * acc;
    }
    rng_state = save;
    double rms = sqrt(ss / nsamp);
    double max_rel = 0.0;
    int bad = 0, bi = -1, bj = -1, bt = -1;
    double bgot = 0.0, bref = 0.0;
    for (int t = 0; t < nsamp; t++) {
        double got = (double)(float)C[(size_t)is[t] * N + js[t]];
        double rel = fabs(got - refs[t]) / (fabs(refs[t]) + 0.02 * rms);
        if (rel > max_rel) {
            max_rel = rel;
            bi = is[t]; bj = js[t]; bt = t; bgot = got; bref = refs[t];
        }
        if (rel >= 0.1) bad++;
    }
    *max_rel_out = max_rel;
    if (bad) {
        printf("  worst sample=%d at (%d,%d) got %.6f ref %.6f rms %.6f max_rel %.6f bad %d/%d\n",
               bt, bi, bj, bgot, bref, rms, max_rel, bad, nsamp);
        return 1;
    }
    return 0;
}

static int call_tl(remote_handle64 ah, int variant, unsigned char *slab, int slab_sz,
                   unsigned char *w, int w_sz, int M, int N, int K, int abl) {
    if (variant == 1) return attnops_tl_gemm_bn128(ah, slab, slab_sz, w, w_sz, M, N, K, abl);
    if (variant == 2) return attnops_tl_gemm_bn32(ah, slab, slab_sz, w, w_sz, M, N, K, abl);
    return attnops_tl_gemm_nt(ah, slab, slab_sz, w, w_sz, M, N, K, abl);
}

static const char *variant_name(int variant) {
    if (variant == 1) return "TL_GEMM_BN128";
    if (variant == 2) return "TL_GEMM_BN32";
    return "TL_GEMM";
}

static int run_one_variant(remote_handle64 ah, int M, int N, int K, int iters, int variant) {
    size_t a_sz = (size_t)M * K * 2;
    size_t c_off = al128(a_sz);
    size_t c_sz = (size_t)M * N * 2;
    size_t prof_off = al128(c_off + c_sz);
    size_t slab_sz = prof_off + 8192;
    size_t w_sz = (size_t)N * K * 2;

    unsigned char *slab = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, slab_sz);
    _Float16 *w = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, w_sz);
    _Float16 *B = malloc(w_sz);
    if (!slab || !w || !B) { printf("alloc fail M=%d N=%d K=%d\n", M, N, K); return 1; }
    _Float16 *A = (_Float16 *)slab;
    _Float16 *C = (_Float16 *)(slab + c_off);
    int *prof = (int *)(slab + prof_off);

    rng_state = 0x1234u ^ (uint32_t)(M * 1000003 + N * 9176 + K);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rand_h();
    for (size_t i = 0; i < (size_t)N * K; i++) B[i] = rand_h();
    double tc = now_s();
    wh_convert_nt(B, N, K, w);
    fprintf(stderr, "tl wh convert %dx%dx%d %.2fs (%.1f MB)\n", M, N, K, now_s() - tc, (double)w_sz / 1e6);

    remote_register_buf_attr2(slab, slab_sz, rpcmem_to_fd(slab),
        FASTRPC_ATTR_COHERENT | FASTRPC_ATTR_KEEP_MAP | FASTRPC_ATTR_TRY_MAP_STATIC);
    remote_register_buf_attr2(w, w_sz, rpcmem_to_fd(w),
        FASTRPC_ATTR_COHERENT | FASTRPC_ATTR_KEEP_MAP | FASTRPC_ATTR_TRY_MAP_STATIC);

    int abl = getenv("ABL") ? atoi(getenv("ABL")) : 0;
    int tl_only = getenv("TL_ONLY") ? atoi(getenv("TL_ONLY")) : 0;
    int ref_only = getenv("REF_ONLY") ? atoi(getenv("REF_ONLY")) : 0;
    int e = 0;
    double ref_ms = 0.0;
    if (!tl_only && (N % 256 == 0)) {
        e = attnops_gemm_nt(ah, slab, (int)slab_sz, (unsigned char *)w, (int)w_sz, M, N, K, abl);
        if (e) { printf("REF_GEMM shape=%dx%dx%d ret=%d (warmup)\n", M, N, K, e); return 1; }
        double r0 = now_s();
        for (int it = 0; it < iters; it++)
            e = attnops_gemm_nt(ah, slab, (int)slab_sz, (unsigned char *)w, (int)w_sz, M, N, K, abl);
        ref_ms = (now_s() - r0) * 1e3 / iters;
        if (ref_only) {
            double ref_tflops = 2.0 * M * N * K / (ref_ms * 1e-3) / 1e12;
            double max_rel = 999.0;
            int fail = e || check_shape(A, B, C, M, N, K, &max_rel);
            printf("REF_GEMM M=%d N=%d K=%d max_rel=%.6f ms=%.3f TFLOPS=%.3f ret=%d %s\n",
                   M, N, K, max_rel, ref_ms, ref_tflops, e, fail ? "FAIL" : "OK");
            rpcmem_free(w);
            rpcmem_free(slab);
            free(B);
            return fail;
        }
    }

    e = call_tl(ah, variant, slab, (int)slab_sz, (unsigned char *)w, (int)w_sz, M, N, K, abl);
    if (e) { printf("%s shape=%dx%dx%d ret=%d (warmup)\n", variant_name(variant), M, N, K, e); return 1; }
    double t0 = now_s();
    for (int it = 0; it < iters; it++)
        e = call_tl(ah, variant, slab, (int)slab_sz, (unsigned char *)w, (int)w_sz, M, N, K, abl);
    double ms = (now_s() - t0) * 1e3 / iters;
    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    double ref_tflops = ref_ms > 0.0 ? 2.0 * M * N * K / (ref_ms * 1e-3) / 1e12 : 0.0;
    double gap = ref_ms > 0.0 ? (ms / ref_ms - 1.0) * 100.0 : 0.0;
    double max_rel = 999.0;
    int fail = e || check_shape(A, B, C, M, N, K, &max_rel);
    double s = 19.2e3;
    printf("%s M=%d N=%d K=%d max_rel=%.6f ms=%.3f TFLOPS=%.3f ref_ms=%.3f ref_TFLOPS=%.3f gap_pct=%.1f ret=%d %s\n",
           variant_name(variant), M, N, K, max_rel, ms, tflops, ref_ms, ref_tflops, gap, e, fail ? "FAIL" : "OK");
    printf("  prof(us): wstage %d act %d mm %d wb %d total %d\n",
           (int)(prof[0] / s), (int)(prof[1] / s), (int)(prof[2] / s),
           (int)(prof[3] / s), (int)(prof[4] / s));
    rpcmem_free(w);
    rpcmem_free(slab);
    free(B);
    return fail;
}

static int run_one(remote_handle64 ah, int M, int N, int K, int iters) {
    return run_one_variant(ah, M, N, K, iters, 0);
}

static int run_small(int iters) {
    struct remote_rpc_control_unsigned_module umod = { .domain = CDSP_DOMAIN_ID, .enable = 1 };
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &umod, sizeof umod);
    int shapes[][4] = {{960, 256, 2560, 1}, {960, 128, 2560, 2}};
    int rc = 0;
    char uri[256];
    snprintf(uri, sizeof uri, "%s&_dom=cdsp", attnops_URI);
    remote_handle64 ah = -1;
    if (attnops_open(uri, &ah)) { printf("open fail\n"); return 1; }
    for (int s = 0; s < 2; s++) {
        rc |= run_one_variant(ah, shapes[s][0], shapes[s][1], shapes[s][2], iters, shapes[s][3]);
    }
    attnops_close(ah);
    printf("TL_GEMM_SMALL_%s\n", rc ? "FAIL" : "OK");
    return rc ? 1 : 0;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 3;
    if (argc > 2 && !strcmp(argv[2], "small")) return run_small(iters);
    int only = argc > 2 ? atoi(argv[2]) : -1;
    if (argc >= 5) {
        int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
        int variant = getenv("TL_VARIANT") ? atoi(getenv("TL_VARIANT")) : 0;
        struct remote_rpc_control_unsigned_module umod = { .domain = CDSP_DOMAIN_ID, .enable = 1 };
        remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &umod, sizeof umod);
        char uri[256];
        snprintf(uri, sizeof uri, "%s&_dom=cdsp", attnops_URI);
        remote_handle64 ah = -1;
        if (attnops_open(uri, &ah)) { printf("open fail\n"); return 1; }
        int rc = run_one_variant(ah, M, N, K, iters, variant);
        attnops_close(ah);
        printf("TL_GEMM_ONE_%s\n", rc ? "FAIL" : "OK");
        return rc ? 1 : 0;
    }
    if (only < 0) {
        int rc = 0;
        for (int s = 0; s < 5; s++) {
            char cmd[256];
            snprintf(cmd, sizeof cmd, "%s %d %d", argv[0], iters, s);
            int st = system(cmd);
            if (st == -1 || !WIFEXITED(st) || WEXITSTATUS(st) != 0) rc = 1;
        }
        printf("TL_GEMM_ALL_%s\n", rc ? "FAIL" : "OK");
        return rc ? 1 : 0;
    }
    int shapes[][3] = {
        {960, 8192, 2560}, {960, 4096, 2560}, {960, 2560, 4096},
        {960, 12288, 2560}, {960, 6144, 2560},
    };
    struct remote_rpc_control_unsigned_module umod = { .domain = CDSP_DOMAIN_ID, .enable = 1 };
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &umod, sizeof umod);
    int rc = 0;
    for (int s = 0; s < 5; s++) {
        if (only >= 0 && s != only) continue;
        char uri[256];
        snprintf(uri, sizeof uri, "%s&_dom=cdsp", attnops_URI);
        remote_handle64 ah = -1;
        if (attnops_open(uri, &ah)) { printf("open fail\n"); return 1; }
        rc |= run_one(ah, shapes[s][0], shapes[s][1], shapes[s][2], iters);
        attnops_close(ah);
    }
    printf("TL_GEMM_ALL_%s\n", rc ? "FAIL" : "OK");
    return rc ? 1 : 0;
}
