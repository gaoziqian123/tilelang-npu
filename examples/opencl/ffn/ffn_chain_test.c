// ffn_chain_test.c - plan-B split FFN chain validation:
//   G = A @ Wg, U = A @ Wu, H = silu(G) * U, out = H @ Wd
// all fp16, GR GEMM kernels emitted by gemm_nt.py + silu_mul.py.
//
// Shapes (Qwen3.5): A [M,K] = [960,2560], Wg/Wu [K,N] = [2560,9216],
// Wd [N2,N2d] = [9216,2560], out [960,2560].
//
// Build:
//   $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang \
//     -target aarch64-linux-android26 -march=armv8.2-a+dotprod+fp16 -std=c11 -O3 -fPIE -pie \
//     -I/root/project/backend/gpu/OpenCL-Headers ffn_chain_test.c -ldl -lm -o ffn_chain_test
// Run:
//   LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ./ffn_chain_test out/ffn_gate.cl out/silu_mul.cl out/ffn_down.cl
//   LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ./ffn_chain_test gate.cl up.cl silu.cl down.cl
//
// Env: TL_M TL_K TL_FF TL_N2 TL_THREADS TL_ITERS TL_CHECK_SAMPLES (256; 0=all)
//      TL_{GATE,UP,DOWN}_{BM,BN,THREADS} override per-stage launch shapes.
//      TL_FUSED_GATE_UP=1 treats argv[1] as a fused gate+up kernel
//      (emitted TileLang ABI: A,G,U,Wg,Wu).
//      Full fp64 FFN reference is very slow; for routine validation prefer a
//      large sample (e.g. TL_CHECK_SAMPLES=8192) over TL_CHECK_SAMPLES=0.

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"

#define DECLS(X) \
X(clGetPlatformIDs); X(clGetDeviceIDs); X(clCreateContext); \
X(clCreateCommandQueueWithProperties); X(clCreateProgramWithSource); \
X(clBuildProgram); X(clGetProgramBuildInfo); X(clCreateKernel); \
X(clCreateBuffer); X(clSetKernelArg); X(clEnqueueNDRangeKernel); \
X(clEnqueueWriteBuffer); X(clEnqueueReadBuffer); X(clFinish); \
X(clGetEventProfilingInfo); X(clReleaseEvent); X(clReleaseMemObject); \
X(clReleaseKernel); X(clReleaseProgram); X(clReleaseCommandQueue); X(clReleaseContext)

#define TD(n) typedef cl_int (*PFN_##n)(void); /* placeholder, replaced below */
/* explicit typedefs (clGetPlatformIDs signature kept exact) */
typedef cl_int (*PFN_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*PFN_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_context (*PFN_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
typedef cl_command_queue (*PFN_clCreateCommandQueueWithProperties)(cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
typedef cl_program (*PFN_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_int (*PFN_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
typedef cl_int (*PFN_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
typedef cl_kernel (*PFN_clCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_mem (*PFN_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
typedef cl_int (*PFN_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (*PFN_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*PFN_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*PFN_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*PFN_clFinish)(cl_command_queue);
typedef cl_int (*PFN_clGetEventProfilingInfo)(cl_event, cl_profiling_info, size_t, void *, size_t *);
typedef cl_int (*PFN_clReleaseEvent)(cl_event);
typedef cl_int (*PFN_clReleaseMemObject)(cl_mem);
typedef cl_int (*PFN_clReleaseKernel)(cl_kernel);
typedef cl_int (*PFN_clReleaseProgram)(cl_program);
typedef cl_int (*PFN_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*PFN_clReleaseContext)(cl_context);

#define DECL(n) static PFN_##n my_##n
DECL(clGetPlatformIDs); DECL(clGetDeviceIDs); DECL(clCreateContext);
DECL(clCreateCommandQueueWithProperties); DECL(clCreateProgramWithSource);
DECL(clBuildProgram); DECL(clGetProgramBuildInfo); DECL(clCreateKernel);
DECL(clCreateBuffer); DECL(clSetKernelArg); DECL(clEnqueueNDRangeKernel);
DECL(clEnqueueWriteBuffer); DECL(clEnqueueReadBuffer); DECL(clFinish);
DECL(clGetEventProfilingInfo); DECL(clReleaseEvent);
DECL(clReleaseMemObject); DECL(clReleaseKernel); DECL(clReleaseProgram);
DECL(clReleaseCommandQueue); DECL(clReleaseContext);

#define CK(x) do { cl_int e__ = (x); if (e__ != CL_SUCCESS) { \
    fprintf(stderr, "CL error %d at %s:%d\n", e__, __FILE__, __LINE__); exit(2); } } while (0)

static void load_cl(void) {
    const char *paths[] = {"/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so", "libOpenCL.so"};
    void *lib = NULL;
    for (int i = 0; i < 3 && !lib; ++i) lib = dlopen(paths[i], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen libOpenCL failed: %s\n", dlerror()); exit(2); }
#define GET(n) do { my_##n = (PFN_##n)dlsym(lib, #n); if (!my_##n) { fprintf(stderr, "missing %s\n", #n); exit(2); } } while (0)
    GET(clGetPlatformIDs); GET(clGetDeviceIDs); GET(clCreateContext);
    GET(clCreateCommandQueueWithProperties); GET(clCreateProgramWithSource);
    GET(clBuildProgram); GET(clGetProgramBuildInfo); GET(clCreateKernel);
    GET(clCreateBuffer); GET(clSetKernelArg); GET(clEnqueueNDRangeKernel);
    GET(clEnqueueWriteBuffer); GET(clEnqueueReadBuffer); GET(clFinish);
    GET(clGetEventProfilingInfo); GET(clReleaseEvent);
    GET(clReleaseMemObject); GET(clReleaseKernel); GET(clReleaseProgram);
    GET(clReleaseCommandQueue); GET(clReleaseContext);
#undef GET
}

static char *read_file(const char *path, size_t *sz_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fprintf(stderr, "oom reading %s\n", path); exit(2); }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    buf[sz] = 0;
    *sz_out = (size_t)sz;
    return buf;
}

static uint32_t lcg_next(uint32_t *s) { *s = (*s * 1664525u) + 1013904223u; return *s; }
static float rand_f32(uint32_t *s) { return ((lcg_next(s) >> 8) * (1.0f / 16777216.0f)) - 0.5f; }
static _Float16 rand_h_s(uint32_t *s) { return (_Float16)(rand_f32(s) * 0.1f); }

static int env_i(const char *name, int def) {
    const char *s = getenv(name);
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    return (end && *end == 0 && v > 0) ? (int)v : def;
}

static int env_i_allow_zero(const char *name, int def) {
    const char *s = getenv(name);
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    return (end && *end == 0 && v >= 0) ? (int)v : def;
}

static cl_platform_id g_pf;
static cl_device_id g_dev;

static cl_program build_prog(cl_context ctx, const char *cl_path, const char *kname) {
    cl_int err;
    size_t sz = 0;
    char *src = read_file(cl_path, &sz);
    const char *srcp = src;
    cl_program prog = my_clCreateProgramWithSource(ctx, 1, &srcp, &sz, &err); CK(err);
    err = my_clBuildProgram(prog, 1, &g_dev, "-cl-fast-relaxed-math", NULL, NULL);
    size_t logn = 0;
    my_clGetProgramBuildInfo(prog, g_dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &logn);
    if (logn > 1) {
        char *log = (char *)calloc(logn + 1, 1);
        if (log) { my_clGetProgramBuildInfo(prog, g_dev, CL_PROGRAM_BUILD_LOG, logn, log, NULL); printf("build_log(%s):\n%s\n", kname, log); free(log); }
    }
    if (err != CL_SUCCESS) exit(3);
    return prog;
}

static double ev_ms(cl_event ev) {
    cl_ulong st = 0, en = 0;
    if (my_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(st), &st, NULL) != CL_SUCCESS) return -1;
    if (my_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(en), &en, NULL) != CL_SUCCESS) return -1;
    return (double)(en - st) * 1e-6;
}

static void check_stage(const char *tag, const _Float16 *got_buf, size_t total,
                        double (*ref_fn)(size_t, void *), void *ctx, int nsamp) {
    int full_check = (nsamp == 0);
    if (full_check) nsamp = (int)total;
    double *ref = (double *)malloc((size_t)nsamp * sizeof(double));
    _Float16 *got = (_Float16 *)malloc((size_t)nsamp * sizeof(_Float16));
    uint32_t sidx = 4242;
    size_t *idxs = (size_t *)malloc((size_t)nsamp * sizeof(size_t));
    for (int t = 0; t < nsamp; ++t) {
        size_t idx = full_check ? (size_t)t : (size_t)(lcg_next(&sidx) % (uint32_t)total);
        idxs[t] = idx;
        ref[t] = ref_fn(idx, ctx);
        got[t] = got_buf[idx];
    }
    double rms = 0.0;
    for (int t = 0; t < nsamp; ++t) rms += ref[t] * ref[t];
    rms = sqrt(rms / nsamp);
    double max_rel = 0.0;
    size_t bad = 0, max_i = 0;
    for (int t = 0; t < nsamp; ++t) {
        double g = (double)(float)got[t];
        double r = fabs(g - ref[t]) / (fabs(ref[t]) + 0.02 * rms);
        if (r > max_rel) { max_rel = r; max_i = t; }
        if (r > 0.1) ++bad;
    }
    printf("%s max_rel %.6g rms %.6g bad %zu/%d %s[%zu] got %.9g ref %.9g %s\n",
           tag, max_rel, rms, bad, nsamp, full_check ? "full" : "sample", idxs[max_i], (double)(float)got[max_i], ref[max_i],
           bad == 0 ? "PASS" : "FAIL");
    free(ref); free(got); free(idxs);
}

/* ---- ref contexts ---- */
typedef struct {
    const _Float16 *A, *Wg, *Wu, *Wd, *Hdev;
    int M, K, FF, N2;
} RefCtx;

static double ref_G(size_t idx, void *ctx_) {
    RefCtx *c = (RefCtx *)ctx_;
    int m = (int)(idx / (size_t)c->FF), n = (int)(idx % (size_t)c->FF);
    double acc = 0.0;
    for (int k = 0; k < c->K; ++k)
        acc += (double)(float)c->A[(size_t)m * c->K + k] * (double)(float)c->Wg[(size_t)k * c->FF + n];
    return acc;
}

static double ref_U(size_t idx, void *ctx_) {
    RefCtx *c = (RefCtx *)ctx_;
    int m = (int)(idx / (size_t)c->FF), n = (int)(idx % (size_t)c->FF);
    double acc = 0.0;
    for (int k = 0; k < c->K; ++k)
        acc += (double)(float)c->A[(size_t)m * c->K + k] * (double)(float)c->Wu[(size_t)k * c->FF + n];
    return acc;
}

static double ref_H(size_t idx, void *ctx_) {
    RefCtx *c = (RefCtx *)ctx_;
    int m = (int)(idx / (size_t)c->FF), n = (int)(idx % (size_t)c->FF);
    double g = 0.0, u = 0.0;
    for (int k = 0; k < c->K; ++k) {
        double av = (double)(float)c->A[(size_t)m * c->K + k];
        g += av * (double)(float)c->Wg[(size_t)k * c->FF + n];
        u += av * (double)(float)c->Wu[(size_t)k * c->FF + n];
    }
    return (g / (1.0 + exp(-g))) * u;
}

static double ref_OUT(size_t idx, void *ctx_) {
    RefCtx *c = (RefCtx *)ctx_;
    int m = (int)(idx / (size_t)c->N2), n = (int)(idx % (size_t)c->N2);
    double acc = 0.0;
    for (int k = 0; k < c->FF; ++k)
        acc += (double)(float)c->Hdev[(size_t)m * c->FF + k] * (double)(float)c->Wd[(size_t)k * c->N2 + n];
    return acc;
}

int main(int argc, char **argv) {
    const char *gate_cl = argc > 1 ? argv[1] : "out/ffn_gate.cl";
    const char *up_cl = gate_cl;
    const char *silu_cl = argc > 2 ? argv[2] : "out/silu_mul.cl";
    const char *down_cl = argc > 3 ? argv[3] : "out/ffn_down.cl";
    if (argc > 4) { up_cl = argv[2]; silu_cl = argv[3]; down_cl = argv[4]; }

    const int M = env_i("TL_M", 960), K = env_i("TL_K", 2560);
    const int FF = env_i("TL_FF", 9216), N2 = env_i("TL_N2", 2560);
    const int THREADS = env_i("TL_THREADS", 64);
    const int GATE_BM = env_i("TL_GATE_BM", 32), GATE_BN = env_i("TL_GATE_BN", 128);
    const int UP_BM = env_i("TL_UP_BM", GATE_BM), UP_BN = env_i("TL_UP_BN", GATE_BN);
    const int DOWN_BM = env_i("TL_DOWN_BM", 32), DOWN_BN = env_i("TL_DOWN_BN", 128);
    const int GATE_THREADS = env_i("TL_GATE_THREADS", THREADS);
    const int UP_THREADS = env_i("TL_UP_THREADS", GATE_THREADS);
    const int DOWN_THREADS = env_i("TL_DOWN_THREADS", THREADS);
    const int iters = env_i("TL_ITERS", 10);
    const int nsamp = env_i_allow_zero("TL_CHECK_SAMPLES", 256);
    const int fused_gate_up = env_i_allow_zero("TL_FUSED_GATE_UP", 0);

    load_cl();
    cl_uint np = 0, nd = 0;
    CK(my_clGetPlatformIDs(1, &g_pf, &np));
    CK(my_clGetDeviceIDs(g_pf, CL_DEVICE_TYPE_GPU, 1, &g_dev, &nd));
    cl_int err;
    cl_context_properties props[] = {0x40C2, 0x40C3, 0};
    cl_context ctx = my_clCreateContext(props, 1, &g_dev, NULL, NULL, &err); CK(err);
    const cl_queue_properties qprops[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = my_clCreateCommandQueueWithProperties(ctx, g_dev, qprops, &err); CK(err);

    size_t nA = (size_t)M * K, nW = (size_t)K * FF, nWd = (size_t)FF * N2;
    size_t nH = (size_t)M * FF, nOut = (size_t)M * N2;
    uint32_t seed = 13;
    _Float16 *A = malloc(nA * 2), *Wg = malloc(nW * 2), *Wu = malloc(nW * 2);
    _Float16 *Wd = malloc(nWd * 2), *G = calloc(nH, 2), *U = calloc(nH, 2);
    _Float16 *H = calloc(nH, 2), *out = calloc(nOut, 2);
    if (!A || !Wg || !Wu || !Wd || !G || !U || !H || !out) { fprintf(stderr, "alloc failed\n"); exit(2); }
    for (size_t i = 0; i < nA; ++i) A[i] = rand_h_s(&seed);
    for (size_t i = 0; i < nW; ++i) Wg[i] = rand_h_s(&seed);
    for (size_t i = 0; i < nW; ++i) Wu[i] = rand_h_s(&seed);
    for (size_t i = 0; i < nWd; ++i) Wd[i] = rand_h_s(&seed);

    cl_mem bA = my_clCreateBuffer(ctx, CL_MEM_READ_ONLY, nA * 2, NULL, &err); CK(err);
    cl_mem bWg = my_clCreateBuffer(ctx, CL_MEM_READ_ONLY, nW * 2, NULL, &err); CK(err);
    cl_mem bWu = my_clCreateBuffer(ctx, CL_MEM_READ_ONLY, nW * 2, NULL, &err); CK(err);
    cl_mem bWd = my_clCreateBuffer(ctx, CL_MEM_READ_ONLY, nWd * 2, NULL, &err); CK(err);
    cl_mem bG = my_clCreateBuffer(ctx, CL_MEM_READ_WRITE, nH * 2, NULL, &err); CK(err);
    cl_mem bU = my_clCreateBuffer(ctx, CL_MEM_READ_WRITE, nH * 2, NULL, &err); CK(err);
    cl_mem bH = my_clCreateBuffer(ctx, CL_MEM_READ_WRITE, nH * 2, NULL, &err); CK(err);
    cl_mem bO = my_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nOut * 2, NULL, &err); CK(err);
    CK(my_clEnqueueWriteBuffer(q, bA, CL_TRUE, 0, nA * 2, A, 0, NULL, NULL));
    CK(my_clEnqueueWriteBuffer(q, bWg, CL_TRUE, 0, nW * 2, Wg, 0, NULL, NULL));
    CK(my_clEnqueueWriteBuffer(q, bWu, CL_TRUE, 0, nW * 2, Wu, 0, NULL, NULL));
    CK(my_clEnqueueWriteBuffer(q, bWd, CL_TRUE, 0, nWd * 2, Wd, 0, NULL, NULL));

    cl_program pGate = build_prog(ctx, gate_cl, "gate");
    cl_program pUp = (fused_gate_up || strcmp(up_cl, gate_cl) == 0) ? pGate : build_prog(ctx, up_cl, "up");
    cl_program pSilu = build_prog(ctx, silu_cl, "silu");
    cl_program pDown = build_prog(ctx, down_cl, "down");
    cl_kernel kGate = my_clCreateKernel(pGate, fused_gate_up ? "ffn_gate_up_kernel_kernel" : "gemm_nt_kernel_kernel", &err); CK(err);
    cl_kernel kUp = fused_gate_up ? NULL : my_clCreateKernel(pUp, "gemm_nt_kernel_kernel", &err); if (!fused_gate_up) { CK(err); }
    cl_kernel kSilu = my_clCreateKernel(pSilu, "silu_mul_kernel_kernel", &err); CK(err);
    cl_kernel kDown = my_clCreateKernel(pDown, "gemm_nt_kernel_kernel", &err); CK(err);

    // gemm grids: dim0 = ceil(N/bn)*threads, dim1 = ceil(M/bm)
    const size_t lgate[2] = {(size_t)GATE_THREADS, 1};
    const size_t lup[2] = {(size_t)UP_THREADS, 1};
    const size_t ldown[2] = {(size_t)DOWN_THREADS, 1};
    const size_t gg[2] = {(size_t)(FF / GATE_BN) * GATE_THREADS, (size_t)(M / GATE_BM)};
    const size_t gu[2] = {(size_t)(FF / UP_BN) * UP_THREADS, (size_t)(M / UP_BM)};
    const size_t gd[2] = {(size_t)(N2 / DOWN_BN) * DOWN_THREADS, (size_t)(M / DOWN_BM)};
    const size_t ls[1] = {256};
    const size_t gs[1] = {nH / 8};  /* work-items: 2048 elems per wg = 256 thr * 8 */

    double ms_gate = 0, ms_up = 0, ms_silu = 0, ms_down = 0;
    for (int it = 0; it < iters; ++it) {
        cl_event e;
        CK(my_clSetKernelArg(kGate, 0, sizeof(bA), &bA));
        if (fused_gate_up) {
            /* TileLang orders the fused signature as (A, G, U, Wg, Wu). */
            CK(my_clSetKernelArg(kGate, 1, sizeof(bG), &bG));
            CK(my_clSetKernelArg(kGate, 2, sizeof(bU), &bU));
            CK(my_clSetKernelArg(kGate, 3, sizeof(bWg), &bWg));
            CK(my_clSetKernelArg(kGate, 4, sizeof(bWu), &bWu));
        } else {
            CK(my_clSetKernelArg(kGate, 1, sizeof(bWg), &bWg));
            CK(my_clSetKernelArg(kGate, 2, sizeof(bG), &bG));
        }
        CK(my_clEnqueueNDRangeKernel(q, kGate, 2, NULL, gg, lgate, 0, NULL, &e));
        CK(my_clFinish(q)); ms_gate += ev_ms(e); my_clReleaseEvent(e);

        if (!fused_gate_up) {
            CK(my_clSetKernelArg(kUp, 0, sizeof(bA), &bA));
            CK(my_clSetKernelArg(kUp, 1, sizeof(bWu), &bWu));
            CK(my_clSetKernelArg(kUp, 2, sizeof(bU), &bU));
            CK(my_clEnqueueNDRangeKernel(q, kUp, 2, NULL, gu, lup, 0, NULL, &e));
            CK(my_clFinish(q)); ms_up += ev_ms(e); my_clReleaseEvent(e);
        }

        CK(my_clSetKernelArg(kSilu, 0, sizeof(bG), &bG));
        CK(my_clSetKernelArg(kSilu, 1, sizeof(bH), &bH));
        CK(my_clSetKernelArg(kSilu, 2, sizeof(bU), &bU));
        CK(my_clEnqueueNDRangeKernel(q, kSilu, 1, NULL, gs, ls, 0, NULL, &e));
        CK(my_clFinish(q)); ms_silu += ev_ms(e); my_clReleaseEvent(e);

        CK(my_clSetKernelArg(kDown, 0, sizeof(bH), &bH));
        CK(my_clSetKernelArg(kDown, 1, sizeof(bWd), &bWd));
        CK(my_clSetKernelArg(kDown, 2, sizeof(bO), &bO));
        CK(my_clEnqueueNDRangeKernel(q, kDown, 2, NULL, gd, ldown, 0, NULL, &e));
        CK(my_clFinish(q)); ms_down += ev_ms(e); my_clReleaseEvent(e);
    }
    ms_gate /= iters; ms_up /= iters; ms_silu /= iters; ms_down /= iters;
    const double ms_total = ms_gate + ms_up + ms_silu + ms_down;
    const double tflops = (4.0 * (double)M * (double)FF * (double)K +
                           2.0 * (double)M * (double)FF * (double)N2) / (ms_total * 1.0e9);
    printf("FFN_CHAIN M=%d FF=%d K=%d iters=%d%s gate %.3f up %.3f silu %.3f down %.3f total %.3f ms tflops %.3f\n",
           M, FF, K, iters, fused_gate_up ? " fused_gate_up" : "", ms_gate, ms_up, ms_silu, ms_down, ms_total, tflops);

    CK(my_clEnqueueReadBuffer(q, bG, CL_TRUE, 0, nH * 2, G, 0, NULL, NULL));
    CK(my_clEnqueueReadBuffer(q, bU, CL_TRUE, 0, nH * 2, U, 0, NULL, NULL));
    printf("DBG G[0]=%g U[0]=%g G[4321]=%g U[4321]=%g\n", (double)(float)G[0], (double)(float)U[0], (double)(float)G[4321], (double)(float)U[4321]);
    CK(my_clEnqueueReadBuffer(q, bH, CL_TRUE, 0, nH * 2, H, 0, NULL, NULL));
    CK(my_clEnqueueReadBuffer(q, bO, CL_TRUE, 0, nOut * 2, out, 0, NULL, NULL));

    RefCtx c = {A, Wg, Wu, Wd, H, M, K, FF, N2};
    check_stage("FFN_chain G", G, nH, ref_G, &c, nsamp);
    check_stage("FFN_chain U", U, nH, ref_U, &c, nsamp);
    check_stage("FFN_chain H", H, nH, ref_H, &c, nsamp);
    check_stage("FFN_chain OUT", out, nOut, ref_OUT, &c, nsamp);

    my_clReleaseMemObject(bA); my_clReleaseMemObject(bWg); my_clReleaseMemObject(bWu);
    my_clReleaseMemObject(bWd); my_clReleaseMemObject(bG); my_clReleaseMemObject(bU);
    my_clReleaseMemObject(bH); my_clReleaseMemObject(bO);
    my_clReleaseKernel(kGate); if (kUp) my_clReleaseKernel(kUp); my_clReleaseKernel(kSilu); my_clReleaseKernel(kDown);
    my_clReleaseProgram(pGate); if (pUp != pGate) my_clReleaseProgram(pUp); my_clReleaseProgram(pSilu); my_clReleaseProgram(pDown);
    my_clReleaseCommandQueue(q); my_clReleaseContext(ctx);
    free(A); free(Wg); free(Wu); free(Wd); free(G); free(U); free(H); free(out);
    return 0;
}
