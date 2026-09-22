// fa_gqa_test.c - host validation for the TileLang-emitted GQA flash-attention
// prefill OpenCL kernel.
//
// Layout: Q/O [HQ*S, D] and K/V [HKV*S, D] row-major head-major fp16.
// Causal: query row q sees keys [0, q].  Softmax scale 1/sqrt(D).
// Verified against an fp64 reference with cosine similarity (default) or the
// rms-scaled relative error gate (TL_CHECK=maxrel).
//
// Build (Android, OpenCL-Headers at backend/gpu/OpenCL-Headers):
//   $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang \
//     -target aarch64-linux-android26 -march=armv8.2-a+dotprod+fp16 -std=c11 -O3 -fPIE -pie \
//     -I/root/project/backend/gpu/OpenCL-Headers fa_gqa_test.c -ldl -lm -o fa_gqa_test
// Run (device):
//   LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ./fa_gqa_test out/fa_gqa.cl fa_gqa_kernel_kernel
//
// Env: TL_S TL_HQ TL_HKV TL_D TL_TILE_Q TL_THREADS TL_ITERS
//      TL_CHECK=cosine|maxrel TL_COS_MIN (0.999) TL_CHECK_SAMPLES (2048, 0=all)

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"

typedef cl_int (*PFN_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*PFN_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_context (*PFN_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *, void (CL_CALLBACK *)(const char *, const void *, size_t, void *), void *, cl_int *);
typedef cl_command_queue (*PFN_clCreateCommandQueueWithProperties)(cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
typedef cl_program (*PFN_clCreateProgramWithSource)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
typedef cl_int (*PFN_clBuildProgram)(cl_program, cl_uint, const cl_device_id *, const char *, void (CL_CALLBACK *)(cl_program, void *), void *);
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

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint32_t lcg_next(uint32_t *s) { *s = (*s * 1664525u) + 1013904223u; return *s; }
static float rand_f32(uint32_t *s) { return ((lcg_next(s) >> 8) * (1.0f / 16777216.0f)) - 0.5f; }
static _Float16 rand_h(uint32_t *s) { return (_Float16)rand_f32(s); }

static int env_i(const char *name, int def) {
    const char *s = getenv(name);
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    return (end && *end == 0 && v > 0) ? (int)v : def;
}

typedef struct {
    cl_context ctx;
    cl_command_queue q;
    cl_device_id dev;
    cl_program prog;
    cl_kernel kernel;
} ClState;

static ClState cl_init_build(const char *cl_path, const char *kernel_name) {
    ClState s;
    memset(&s, 0, sizeof(s));
    load_cl();
    cl_platform_id pf; cl_uint np = 0;
    CK(my_clGetPlatformIDs(1, &pf, &np));
    cl_uint nd = 0;
    CK(my_clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &s.dev, &nd));
    cl_int err;
    cl_context_properties props[] = {0x40C2 /*CL_CONTEXT_PERF_HINT_QCOM*/, 0x40C3 /*HIGH*/, 0};
    s.ctx = my_clCreateContext(props, 1, &s.dev, NULL, NULL, &err); CK(err);
    const cl_queue_properties qprops[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    s.q = my_clCreateCommandQueueWithProperties(s.ctx, s.dev, qprops, &err); CK(err);
    size_t src_sz = 0;
    char *src = read_file(cl_path, &src_sz);
    const char *srcp = src;
    s.prog = my_clCreateProgramWithSource(s.ctx, 1, &srcp, &src_sz, &err); CK(err);
    err = my_clBuildProgram(s.prog, 1, &s.dev, "-cl-fast-relaxed-math", NULL, NULL);
    size_t logn = 0;
    my_clGetProgramBuildInfo(s.prog, s.dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &logn);
    if (logn > 1) {
        char *log = (char *)calloc(logn + 1, 1);
        if (log) { my_clGetProgramBuildInfo(s.prog, s.dev, CL_PROGRAM_BUILD_LOG, logn, log, NULL); printf("build_log:\n%s\n", log); free(log); }
    }
    if (err != CL_SUCCESS) exit(3);
    s.kernel = my_clCreateKernel(s.prog, kernel_name, &err); CK(err);
    free(src);
    return s;
}

// fp64 reference for one output element O[hq*S+q, d].
static double ref_elem(const _Float16 *Q, const _Float16 *K, const _Float16 *V,
                       int S, int HQ, int HKV, int D, int hq, int q, int d,
                       double *score_buf) {
    int g = hq / (HQ / HKV);
    const _Float16 *qh = Q + (size_t)hq * S * D;
    const _Float16 *kh = K + (size_t)g * S * D;
    const _Float16 *vh = V + (size_t)g * S * D;
    double scale = 1.0 / sqrt((double)D);
    double m = -1e30;
    for (int kv = 0; kv <= q; ++kv) {
        double acc = 0.0;
        for (int dd = 0; dd < D; ++dd)
            acc += (double)(float)qh[(size_t)q * D + dd] * (double)(float)kh[(size_t)kv * D + dd];
        acc *= scale;
        score_buf[kv] = acc;
        if (acc > m) m = acc;
    }
    double l = 0.0, out = 0.0;
    for (int kv = 0; kv <= q; ++kv) {
        double p = exp(score_buf[kv] - m);
        l += p;
        out += p * (double)(float)vh[(size_t)kv * D + d];
    }
    return out / l;
}

int main(int argc, char **argv) {
    const char *cl_path = argc > 1 ? argv[1] : "out/fa_gqa.cl";
    const char *kname = argc > 2 ? argv[2] : "fa_gqa_kernel_kernel";
    ClState s_ = cl_init_build(cl_path, kname);
    ClState *s = &s_;

    const int S = env_i("TL_S", 1024), HQ = env_i("TL_HQ", 16), HKV = env_i("TL_HKV", 4);
    const int D = env_i("TL_D", 256);
    const int TILE_Q = env_i("TL_TILE_Q", 32), THREADS = env_i("TL_THREADS", 64);
    const int iters = env_i("TL_ITERS", 20);
    const size_t nQ = (size_t)HQ * S * D, nKV = (size_t)HKV * S * D;
    uint32_t seed = 7;
    _Float16 *Q = (_Float16 *)malloc(nQ * sizeof(_Float16));
    _Float16 *K = (_Float16 *)malloc(nKV * sizeof(_Float16));
    _Float16 *V = (_Float16 *)malloc(nKV * sizeof(_Float16));
    _Float16 *O = (_Float16 *)calloc(nQ, sizeof(_Float16));
    if (!Q || !K || !V || !O) { fprintf(stderr, "fa alloc failed\n"); exit(2); }
    for (size_t i = 0; i < nQ; ++i) Q[i] = rand_h(&seed);
    for (size_t i = 0; i < nKV; ++i) K[i] = rand_h(&seed);
    for (size_t i = 0; i < nKV; ++i) V[i] = rand_h(&seed);

    cl_int err;
    cl_mem bQ = my_clCreateBuffer(s->ctx, CL_MEM_READ_ONLY, nQ * sizeof(_Float16), NULL, &err); CK(err);
    cl_mem bK = my_clCreateBuffer(s->ctx, CL_MEM_READ_ONLY, nKV * sizeof(_Float16), NULL, &err); CK(err);
    cl_mem bV = my_clCreateBuffer(s->ctx, CL_MEM_READ_ONLY, nKV * sizeof(_Float16), NULL, &err); CK(err);
    cl_mem bO = my_clCreateBuffer(s->ctx, CL_MEM_WRITE_ONLY, nQ * sizeof(_Float16), NULL, &err); CK(err);
    CK(my_clEnqueueWriteBuffer(s->q, bQ, CL_TRUE, 0, nQ * sizeof(_Float16), Q, 0, NULL, NULL));
    CK(my_clEnqueueWriteBuffer(s->q, bK, CL_TRUE, 0, nKV * sizeof(_Float16), K, 0, NULL, NULL));
    CK(my_clEnqueueWriteBuffer(s->q, bV, CL_TRUE, 0, nKV * sizeof(_Float16), V, 0, NULL, NULL));
    // NOTE: TVM codegen emits kernel params in ALPHABETICAL order:
    // (K, O, Q, V) for this kernel - not the TVMScript declaration order.
    CK(my_clSetKernelArg(s->kernel, 0, sizeof(bK), &bK));
    CK(my_clSetKernelArg(s->kernel, 1, sizeof(bO), &bO));
    CK(my_clSetKernelArg(s->kernel, 2, sizeof(bQ), &bQ));
    CK(my_clSetKernelArg(s->kernel, 3, sizeof(bV), &bV));

    const size_t local[2] = {(size_t)THREADS, 1};
    const size_t global[2] = {(size_t)((S + TILE_Q - 1) / TILE_Q) * (size_t)THREADS, (size_t)HQ};
    cl_event *events = (cl_event *)calloc((size_t)iters, sizeof(cl_event));
    if (!events) { fprintf(stderr, "fa event alloc failed\n"); exit(2); }
    CK(my_clFinish(s->q));
    double t0 = now_s();
    for (int it = 0; it < iters; ++it)
        CK(my_clEnqueueNDRangeKernel(s->q, s->kernel, 2, NULL, global, local, 0, NULL, &events[it]));
    CK(my_clFinish(s->q));
    double t1 = now_s();
    double event_ms = 0.0;
    int profiling_ok = 1;
    for (int it = 0; it < iters; ++it) {
        cl_ulong st = 0, en = 0;
        profiling_ok &= my_clGetEventProfilingInfo(events[it], CL_PROFILING_COMMAND_START, sizeof(st), &st, NULL) == CL_SUCCESS;
        profiling_ok &= my_clGetEventProfilingInfo(events[it], CL_PROFILING_COMMAND_END, sizeof(en), &en, NULL) == CL_SUCCESS;
        event_ms += (double)(en - st) * 1e-6;
        my_clReleaseEvent(events[it]);
    }
    free(events);
    const double wall_ms = (t1 - t0) * 1e3 / (double)iters;
    const double ms = profiling_ok ? event_ms / (double)iters : wall_ms;
    // Causal: per head ~ 2 gemms * (S^2/2 visible pairs) * D * 2 flops.
    const double tflops = (4.0 * (double)HQ * (double)S * ((double)S / 2.0) * (double)D) / (ms * 1.0e9);
    CK(my_clEnqueueReadBuffer(s->q, bO, CL_TRUE, 0, nQ * sizeof(_Float16), O, 0, NULL, NULL));
    printf("FA_GQA S=%d HQ=%d HKV=%d D=%d TILE_Q=%d TH=%d iters=%d ms_iter %.3f tflops %.3f timing=%s wall_ms_iter %.3f\n",
           S, HQ, HKV, D, TILE_Q, THREADS, iters, ms, tflops, profiling_ok ? "event" : "wall", wall_ms);

    // Sampled fp64 reference + cosine / rms-scaled max_rel check.
    int nsamp = env_i("TL_CHECK_SAMPLES", 2048);
    double *score_buf = (double *)malloc((size_t)S * sizeof(double));
    double *ref = (double *)malloc((size_t)nsamp * sizeof(double));
    _Float16 *got = (_Float16 *)malloc((size_t)nsamp * sizeof(_Float16));
    if (!score_buf || !ref || !got) { fprintf(stderr, "fa check alloc failed\n"); exit(2); }
    uint32_t sidx = 999;
    for (int t = 0; t < nsamp; ++t) {
        size_t idx = (size_t)(lcg_next(&sidx) % (uint32_t)nQ);
        int hq = (int)(idx / ((size_t)S * D));
        int q = (int)((idx / (size_t)D) % (size_t)S);
        int d = (int)(idx % (size_t)D);
        ref[t] = ref_elem(Q, K, V, S, HQ, HKV, D, hq, q, d, score_buf);
        got[t] = O[idx];
    }
    double rms = 0.0;
    for (int t = 0; t < nsamp; ++t) rms += ref[t] * ref[t];
    rms = sqrt(rms / nsamp);
    double max_rel = 0.0;
    size_t max_i = 0, bad = 0;
    long double dot = 0.0L, g2 = 0.0L, r2 = 0.0L;
    for (int t = 0; t < nsamp; ++t) {
        double g = (double)(float)got[t];
        double r = fabs(g - ref[t]) / (fabs(ref[t]) + 0.02 * rms);
        if (r > max_rel) { max_rel = r; max_i = t; }
        if (r > 0.1) ++bad;
        dot += (long double)g * (long double)ref[t];
        g2 += (long double)g * (long double)g;
        r2 += (long double)ref[t] * (long double)ref[t];
    }
    double cos_sim = (g2 > 0.0L && r2 > 0.0L) ? (double)(dot / (sqrt(g2) * sqrt(r2))) : 0.0;
    const char *check_env = getenv("TL_CHECK");
    int cosine_mode = !check_env || strcmp(check_env, "maxrel") != 0;
    double cos_min = getenv("TL_COS_MIN") ? atof(getenv("TL_COS_MIN")) : 0.999;
    int pass = cosine_mode ? (cos_sim >= cos_min) : (max_rel < 0.1);
    printf("FA_GQA max_rel %.6g rms %.6g bad %zu/%d cos %.9f sample[%zu] got %.9g ref %.9g %s%s\n",
           max_rel, rms, bad, nsamp, cos_sim, max_i, (double)(float)got[max_i], ref[max_i],
           pass ? "PASS" : "FAIL", cosine_mode ? " (cosine)" : "");

    my_clReleaseMemObject(bQ); my_clReleaseMemObject(bK);
    my_clReleaseMemObject(bV); my_clReleaseMemObject(bO);
    my_clReleaseKernel(s->kernel); my_clReleaseProgram(s->prog);
    my_clReleaseCommandQueue(s->q); my_clReleaseContext(s->ctx);
    free(Q); free(K); free(V); free(O); free(score_buf); free(ref); free(got);
    return pass ? 0 : 1;
}
