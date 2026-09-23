// gemm_nt_test.c - host validation for TileLang-emitted gemm_nt OpenCL kernels.
//
// Runs the emitted GEMM_NT kernel against an fp64 reference.  fp16 inner-chain
// accumulation is verified with cosine similarity (default), fp32-accumulated
// kernels can use the rms-scaled relative error gate |got-ref|/(|ref|+0.02*rms).
//
// Build (Android, OpenCL-Headers at backend/gpu/OpenCL-Headers):
//   $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang \
//     -target aarch64-linux-android26 -march=armv8.2-a+dotprod+fp16 -std=c11 -O3 -fPIE -pie \
//     -I/root/project/backend/gpu/OpenCL-Headers gemm_nt_test.c -ldl -lm -o gemm_nt_test
// Run (device, e.g. OnePlus 13):
//   LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ./gemm_nt_test out/gemm_nt.cl gemm_nt_kernel_kernel
//
// Env: TL_M TL_N TL_K TL_BM TL_BN TL_THREADS TL_ITERS TL_B_LAYOUT=kn|nk TL_B_TEX=1
//      TL_CHECK=cosine|maxrel TL_COS_MIN (default 0.999) TL_CHECK_SAMPLES (0=all)

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
typedef cl_mem (*PFN_clCreateImage2D)(cl_context, cl_mem_flags, const cl_image_format *, size_t, size_t, size_t, void *, cl_int *);
typedef cl_int (*PFN_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (*PFN_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*PFN_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
typedef cl_int (*PFN_clEnqueueWriteImage)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
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
DECL(clCreateBuffer); DECL(clCreateImage2D); DECL(clSetKernelArg); DECL(clEnqueueNDRangeKernel);
DECL(clEnqueueWriteBuffer); DECL(clEnqueueWriteImage); DECL(clEnqueueReadBuffer); DECL(clFinish);
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
    GET(clCreateBuffer); GET(clCreateImage2D); GET(clSetKernelArg); GET(clEnqueueNDRangeKernel);
    GET(clEnqueueWriteBuffer); GET(clEnqueueWriteImage); GET(clEnqueueReadBuffer); GET(clFinish);
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

static void print_build_log(cl_program prog, cl_device_id dev, const char *prefix) {
    size_t n = 0;
    my_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &n);
    char *log = (char *)calloc(n + 1, 1);
    if (log && n) my_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log, NULL);
    printf("%s build_log_begin\n%.*s\n%s build_log_end\n", prefix, (int)n, log ? log : "", prefix);
    free(log);
}

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
    print_build_log(s.prog, s.dev, err == CL_SUCCESS ? "BUILD_OK" : "BUILD_FAIL");
    if (err != CL_SUCCESS) exit(3);
    s.kernel = my_clCreateKernel(s.prog, kernel_name, &err); CK(err);
    free(src);
    return s;
}

static double rms_from_refs(const double *ref, size_t n) {
    long double ss = 0.0;
    for (size_t i = 0; i < n; ++i) ss += (long double)ref[i] * (long double)ref[i];
    return sqrt((double)(ss / (long double)n));
}

static int check_h(const char *tag, const _Float16 *got, const double *ref, size_t n) {
    double rms = rms_from_refs(ref, n);
    double max_rel = 0.0;
    size_t max_i = 0, bad = 0;
    long double dot = 0.0L, g2 = 0.0L, r2 = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        double g = (double)(float)got[i];
        double r = fabs(g - ref[i]) / (fabs(ref[i]) + 0.02 * rms);
        if (r > max_rel) { max_rel = r; max_i = i; }
        if (r > 0.1) ++bad;
        dot += (long double)g * (long double)ref[i];
        g2 += (long double)g * (long double)g;
        r2 += (long double)ref[i] * (long double)ref[i];
    }
    double cos_sim = (g2 > 0.0L && r2 > 0.0L) ? (double)(dot / (sqrt(g2) * sqrt(r2))) : 0.0;
    const char *check_env = getenv("TL_CHECK");
    int cosine_mode = !check_env || strcmp(check_env, "maxrel") != 0;
    double cos_min = getenv("TL_COS_MIN") ? atof(getenv("TL_COS_MIN")) : 0.999;
    int pass = cosine_mode ? (cos_sim >= cos_min) : (max_rel < 0.1);
    printf("%s max_rel %.6g rms %.6g bad %zu/%zu cos %.9f sample[%zu] got %.9g ref %.9g %s%s\n",
           tag, max_rel, rms, bad, n, cos_sim, max_i, (double)(float)got[max_i], ref[max_i],
           pass ? "PASS" : "FAIL", cosine_mode ? " (cosine)" : "");
    return pass ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *cl_path = argc > 1 ? argv[1] : "out/gemm_nt.cl";
    const char *kname = argc > 2 ? argv[2] : "gemm_nt_kernel_kernel";
    ClState s_ = cl_init_build(cl_path, kname);
    ClState *s = &s_;

    const int M = env_i("TL_M", 1024), N = env_i("TL_N", 2560), K = env_i("TL_K", 2560);
    const int BM = env_i("TL_BM", 32), BN = env_i("TL_BN", 128), THREADS = env_i("TL_THREADS", 64);
    const int iters = env_i("TL_ITERS", 20);
    const char *b_layout_env = getenv("TL_B_LAYOUT");
    const int b_kn = !b_layout_env || !strcmp(b_layout_env, "kn");
    const int b_tex = getenv("TL_B_TEX") && strcmp(getenv("TL_B_TEX"), "0");
    if (b_tex && (!b_kn || (N % 4) != 0)) { fprintf(stderr, "TL_B_TEX requires KN layout and N%%4==0\n"); exit(2); }
    const size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
    uint32_t seed = 3;
    _Float16 *A = (_Float16 *)malloc(nA * sizeof(_Float16));
    _Float16 *B = (_Float16 *)malloc(nB * sizeof(_Float16));
    _Float16 *C = (_Float16 *)calloc(nC, sizeof(_Float16));
    int nsamp = env_i("TL_CHECK_SAMPLES", 0);
    double *ref = (double *)calloc(nsamp > 0 ? (size_t)nsamp : nC, sizeof(double));
    if (!A || !B || !C || !ref) { fprintf(stderr, "gemm alloc failed\n"); exit(2); }
    for (size_t i = 0; i < nA; ++i) A[i] = rand_h(&seed);
    for (size_t i = 0; i < nB; ++i) B[i] = rand_h(&seed);
    cl_int err;
    cl_mem bA = my_clCreateBuffer(s->ctx, CL_MEM_READ_ONLY, nA * sizeof(_Float16), NULL, &err); CK(err);
    cl_mem bB = NULL;
    if (b_tex) {
        cl_image_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.image_channel_order = CL_RGBA;
        fmt.image_channel_data_type = CL_HALF_FLOAT;
        bB = my_clCreateImage2D(s->ctx, CL_MEM_READ_ONLY, &fmt, (size_t)N / 4, (size_t)K, 0, NULL, &err); CK(err);
    } else {
        bB = my_clCreateBuffer(s->ctx, CL_MEM_READ_ONLY, nB * sizeof(_Float16), NULL, &err); CK(err);
    }
    cl_mem bC = my_clCreateBuffer(s->ctx, CL_MEM_WRITE_ONLY, nC * sizeof(_Float16), NULL, &err); CK(err);
    CK(my_clEnqueueWriteBuffer(s->q, bA, CL_TRUE, 0, nA * sizeof(_Float16), A, 0, NULL, NULL));
    if (b_tex) {
        const size_t origin[3] = {0, 0, 0};
        const size_t region[3] = {(size_t)N / 4, (size_t)K, 1};
        CK(my_clEnqueueWriteImage(s->q, bB, CL_TRUE, origin, region, (size_t)N * sizeof(_Float16), 0, B, 0, NULL, NULL));
    } else {
        CK(my_clEnqueueWriteBuffer(s->q, bB, CL_TRUE, 0, nB * sizeof(_Float16), B, 0, NULL, NULL));
    }
    CK(my_clSetKernelArg(s->kernel, 0, sizeof(bA), &bA));
    CK(my_clSetKernelArg(s->kernel, 1, sizeof(bB), &bB));
    CK(my_clSetKernelArg(s->kernel, 2, sizeof(bC), &bC));
    const size_t local[2] = {(size_t)THREADS, 1};
    const size_t global[2] = {(size_t)((N + BN - 1) / BN) * (size_t)THREADS,
                              (size_t)((M + BM - 1) / BM)};
    cl_event *events = (cl_event *)calloc((size_t)iters, sizeof(cl_event));
    if (!events) { fprintf(stderr, "gemm event alloc failed\n"); exit(2); }
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
    const double tflops = (2.0 * (double)M * (double)N * (double)K) / (ms * 1.0e9);
    CK(my_clEnqueueReadBuffer(s->q, bC, CL_TRUE, 0, nC * sizeof(_Float16), C, 0, NULL, NULL));
    printf("GEMM M=%d N=%d K=%d BM=%d BN=%d TH=%d B_LAYOUT=%s B_TEX=%d iters=%d ms_iter %.3f tflops %.3f timing=%s wall_ms_iter %.3f\n",
           M, N, K, BM, BN, THREADS, b_kn ? "kn" : "nk", b_tex, iters, ms, tflops, profiling_ok ? "event" : "wall", wall_ms);

    int rc = 0;
    if (nsamp > 0) {
        // Sampled fp64 reference (large shapes: full-K dot per sample).
        _Float16 *Cs = (_Float16 *)malloc((size_t)nsamp * sizeof(_Float16));
        if (!Cs) { fprintf(stderr, "gemm sample alloc failed\n"); exit(2); }
        uint32_t sidx = 12345;
        for (int t = 0; t < nsamp; ++t) {
            size_t idx = (size_t)(lcg_next(&sidx) % (uint32_t)nC);
            int i = (int)(idx / (size_t)N), j = (int)(idx - (size_t)i * N);
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                _Float16 bv = b_kn ? B[(size_t)k * N + j] : B[(size_t)j * K + k];
                acc += (double)(float)A[(size_t)i * K + k] * (double)(float)bv;
            }
            ref[t] = acc;
            Cs[t] = C[idx];
        }
        rc = check_h("GEMM(sample)", Cs, ref, (size_t)nsamp);
        free(Cs);
    } else {
        for (size_t i = 0; i < nC; ++i) {
            int r = (int)(i / (size_t)N), j = (int)(i - (size_t)r * N);
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                _Float16 bv = b_kn ? B[(size_t)k * N + j] : B[(size_t)j * K + k];
                acc += (double)(float)A[(size_t)r * K + k] * (double)(float)bv;
            }
            ref[i] = acc;
        }
        rc = check_h("GEMM", C, ref, nC);
    }

    my_clReleaseMemObject(bA); my_clReleaseMemObject(bB); my_clReleaseMemObject(bC);
    my_clReleaseKernel(s->kernel); my_clReleaseProgram(s->prog);
    my_clReleaseCommandQueue(s->q); my_clReleaseContext(s->ctx);
    free(A); free(B); free(C); free(ref);
    return rc;
}
