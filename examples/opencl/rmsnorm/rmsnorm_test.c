// rmsnorm_test.c - host validation for the TileLang-emitted rmsnorm OpenCL kernel.
//
// Verifies O[r,c] = A[r,c] / sqrt(mean_c(A[r,..]^2) + eps) against an fp64
// reference with the rms-scaled relative error gate |got-ref|/(|ref|+0.02*rms).
//
// Build (Android, OpenCL-Headers at backend/gpu/OpenCL-Headers):
//   $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang \
//     -target aarch64-linux-android26 -std=c11 -O3 -fPIE -pie \
//     -I/root/project/backend/gpu/OpenCL-Headers rmsnorm_test.c -ldl -lm -o rmsnorm_test
// Run (device):
//   LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ./rmsnorm_test out/rmsnorm.cl rmsnorm_kernel_kernel
//
// Env: TL_ROWS (960) TL_COLS (2560) TL_THREADS (256)

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

static int env_i(const char *name, int def) {
    const char *s = getenv(name);
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    return (end && *end == 0 && v > 0) ? (int)v : def;
}

static double rms_from_refs(const double *ref, size_t n) {
    long double ss = 0.0;
    for (size_t i = 0; i < n; ++i) ss += (long double)ref[i] * (long double)ref[i];
    return sqrt((double)(ss / (long double)n));
}

static int check_f32(const char *tag, const float *got, const double *ref, size_t n) {
    double rms = rms_from_refs(ref, n);
    double max_rel = 0.0;
    size_t max_i = 0, bad = 0;
    for (size_t i = 0; i < n; ++i) {
        double r = fabs((double)got[i] - ref[i]) / (fabs(ref[i]) + 0.02 * rms);
        if (r > max_rel) { max_rel = r; max_i = i; }
        if (r > 0.1) ++bad;
    }
    printf("%s max_rel %.6g rms %.6g bad %zu/%zu sample[%zu] got %.9g ref %.9g %s\n",
           tag, max_rel, rms, bad, n, max_i, (double)got[max_i], ref[max_i], max_rel < 0.1 ? "PASS" : "FAIL");
    return max_rel < 0.1 ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *cl_path = argc > 1 ? argv[1] : "out/rmsnorm.cl";
    const char *kname = argc > 2 ? argv[2] : "rmsnorm_kernel_kernel";
    const int R = env_i("TL_ROWS", 960), C = env_i("TL_COLS", 2560);
    const int THREADS = env_i("TL_THREADS", 256);
    const size_t n = (size_t)R * C;
    const float eps = 1.0e-5f;

    load_cl();
    cl_platform_id pf; cl_uint np = 0;
    CK(my_clGetPlatformIDs(1, &pf, &np));
    cl_device_id dev; cl_uint nd = 0;
    CK(my_clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &dev, &nd));
    cl_int err;
    cl_context ctx = my_clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CK(err);
    const cl_queue_properties qprops[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = my_clCreateCommandQueueWithProperties(ctx, dev, qprops, &err); CK(err);
    size_t src_sz = 0;
    char *src = read_file(cl_path, &src_sz);
    const char *srcp = src;
    cl_program prog = my_clCreateProgramWithSource(ctx, 1, &srcp, &src_sz, &err); CK(err);
    err = my_clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
    size_t logn = 0;
    my_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &logn);
    if (logn > 1) {
        char *log = (char *)calloc(logn + 1, 1);
        if (log) { my_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logn, log, NULL); printf("build_log:\n%s\n", log); free(log); }
    }
    if (err != CL_SUCCESS) exit(3);
    cl_kernel kernel = my_clCreateKernel(prog, kname, &err); CK(err);
    free(src);

    uint32_t seed = 1;
    float *A = (float *)malloc(n * sizeof(float));
    float *O = (float *)calloc(n, sizeof(float));
    double *ref = (double *)malloc(n * sizeof(double));
    if (!A || !O || !ref) { fprintf(stderr, "rmsnorm alloc failed\n"); exit(2); }
    for (size_t i = 0; i < n; ++i) A[i] = rand_f32(&seed);
    for (int r = 0; r < R; ++r) {
        double ss = 0.0;
        for (int c = 0; c < C; ++c) { double x = (double)A[(size_t)r * C + c]; ss += x * x; }
        double scale = 1.0 / sqrt(ss / (double)C + (double)eps);
        for (int c = 0; c < C; ++c) ref[(size_t)r * C + c] = (double)A[(size_t)r * C + c] * scale;
    }

    cl_mem bA = my_clCreateBuffer(ctx, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &err); CK(err);
    cl_mem bO = my_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &err); CK(err);
    CK(my_clEnqueueWriteBuffer(q, bA, CL_TRUE, 0, n * sizeof(float), A, 0, NULL, NULL));
    CK(my_clSetKernelArg(kernel, 0, sizeof(bA), &bA));
    CK(my_clSetKernelArg(kernel, 1, sizeof(bO), &bO));
    CK(my_clSetKernelArg(kernel, 2, sizeof(C), &C));
    CK(my_clSetKernelArg(kernel, 3, sizeof(eps), &eps));
    const size_t local[1] = {(size_t)THREADS}, global[1] = {(size_t)R * (size_t)THREADS};
    double t0 = now_s();
    CK(my_clEnqueueNDRangeKernel(q, kernel, 1, NULL, global, local, 0, NULL, NULL));
    CK(my_clFinish(q));
    double t1 = now_s();
    CK(my_clEnqueueReadBuffer(q, bO, CL_TRUE, 0, n * sizeof(float), O, 0, NULL, NULL));
    printf("RMSNORM R=%d C=%d TH=%d ran %.3f ms\n", R, C, THREADS, (t1 - t0) * 1e3);
    int rc = check_f32("RMSNORM", O, ref, n);

    my_clReleaseMemObject(bA); my_clReleaseMemObject(bO);
    my_clReleaseKernel(kernel); my_clReleaseProgram(prog);
    my_clReleaseCommandQueue(q); my_clReleaseContext(ctx);
    free(A); free(O); free(ref);
    return rc;
}
