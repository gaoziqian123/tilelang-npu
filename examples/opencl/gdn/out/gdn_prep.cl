// Function: gdn_prep_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

#define DK 128
#define CH 32
#define WG 128
__kernel void gdn_prep_kernel_kernel(__global half* restrict A2buf, __global float* restrict B, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global float* restrict G, __global half* restrict K, __global half* restrict KDbuf, __global half* restrict Q, __global half* restrict Ubuf, __global half* restrict V, __global half* restrict Wbuf);
__kernel void gdn_prep_kernel_kernel(__global half* restrict A2buf, __global float* restrict B, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global float* restrict G, __global half* restrict K, __global half* restrict KDbuf, __global half* restrict Q, __global half* restrict Ubuf, __global half* restrict V, __global half* restrict Wbuf) {
    int c = get_group_id(0), h = get_group_id(1), d = get_local_id(0);
    int hk = h & 15;
    size_t base_k = ((size_t)hk * 1024 + (size_t)c * CH) * DK;
    size_t base_v = ((size_t)h * 1024 + (size_t)c * CH) * DK;
    size_t gb = (size_t)h * 1024 + (size_t)c * CH;
    size_t cbase = ((size_t)h * 32 + c) * CH * DK;
    __local float gf[CH], bf[CH], gc[CH], egc[CH];
    __local float L[CH * CH];
    __local half UW[CH * DK];
    if (d < CH) { gf[d] = G[gb + d]; bf[d] = B[gb + d]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (d < CH) {
        float acc = 0.0f;
        for (int r = 0; r <= d; r++) acc += gf[r];
        gc[d] = acc; egc[d] = native_exp(acc);
        EgcBuf[((size_t)h * 32 + c) * CH + d] = egc[d];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float glast = gc[CH - 1];
    EglBuf[(size_t)h * 32 + c] = native_exp(glast);
    for (int t = d; t < CH * CH; t += WG) {
        int i = t / CH, j = t & 31;
        float lv = 0.0f, av = 0.0f;
        if (j <= i) {
            float decay = native_exp(gc[i] - gc[j]);
            half8 acc8 = (half8)(0.0h), qacc = (half8)(0.0h);
            #pragma unroll
            for (int k = 0; k < DK; k += 8) {
                half8 kj = vload8(0, K + base_k + (size_t)j * DK + k);
                acc8 += vload8(0, K + base_k + (size_t)i * DK + k) * kj;
                qacc += vload8(0, Q + base_k + (size_t)i * DK + k) * kj;
            }
            float kk = (float)acc8.s0 + acc8.s1 + acc8.s2 + acc8.s3 + acc8.s4 + acc8.s5 + acc8.s6 + acc8.s7;
            if (j < i) lv = bf[i] * kk * decay;
            av = ((float)qacc.s0 + qacc.s1 + qacc.s2 + qacc.s3 + qacc.s4 + qacc.s5 + qacc.s6 + qacc.s7) * decay;
        }
        L[t] = lv;
        A2buf[((size_t)h * 32 + c) * CH * CH + t] = (half)av;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < CH; i++) {
        float au = bf[i] * (float)V[base_v + (size_t)i * DK + d];
        for (int m = 0; m < i; m++) au -= L[i * CH + m] * (float)UW[m * DK + d];
        UW[i * DK + d] = (half)au;
    }
    #pragma unroll
    for (int r = 0; r < CH; r++) Ubuf[cbase + (size_t)r * DK + d] = UW[r * DK + d];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < CH; i++) {
        float aw = bf[i] * (float)K[base_k + (size_t)i * DK + d] * egc[i];
        for (int m = 0; m < i; m++) aw -= L[i * CH + m] * (float)UW[m * DK + d];
        UW[i * DK + d] = (half)aw;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int r = 0; r < CH; r++) {
        Wbuf[cbase + (size_t)r * DK + d] = UW[r * DK + d];
        KDbuf[cbase + (size_t)r * DK + d] = (half)((float)K[base_k + (size_t)r * DK + d] * native_exp(glast - gc[r]));
    }
}
