// Function: gdn_seq_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

#define DK 128
#define DV 128
#define CH 32
#define WG 128
__kernel void gdn_seq_kernel_kernel(__global half* restrict A2buf, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global half* restrict KDbuf, __global half* restrict O, __global half* restrict Q, __global float* restrict S, __global half* restrict Ubuf, __global half* restrict Wbuf);
__kernel void gdn_seq_kernel_kernel(__global half* restrict A2buf, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global half* restrict KDbuf, __global half* restrict O, __global half* restrict Q, __global float* restrict S, __global half* restrict Ubuf, __global half* restrict Wbuf) {
    int dvb = get_group_id(0), h = get_group_id(1), d = get_local_id(0);
    int hk = h & 15;
    __local half kdl[CH * DK];
    __local half vn[CH * 32];
    __global float *cur = S + (size_t)h * DK * DV;
    for (int c = 0; c < 32; c++) {
        size_t cb = ((size_t)h * 32 + c) * CH * DK;
        for (int i = d; i < CH * DK; i += WG) kdl[i] = KDbuf[cb + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        int t = d / 4;
        int dvg = d & 3;
        int dv0 = dvb * 32 + dvg * 8;
        float8 acc8 = (float8)(0.0f), acco = (float8)(0.0f);
        size_t qb = ((size_t)hk * 1024 + (size_t)c * CH) * DK + (size_t)t * DK;
        for (int dk = 0; dk < DK; dk++) {
            float8 s8 = vload8(0, cur + (size_t)dk * DV + dv0);
            float w = (float)Wbuf[cb + (size_t)t * DK + dk];
            float qv = (float)Q[qb + dk];
            acc8 += w * s8;
            acco += qv * s8;
        }
        float8 u8 = convert_float8(vload8(0, Ubuf + cb + (size_t)t * DK + dv0));
        vstore8(convert_half8(u8 - acc8), 0, vn + t * 32 + dvg * 8);
        float egl = EglBuf[(size_t)h * 32 + c];
        barrier(CLK_LOCAL_MEM_FENCE);
        float8 intra8 = (float8)(0.0f);
        __global half *a2 = A2buf + ((size_t)h * 32 + c) * CH * CH + t * CH;
        for (int j = 0; j <= t; j++) intra8 += (float8)a2[j] * convert_float8(vload8(0, vn + j * 32 + dvg * 8));
        float egc = EgcBuf[((size_t)h * 32 + c) * CH + t];
        size_t ob = ((size_t)h * 1024 + (size_t)c * CH) * DV + (size_t)t * DV + dv0;
        vstore8(convert_half8(egc * acco + intra8), 0, O + ob);
        barrier(CLK_LOCAL_MEM_FENCE);
        int dk8 = d / 32, dvl = d & 31, dv = dvb * 32 + dvl;
        #pragma unroll
        for (int sb = 0; sb < 4; sb++) {
            int dk0 = sb * 32 + dk8 * 8;
            half8 m8 = (half8)(0.0h);
            #pragma unroll
            for (int t2 = 0; t2 < CH; t2++) m8 += vload8(0, kdl + t2 * DK + dk0) * (half8)vn[t2 * 32 + dvl];
            float8 mf = convert_float8(m8);
            cur[(size_t)(dk0 + 0) * DV + dv] = egl * cur[(size_t)(dk0 + 0) * DV + dv] + mf.s0;
            cur[(size_t)(dk0 + 1) * DV + dv] = egl * cur[(size_t)(dk0 + 1) * DV + dv] + mf.s1;
            cur[(size_t)(dk0 + 2) * DV + dv] = egl * cur[(size_t)(dk0 + 2) * DV + dv] + mf.s2;
            cur[(size_t)(dk0 + 3) * DV + dv] = egl * cur[(size_t)(dk0 + 3) * DV + dv] + mf.s3;
            cur[(size_t)(dk0 + 4) * DV + dv] = egl * cur[(size_t)(dk0 + 4) * DV + dv] + mf.s4;
            cur[(size_t)(dk0 + 5) * DV + dv] = egl * cur[(size_t)(dk0 + 5) * DV + dv] + mf.s5;
            cur[(size_t)(dk0 + 6) * DV + dv] = egl * cur[(size_t)(dk0 + 6) * DV + dv] + mf.s6;
            cur[(size_t)(dk0 + 7) * DV + dv] = egl * cur[(size_t)(dk0 + 7) * DV + dv] + mf.s7;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
