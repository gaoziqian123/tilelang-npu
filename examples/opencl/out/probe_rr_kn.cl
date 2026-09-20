// Function: gemm_nt_kernel_kernel
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TL_M 512
#define TL_N 512
#define TL_K 512
#define TL_BM 32
#define TL_BN 128
#define TL_BK 64
#define TL_WG 64
#define TL_NT 16

__kernel void gemm_nt_kernel_kernel(__global const half *restrict A,
                                    __global const half *restrict B,
                                    __global half *restrict C) {
    const int bx = get_group_id(0);
    const int by = get_group_id(1);
    const int tx = get_local_id(0);
    const int tm = tx / TL_NT;
    const int tn = tx - tm * TL_NT;
    const int r0 = by * TL_BM + tm * 8;
    const int c0 = bx * TL_BN + tn * 8;
    float8 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = (float8)(0.0f);
    for (int kb = 0; kb < TL_K; kb += TL_BK) {
        for (int pos = kb; pos < kb + TL_BK; pos += 4) {
            float4 a[8];
            float8 b[4];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                a[i] = convert_float4(vload4(0, A + (size_t)(r0 + i) * TL_K + pos));
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            b[i] = convert_float8(vload8(0, B + (size_t)(pos + i) * TL_N + c0));
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
            acc[i] += (float8)(a[i].x) * b[0];
            acc[i] += (float8)(a[i].y) * b[1];
            acc[i] += (float8)(a[i].z) * b[2];
            acc[i] += (float8)(a[i].w) * b[3];
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        if (r0 + i < TL_M && c0 + 7 < TL_N)
            vstore8(convert_half8(acc[i]), 0, C + (size_t)(r0 + i) * TL_N + c0);
}
