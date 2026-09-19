// Function: gemm_nt_kernel_kernel
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TL_N 2560
#define TL_K 2560
#define TL_BM 64
#define TL_BN 64
#define TL_BK 32
#define TL_WG 64
#define TL_NT 8

#define TL_BLOCK_F32(ACC, AS, BS, TM, TN)                                    \
    _Pragma("unroll 2")                                                     \
    for (int kk = 0; kk < TL_BK; kk++) {                                      \
        float8 a8 = convert_float8(*(const half8 *)&AS[kk][(TM) * 8]);        \
        float8 b8 = convert_float8(*(const half8 *)&BS[kk][(TN) * 8]);        \
        ACC[0] += (float8)a8.s0 * b8; ACC[1] += (float8)a8.s1 * b8;           \
        ACC[2] += (float8)a8.s2 * b8; ACC[3] += (float8)a8.s3 * b8;           \
        ACC[4] += (float8)a8.s4 * b8; ACC[5] += (float8)a8.s5 * b8;           \
        ACC[6] += (float8)a8.s6 * b8; ACC[7] += (float8)a8.s7 * b8;           \
    }

__kernel void gemm_nt_kernel_kernel(__global half *restrict A,
                                    __global half *restrict B,
                                    __global half *restrict C) {
    const int bm = get_group_id(1);
    const int bn = get_group_id(0);
    const int lid = get_local_id(0);
    const int tm = lid / TL_NT;
    const int tn = lid - tm * TL_NT;
    const int rbase = bm * TL_BM;
    const int cbase = bn * TL_BN;

    __local half As[TL_BK][TL_BM + 4];
    __local half Bs[TL_BK][TL_BN + 4];
    float8 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = (float8)(0.0f);

    for (int kb = 0; kb < TL_K; kb += TL_BK) {
        for (int v = lid; v < TL_BM * TL_BK / 4; v += TL_WG) {
            const int r = v / (TL_BK / 4);
            const int c4 = v - r * (TL_BK / 4);
            const half4 val = vload4(0, A + (size_t)(rbase + r) * TL_K + kb + c4 * 4);
            As[c4 * 4 + 0][r] = val.x;
            As[c4 * 4 + 1][r] = val.y;
            As[c4 * 4 + 2][r] = val.z;
            As[c4 * 4 + 3][r] = val.w;
        }
        for (int v = lid; v < TL_BN * TL_BK / 4; v += TL_WG) {
            const int c = v / (TL_BK / 4);
            const int k4 = v - c * (TL_BK / 4);
            const half4 val = vload4(0, B + (size_t)(cbase + c) * TL_K + kb + k4 * 4);
            Bs[k4 * 4 + 0][c] = val.x;
            Bs[k4 * 4 + 1][c] = val.y;
            Bs[k4 * 4 + 2][c] = val.z;
            Bs[k4 * 4 + 3][c] = val.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        TL_BLOCK_F32(acc, As, Bs, tm, tn);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int r0 = rbase + tm * 8;
    const int c0 = cbase + tn * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        vstore8(convert_half8(acc[i]), 0, C + (size_t)(r0 + i) * TL_N + c0);
}
