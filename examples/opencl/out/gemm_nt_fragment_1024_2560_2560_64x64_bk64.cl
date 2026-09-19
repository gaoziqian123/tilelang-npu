// Function: gemm_nt_kernel_kernel
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TL_M 1024
#define TL_N 2560
#define TL_K 2560

__kernel void gemm_nt_kernel_kernel(__global half *restrict A,
                                    __global half *restrict B,
                                    __global half *restrict C) {
  const int lid = convert_int(get_local_id(0));
  const int bx = convert_int(get_group_id(0));
  const int by = convert_int(get_group_id(1));
  const int tm = lid / 8;
  const int tn = lid - tm * 8;
  const int r0 = by * 64 + tm * 8;
  const int c0 = bx * 64 + tn * 8;
  float8 acc_0 = (float8)(0.0f);
  float8 acc_1 = (float8)(0.0f);
  float8 acc_2 = (float8)(0.0f);
  float8 acc_3 = (float8)(0.0f);
  float8 acc_4 = (float8)(0.0f);
  float8 acc_5 = (float8)(0.0f);
  float8 acc_6 = (float8)(0.0f);
  float8 acc_7 = (float8)(0.0f);
  for (int pos = 0; pos < 2560; pos += 4) {
    const float4 a0 = convert_float4(vload4(0, A + (size_t)(r0 + 0) * 2560 + pos));
    const float4 a1 = convert_float4(vload4(0, A + (size_t)(r0 + 1) * 2560 + pos));
    const float4 a2 = convert_float4(vload4(0, A + (size_t)(r0 + 2) * 2560 + pos));
    const float4 a3 = convert_float4(vload4(0, A + (size_t)(r0 + 3) * 2560 + pos));
    const float4 a4 = convert_float4(vload4(0, A + (size_t)(r0 + 4) * 2560 + pos));
    const float4 a5 = convert_float4(vload4(0, A + (size_t)(r0 + 5) * 2560 + pos));
    const float4 a6 = convert_float4(vload4(0, A + (size_t)(r0 + 6) * 2560 + pos));
    const float4 a7 = convert_float4(vload4(0, A + (size_t)(r0 + 7) * 2560 + pos));
    const float4 b0 = convert_float4(vload4(0, B + (size_t)(c0 + 0) * 2560 + pos));
    const float4 b1 = convert_float4(vload4(0, B + (size_t)(c0 + 1) * 2560 + pos));
    const float4 b2 = convert_float4(vload4(0, B + (size_t)(c0 + 2) * 2560 + pos));
    const float4 b3 = convert_float4(vload4(0, B + (size_t)(c0 + 3) * 2560 + pos));
    const float4 b4 = convert_float4(vload4(0, B + (size_t)(c0 + 4) * 2560 + pos));
    const float4 b5 = convert_float4(vload4(0, B + (size_t)(c0 + 5) * 2560 + pos));
    const float4 b6 = convert_float4(vload4(0, B + (size_t)(c0 + 6) * 2560 + pos));
    const float4 b7 = convert_float4(vload4(0, B + (size_t)(c0 + 7) * 2560 + pos));
    acc_0 += (float8)(a0.x) * (float8)(b0.x, b1.x, b2.x, b3.x, b4.x, b5.x, b6.x, b7.x);
    acc_0 += (float8)(a0.y) * (float8)(b0.y, b1.y, b2.y, b3.y, b4.y, b5.y, b6.y, b7.y);
    acc_0 += (float8)(a0.z) * (float8)(b0.z, b1.z, b2.z, b3.z, b4.z, b5.z, b6.z, b7.z);
    acc_0 += (float8)(a0.w) * (float8)(b0.w, b1.w, b2.w, b3.w, b4.w, b5.w, b6.w, b7.w);
    acc_1 += (float8)(a1.x) * (float8)(b0.x, b1.x, b2.x, b3.x, b4.x, b5.x, b6.x, b7.x);
    acc_1 += (float8)(a1.y) * (float8)(b0.y, b1.y, b2.y, b3.y, b4.y, b5.y, b6.y, b7.y);
    acc_1 += (float8)(a1.z) * (float8)(b0.z, b1.z, b2.z, b3.z, b4.z, b5.z, b6.z, b7.z);
    acc_1 += (float8)(a1.w) * (float8)(b0.w, b1.w, b2.w, b3.w, b4.w, b5.w, b6.w, b7.w);
    acc_2 += (float8)(a2.x) * (float8)(b0.x, b1.x, b2.x, b3.x, b4.x, b5.x, b6.x, b7.x);
    acc_2 += (float8)(a2.y) * (float8)(b0.y, b1.y, b2.y, b3.y, b4.y, b5.y, b6.y, b7.y);
    acc_2 += (float8)(a2.z) * (float8)(b0.z, b1.z, b2.z, b3.z, b4.z, b5.z, b6.z, b7.z);
    acc_2 += (float8)(a2.w) * (float8)(b0.w, b1.w, b2.w, b3.w, b4.w, b5.w, b6.w, b7.w);
    acc_3 += (float8)(a3.x) * (float8)(b0.x, b1.x, b2.x, b3.x, b4.x, b5.x, b6.x, b7.x);
    acc_3 += (float8)(a3.y) * (float8)(b0.y, b1.y, b2.y, b3.y, b4.y, b5.y, b6.y, b7.y);
    acc_3 += (float8)(a3.z) * (float8)(b0.z, b1.z, b2.z, b3.z, b4.z, b5.z, b6.z, b7.z);
    acc_3 += (float8)(a3.w) * (float8)(b0.w, b1.w, b2.w, b3.w, b4.w, b5.w, b6.w, b7.w);
    acc_4 += (float8)(a4.x) * (float8)(b0.x, b1.x, b2.x, b3.x, b4.x, b5.x, b6.x, b7.x);
    acc_4 += (float8)(a4.y) * (float8)(b0.y, b1.y, b2.y, b3.y, b4.y, b5.y, b6.y, b7.y);
    acc_4 += (float8)(a4.z) * (float8)(b0.z, b1.z, b2.z, b3.z, b4.z, b5.z, b6.z, b7.z);
    acc_4 += (float8)(a4.w) * (float8)(b0.w, b1.w, b2.w, b3.w, b4.w, b5.w, b6.w, b7.w);
    acc_5 += (float8)(a5.x) * (float8)(b0.x, b1.x, b2.x, b3.x, b4.x, b5.x, b6.x, b7.x);
    acc_5 += (float8)(a5.y) * (float8)(b0.y, b1.y, b2.y, b3.y, b4.y, b5.y, b6.y, b7.y);
    acc_5 += (float8)(a5.z) * (float8)(b0.z, b1.z, b2.z, b3.z, b4.z, b5.z, b6.z, b7.z);
    acc_5 += (float8)(a5.w) * (float8)(b0.w, b1.w, b2.w, b3.w, b4.w, b5.w, b6.w, b7.w);
    acc_6 += (float8)(a6.x) * (float8)(b0.x, b1.x, b2.x, b3.x, b4.x, b5.x, b6.x, b7.x);
    acc_6 += (float8)(a6.y) * (float8)(b0.y, b1.y, b2.y, b3.y, b4.y, b5.y, b6.y, b7.y);
    acc_6 += (float8)(a6.z) * (float8)(b0.z, b1.z, b2.z, b3.z, b4.z, b5.z, b6.z, b7.z);
    acc_6 += (float8)(a6.w) * (float8)(b0.w, b1.w, b2.w, b3.w, b4.w, b5.w, b6.w, b7.w);
    acc_7 += (float8)(a7.x) * (float8)(b0.x, b1.x, b2.x, b3.x, b4.x, b5.x, b6.x, b7.x);
    acc_7 += (float8)(a7.y) * (float8)(b0.y, b1.y, b2.y, b3.y, b4.y, b5.y, b6.y, b7.y);
    acc_7 += (float8)(a7.z) * (float8)(b0.z, b1.z, b2.z, b3.z, b4.z, b5.z, b6.z, b7.z);
    acc_7 += (float8)(a7.w) * (float8)(b0.w, b1.w, b2.w, b3.w, b4.w, b5.w, b6.w, b7.w);
  }
  vstore8(convert_half8(acc_0), 0, C + (size_t)(r0 + 0) * 2560 + c0);
  vstore8(convert_half8(acc_1), 0, C + (size_t)(r0 + 1) * 2560 + c0);
  vstore8(convert_half8(acc_2), 0, C + (size_t)(r0 + 2) * 2560 + c0);
  vstore8(convert_half8(acc_3), 0, C + (size_t)(r0 + 3) * 2560 + c0);
  vstore8(convert_half8(acc_4), 0, C + (size_t)(r0 + 4) * 2560 + c0);
  vstore8(convert_half8(acc_5), 0, C + (size_t)(r0 + 5) * 2560 + c0);
  vstore8(convert_half8(acc_6), 0, C + (size_t)(r0 + 6) * 2560 + c0);
  vstore8(convert_half8(acc_7), 0, C + (size_t)(r0 + 7) * 2560 + c0);
}
