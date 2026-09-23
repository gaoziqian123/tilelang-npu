// Function: gemm_nt_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void gemm_nt_kernel_kernel(__global half* restrict A, __global half* restrict B, __global half* restrict C);
__kernel void gemm_nt_kernel_kernel(__global half* restrict A, __global half* restrict B, __global half* restrict C) { const int tl_gid0 = convert_int(get_group_id(0)); const int tl_gid1 = convert_int(get_group_id(1)); const int tl_lid0 = convert_int(get_local_id(0)); const int tl_wi_aff0 = ((tl_gid1 * 81920) + ((tl_lid0 >> 3) * 20480)); const int tl_wi_aff1 = ((tl_wi_aff0 + (tl_gid0 * 64)) + ((tl_lid0 & 7) * 8)); const int tl_wi_aff2 = ((tl_lid0 >> 3) * 163840); const int tl_wi_aff3 = (tl_gid0 * 163840); const int tl_wi_aff4 = (tl_lid0 * 20480); __global half* tl_C_base = C + tl_wi_aff1;
  float8 acc_1_0;
  float8 acc_1_1;
  float8 acc_1_2;
  float8 acc_1_3;
  float8 acc_1_4;
  float8 acc_1_5;
  float8 acc_1_6;
  float8 acc_1_7;
  float broadcast_var = 0.000000e+00f;
  acc_1_0 = ((float8)(broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var));
  float broadcast_var_1 = 0.000000e+00f;
  acc_1_1 = ((float8)(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1));
  float broadcast_var_2 = 0.000000e+00f;
  acc_1_2 = ((float8)(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2));
  float broadcast_var_3 = 0.000000e+00f;
  acc_1_3 = ((float8)(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3));
  float broadcast_var_4 = 0.000000e+00f;
  acc_1_4 = ((float8)(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4));
  float broadcast_var_5 = 0.000000e+00f;
  acc_1_5 = ((float8)(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5));
  float broadcast_var_6 = 0.000000e+00f;
  acc_1_6 = ((float8)(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6));
  float broadcast_var_7 = 0.000000e+00f;
  acc_1_7 = ((float8)(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7));
  for (int pos4 = 0; pos4 < 640; ++pos4) { const int tl_loop_aff0 = (tl_wi_aff0 + (pos4 * 4)); const int tl_loop_aff1 = (pos4 * 4); __global half* tl_Ap = A + tl_loop_aff0; __global half* tl_Bp = B + (tl_loop_aff1 - tl_wi_aff2);
    float4 av_1[8];
    float4 bg_1[8];
    av_1[0] = (convert_float4(vload4(0, tl_Ap)));
    av_1[1] = (convert_float4(vload4(0, tl_Ap + ( 2560))));
    av_1[2] = (convert_float4(vload4(0, tl_Ap + ( 5120))));
    av_1[3] = (convert_float4(vload4(0, tl_Ap + ( 7680))));
    av_1[4] = (convert_float4(vload4(0, tl_Ap + ( 10240))));
    av_1[5] = (convert_float4(vload4(0, tl_Ap + ( 12800))));
    av_1[6] = (convert_float4(vload4(0, tl_Ap + ( 15360))));
    av_1[7] = (convert_float4(vload4(0, tl_Ap + ( 17920))));
    bg_1[0] = (convert_float4(vload4(0, B + (((tl_wi_aff3 + tl_wi_aff4) + tl_loop_aff1) - tl_wi_aff2))));
    bg_1[1] = (convert_float4(vload4(0, B + ((((tl_wi_aff3 + tl_wi_aff4) + tl_loop_aff1) + 2560) - tl_wi_aff2))));
    bg_1[2] = (convert_float4(vload4(0, B + ((((tl_wi_aff3 + tl_wi_aff4) + tl_loop_aff1) + 5120) - tl_wi_aff2))));
    bg_1[3] = (convert_float4(vload4(0, B + ((((tl_wi_aff3 + tl_wi_aff4) + tl_loop_aff1) + 7680) - tl_wi_aff2))));
    bg_1[4] = (convert_float4(vload4(0, B + ((((tl_wi_aff3 + tl_wi_aff4) + tl_loop_aff1) + 10240) - tl_wi_aff2))));
    bg_1[5] = (convert_float4(vload4(0, B + ((((tl_wi_aff3 + tl_wi_aff4) + tl_loop_aff1) + 12800) - tl_wi_aff2))));
    bg_1[6] = (convert_float4(vload4(0, B + ((((tl_wi_aff3 + tl_wi_aff4) + tl_loop_aff1) + 15360) - tl_wi_aff2))));
    bg_1[7] = (convert_float4(vload4(0, B + ((((tl_wi_aff3 + tl_wi_aff4) + tl_loop_aff1) + 17920) - tl_wi_aff2))));
    acc_1_0 = mad(((float8)((av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0)), (float8)((bg_1[0]).s0, (bg_1[1]).s0, (bg_1[2]).s0, (bg_1[3]).s0, (bg_1[4]).s0, (bg_1[5]).s0, (bg_1[6]).s0, (bg_1[7]).s0), mad(((float8)((av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1)), (float8)((bg_1[0]).s1, (bg_1[1]).s1, (bg_1[2]).s1, (bg_1[3]).s1, (bg_1[4]).s1, (bg_1[5]).s1, (bg_1[6]).s1, (bg_1[7]).s1), mad(((float8)((av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2)), (float8)((bg_1[0]).s2, (bg_1[1]).s2, (bg_1[2]).s2, (bg_1[3]).s2, (bg_1[4]).s2, (bg_1[5]).s2, (bg_1[6]).s2, (bg_1[7]).s2), mad(((float8)((av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3)), (float8)((bg_1[0]).s3, (bg_1[1]).s3, (bg_1[2]).s3, (bg_1[3]).s3, (bg_1[4]).s3, (bg_1[5]).s3, (bg_1[6]).s3, (bg_1[7]).s3), acc_1_0))));
    acc_1_1 = mad(((float8)((av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0)), (float8)((bg_1[0]).s0, (bg_1[1]).s0, (bg_1[2]).s0, (bg_1[3]).s0, (bg_1[4]).s0, (bg_1[5]).s0, (bg_1[6]).s0, (bg_1[7]).s0), mad(((float8)((av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1)), (float8)((bg_1[0]).s1, (bg_1[1]).s1, (bg_1[2]).s1, (bg_1[3]).s1, (bg_1[4]).s1, (bg_1[5]).s1, (bg_1[6]).s1, (bg_1[7]).s1), mad(((float8)((av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2)), (float8)((bg_1[0]).s2, (bg_1[1]).s2, (bg_1[2]).s2, (bg_1[3]).s2, (bg_1[4]).s2, (bg_1[5]).s2, (bg_1[6]).s2, (bg_1[7]).s2), mad(((float8)((av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3)), (float8)((bg_1[0]).s3, (bg_1[1]).s3, (bg_1[2]).s3, (bg_1[3]).s3, (bg_1[4]).s3, (bg_1[5]).s3, (bg_1[6]).s3, (bg_1[7]).s3), acc_1_1))));
    acc_1_2 = mad(((float8)((av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0)), (float8)((bg_1[0]).s0, (bg_1[1]).s0, (bg_1[2]).s0, (bg_1[3]).s0, (bg_1[4]).s0, (bg_1[5]).s0, (bg_1[6]).s0, (bg_1[7]).s0), mad(((float8)((av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1)), (float8)((bg_1[0]).s1, (bg_1[1]).s1, (bg_1[2]).s1, (bg_1[3]).s1, (bg_1[4]).s1, (bg_1[5]).s1, (bg_1[6]).s1, (bg_1[7]).s1), mad(((float8)((av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2)), (float8)((bg_1[0]).s2, (bg_1[1]).s2, (bg_1[2]).s2, (bg_1[3]).s2, (bg_1[4]).s2, (bg_1[5]).s2, (bg_1[6]).s2, (bg_1[7]).s2), mad(((float8)((av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3)), (float8)((bg_1[0]).s3, (bg_1[1]).s3, (bg_1[2]).s3, (bg_1[3]).s3, (bg_1[4]).s3, (bg_1[5]).s3, (bg_1[6]).s3, (bg_1[7]).s3), acc_1_2))));
    acc_1_3 = mad(((float8)((av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0)), (float8)((bg_1[0]).s0, (bg_1[1]).s0, (bg_1[2]).s0, (bg_1[3]).s0, (bg_1[4]).s0, (bg_1[5]).s0, (bg_1[6]).s0, (bg_1[7]).s0), mad(((float8)((av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1)), (float8)((bg_1[0]).s1, (bg_1[1]).s1, (bg_1[2]).s1, (bg_1[3]).s1, (bg_1[4]).s1, (bg_1[5]).s1, (bg_1[6]).s1, (bg_1[7]).s1), mad(((float8)((av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2)), (float8)((bg_1[0]).s2, (bg_1[1]).s2, (bg_1[2]).s2, (bg_1[3]).s2, (bg_1[4]).s2, (bg_1[5]).s2, (bg_1[6]).s2, (bg_1[7]).s2), mad(((float8)((av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3)), (float8)((bg_1[0]).s3, (bg_1[1]).s3, (bg_1[2]).s3, (bg_1[3]).s3, (bg_1[4]).s3, (bg_1[5]).s3, (bg_1[6]).s3, (bg_1[7]).s3), acc_1_3))));
    acc_1_4 = mad(((float8)((av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0)), (float8)((bg_1[0]).s0, (bg_1[1]).s0, (bg_1[2]).s0, (bg_1[3]).s0, (bg_1[4]).s0, (bg_1[5]).s0, (bg_1[6]).s0, (bg_1[7]).s0), mad(((float8)((av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1)), (float8)((bg_1[0]).s1, (bg_1[1]).s1, (bg_1[2]).s1, (bg_1[3]).s1, (bg_1[4]).s1, (bg_1[5]).s1, (bg_1[6]).s1, (bg_1[7]).s1), mad(((float8)((av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2)), (float8)((bg_1[0]).s2, (bg_1[1]).s2, (bg_1[2]).s2, (bg_1[3]).s2, (bg_1[4]).s2, (bg_1[5]).s2, (bg_1[6]).s2, (bg_1[7]).s2), mad(((float8)((av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3)), (float8)((bg_1[0]).s3, (bg_1[1]).s3, (bg_1[2]).s3, (bg_1[3]).s3, (bg_1[4]).s3, (bg_1[5]).s3, (bg_1[6]).s3, (bg_1[7]).s3), acc_1_4))));
    acc_1_5 = mad(((float8)((av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0)), (float8)((bg_1[0]).s0, (bg_1[1]).s0, (bg_1[2]).s0, (bg_1[3]).s0, (bg_1[4]).s0, (bg_1[5]).s0, (bg_1[6]).s0, (bg_1[7]).s0), mad(((float8)((av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1)), (float8)((bg_1[0]).s1, (bg_1[1]).s1, (bg_1[2]).s1, (bg_1[3]).s1, (bg_1[4]).s1, (bg_1[5]).s1, (bg_1[6]).s1, (bg_1[7]).s1), mad(((float8)((av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2)), (float8)((bg_1[0]).s2, (bg_1[1]).s2, (bg_1[2]).s2, (bg_1[3]).s2, (bg_1[4]).s2, (bg_1[5]).s2, (bg_1[6]).s2, (bg_1[7]).s2), mad(((float8)((av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3)), (float8)((bg_1[0]).s3, (bg_1[1]).s3, (bg_1[2]).s3, (bg_1[3]).s3, (bg_1[4]).s3, (bg_1[5]).s3, (bg_1[6]).s3, (bg_1[7]).s3), acc_1_5))));
    acc_1_6 = mad(((float8)((av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0)), (float8)((bg_1[0]).s0, (bg_1[1]).s0, (bg_1[2]).s0, (bg_1[3]).s0, (bg_1[4]).s0, (bg_1[5]).s0, (bg_1[6]).s0, (bg_1[7]).s0), mad(((float8)((av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1)), (float8)((bg_1[0]).s1, (bg_1[1]).s1, (bg_1[2]).s1, (bg_1[3]).s1, (bg_1[4]).s1, (bg_1[5]).s1, (bg_1[6]).s1, (bg_1[7]).s1), mad(((float8)((av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2)), (float8)((bg_1[0]).s2, (bg_1[1]).s2, (bg_1[2]).s2, (bg_1[3]).s2, (bg_1[4]).s2, (bg_1[5]).s2, (bg_1[6]).s2, (bg_1[7]).s2), mad(((float8)((av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3)), (float8)((bg_1[0]).s3, (bg_1[1]).s3, (bg_1[2]).s3, (bg_1[3]).s3, (bg_1[4]).s3, (bg_1[5]).s3, (bg_1[6]).s3, (bg_1[7]).s3), acc_1_6))));
    acc_1_7 = mad(((float8)((av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0)), (float8)((bg_1[0]).s0, (bg_1[1]).s0, (bg_1[2]).s0, (bg_1[3]).s0, (bg_1[4]).s0, (bg_1[5]).s0, (bg_1[6]).s0, (bg_1[7]).s0), mad(((float8)((av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1)), (float8)((bg_1[0]).s1, (bg_1[1]).s1, (bg_1[2]).s1, (bg_1[3]).s1, (bg_1[4]).s1, (bg_1[5]).s1, (bg_1[6]).s1, (bg_1[7]).s1), mad(((float8)((av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2)), (float8)((bg_1[0]).s2, (bg_1[1]).s2, (bg_1[2]).s2, (bg_1[3]).s2, (bg_1[4]).s2, (bg_1[5]).s2, (bg_1[6]).s2, (bg_1[7]).s2), mad(((float8)((av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3)), (float8)((bg_1[0]).s3, (bg_1[1]).s3, (bg_1[2]).s3, (bg_1[3]).s3, (bg_1[4]).s3, (bg_1[5]).s3, (bg_1[6]).s3, (bg_1[7]).s3), acc_1_7))));
  }
  vstore8(convert_half8(acc_1_0), 0, C + tl_wi_aff1);

  vstore8(convert_half8(acc_1_1), 0, tl_C_base + ( + 2560));

  vstore8(convert_half8(acc_1_2), 0, tl_C_base + ( + 5120));

  vstore8(convert_half8(acc_1_3), 0, tl_C_base + ( + 7680));

  vstore8(convert_half8(acc_1_4), 0, tl_C_base + ( + 10240));

  vstore8(convert_half8(acc_1_5), 0, tl_C_base + ( + 12800));

  vstore8(convert_half8(acc_1_6), 0, tl_C_base + ( + 15360));

  vstore8(convert_half8(acc_1_7), 0, tl_C_base + ( + 17920));

}

