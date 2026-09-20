// Function: gemm_nt_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void gemm_nt_kernel_kernel(__global half* restrict A_1, __global half* restrict B_1, __global half* restrict C_1);
__kernel void gemm_nt_kernel_kernel(__global half* restrict A_1, __global half* restrict B_1, __global half* restrict C_1) {
  float8 acc_0;
  float8 acc_1;
  float8 acc_2;
  float8 acc_3;
  float8 acc_4;
  float8 acc_5;
  float8 acc_6;
  float8 acc_7;
  int tm = ((convert_int(get_local_id(0))) >> 4);
  int tn = ((convert_int(get_local_id(0))) - (((convert_int(get_local_id(0))) >> 4) * 16));
  int r0 = (((convert_int(get_group_id(1))) * 32) + (((convert_int(get_local_id(0))) >> 4) * 8));
  int c0 = ((((convert_int(get_group_id(0))) * 128) + ((convert_int(get_local_id(0))) * 8)) - (((convert_int(get_local_id(0))) >> 4) * 128));
  float broadcast_var = 0.000000e+00f;
  acc_0 = ((float8)(broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var));
  float broadcast_var_1 = 0.000000e+00f;
  acc_1 = ((float8)(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1));
  float broadcast_var_2 = 0.000000e+00f;
  acc_2 = ((float8)(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2));
  float broadcast_var_3 = 0.000000e+00f;
  acc_3 = ((float8)(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3));
  float broadcast_var_4 = 0.000000e+00f;
  acc_4 = ((float8)(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4));
  float broadcast_var_5 = 0.000000e+00f;
  acc_5 = ((float8)(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5));
  float broadcast_var_6 = 0.000000e+00f;
  acc_6 = ((float8)(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6));
  float broadcast_var_7 = 0.000000e+00f;
  acc_7 = ((float8)(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7));
  for (int pos4 = 0; pos4 < 640; ++pos4) {
    float4 a0 = (convert_float4(vload4(0, A_1 + ((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20480)) + (pos4 * 4)))));
    float4 a1 = (convert_float4(vload4(0, A_1 + (((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20480)) + (pos4 * 4)) + 2560))));
    float4 a2 = (convert_float4(vload4(0, A_1 + (((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20480)) + (pos4 * 4)) + 5120))));
    float4 a3 = (convert_float4(vload4(0, A_1 + (((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20480)) + (pos4 * 4)) + 7680))));
    float4 a4 = (convert_float4(vload4(0, A_1 + (((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20480)) + (pos4 * 4)) + 10240))));
    float4 a5 = (convert_float4(vload4(0, A_1 + (((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20480)) + (pos4 * 4)) + 12800))));
    float4 a6 = (convert_float4(vload4(0, A_1 + (((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20480)) + (pos4 * 4)) + 15360))));
    float4 a7 = (convert_float4(vload4(0, A_1 + (((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20480)) + (pos4 * 4)) + 17920))));
    float8 b0 = (convert_float8(vload8(0, B_1 + ((((pos4 * 10240) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) - (((convert_int(get_local_id(0))) >> 4) * 128)))));
    float8 b1 = (convert_float8(vload8(0, B_1 + (((((pos4 * 10240) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 2560) - (((convert_int(get_local_id(0))) >> 4) * 128)))));
    float8 b2 = (convert_float8(vload8(0, B_1 + (((((pos4 * 10240) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 5120) - (((convert_int(get_local_id(0))) >> 4) * 128)))));
    float8 b3 = (convert_float8(vload8(0, B_1 + (((((pos4 * 10240) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 7680) - (((convert_int(get_local_id(0))) >> 4) * 128)))));
    acc_0 = (acc_0 + (((float8)((a0).s0, (a0).s0, (a0).s0, (a0).s0, (a0).s0, (a0).s0, (a0).s0, (a0).s0)) * b0));
    acc_0 = (acc_0 + (((float8)((a0).s1, (a0).s1, (a0).s1, (a0).s1, (a0).s1, (a0).s1, (a0).s1, (a0).s1)) * b1));
    acc_0 = (acc_0 + (((float8)((a0).s2, (a0).s2, (a0).s2, (a0).s2, (a0).s2, (a0).s2, (a0).s2, (a0).s2)) * b2));
    acc_0 = (acc_0 + (((float8)((a0).s3, (a0).s3, (a0).s3, (a0).s3, (a0).s3, (a0).s3, (a0).s3, (a0).s3)) * b3));
    acc_1 = (acc_1 + (((float8)((a1).s0, (a1).s0, (a1).s0, (a1).s0, (a1).s0, (a1).s0, (a1).s0, (a1).s0)) * b0));
    acc_1 = (acc_1 + (((float8)((a1).s1, (a1).s1, (a1).s1, (a1).s1, (a1).s1, (a1).s1, (a1).s1, (a1).s1)) * b1));
    acc_1 = (acc_1 + (((float8)((a1).s2, (a1).s2, (a1).s2, (a1).s2, (a1).s2, (a1).s2, (a1).s2, (a1).s2)) * b2));
    acc_1 = (acc_1 + (((float8)((a1).s3, (a1).s3, (a1).s3, (a1).s3, (a1).s3, (a1).s3, (a1).s3, (a1).s3)) * b3));
    acc_2 = (acc_2 + (((float8)((a2).s0, (a2).s0, (a2).s0, (a2).s0, (a2).s0, (a2).s0, (a2).s0, (a2).s0)) * b0));
    acc_2 = (acc_2 + (((float8)((a2).s1, (a2).s1, (a2).s1, (a2).s1, (a2).s1, (a2).s1, (a2).s1, (a2).s1)) * b1));
    acc_2 = (acc_2 + (((float8)((a2).s2, (a2).s2, (a2).s2, (a2).s2, (a2).s2, (a2).s2, (a2).s2, (a2).s2)) * b2));
    acc_2 = (acc_2 + (((float8)((a2).s3, (a2).s3, (a2).s3, (a2).s3, (a2).s3, (a2).s3, (a2).s3, (a2).s3)) * b3));
    acc_3 = (acc_3 + (((float8)((a3).s0, (a3).s0, (a3).s0, (a3).s0, (a3).s0, (a3).s0, (a3).s0, (a3).s0)) * b0));
    acc_3 = (acc_3 + (((float8)((a3).s1, (a3).s1, (a3).s1, (a3).s1, (a3).s1, (a3).s1, (a3).s1, (a3).s1)) * b1));
    acc_3 = (acc_3 + (((float8)((a3).s2, (a3).s2, (a3).s2, (a3).s2, (a3).s2, (a3).s2, (a3).s2, (a3).s2)) * b2));
    acc_3 = (acc_3 + (((float8)((a3).s3, (a3).s3, (a3).s3, (a3).s3, (a3).s3, (a3).s3, (a3).s3, (a3).s3)) * b3));
    acc_4 = (acc_4 + (((float8)((a4).s0, (a4).s0, (a4).s0, (a4).s0, (a4).s0, (a4).s0, (a4).s0, (a4).s0)) * b0));
    acc_4 = (acc_4 + (((float8)((a4).s1, (a4).s1, (a4).s1, (a4).s1, (a4).s1, (a4).s1, (a4).s1, (a4).s1)) * b1));
    acc_4 = (acc_4 + (((float8)((a4).s2, (a4).s2, (a4).s2, (a4).s2, (a4).s2, (a4).s2, (a4).s2, (a4).s2)) * b2));
    acc_4 = (acc_4 + (((float8)((a4).s3, (a4).s3, (a4).s3, (a4).s3, (a4).s3, (a4).s3, (a4).s3, (a4).s3)) * b3));
    acc_5 = (acc_5 + (((float8)((a5).s0, (a5).s0, (a5).s0, (a5).s0, (a5).s0, (a5).s0, (a5).s0, (a5).s0)) * b0));
    acc_5 = (acc_5 + (((float8)((a5).s1, (a5).s1, (a5).s1, (a5).s1, (a5).s1, (a5).s1, (a5).s1, (a5).s1)) * b1));
    acc_5 = (acc_5 + (((float8)((a5).s2, (a5).s2, (a5).s2, (a5).s2, (a5).s2, (a5).s2, (a5).s2, (a5).s2)) * b2));
    acc_5 = (acc_5 + (((float8)((a5).s3, (a5).s3, (a5).s3, (a5).s3, (a5).s3, (a5).s3, (a5).s3, (a5).s3)) * b3));
    acc_6 = (acc_6 + (((float8)((a6).s0, (a6).s0, (a6).s0, (a6).s0, (a6).s0, (a6).s0, (a6).s0, (a6).s0)) * b0));
    acc_6 = (acc_6 + (((float8)((a6).s1, (a6).s1, (a6).s1, (a6).s1, (a6).s1, (a6).s1, (a6).s1, (a6).s1)) * b1));
    acc_6 = (acc_6 + (((float8)((a6).s2, (a6).s2, (a6).s2, (a6).s2, (a6).s2, (a6).s2, (a6).s2, (a6).s2)) * b2));
    acc_6 = (acc_6 + (((float8)((a6).s3, (a6).s3, (a6).s3, (a6).s3, (a6).s3, (a6).s3, (a6).s3, (a6).s3)) * b3));
    acc_7 = (acc_7 + (((float8)((a7).s0, (a7).s0, (a7).s0, (a7).s0, (a7).s0, (a7).s0, (a7).s0, (a7).s0)) * b0));
    acc_7 = (acc_7 + (((float8)((a7).s1, (a7).s1, (a7).s1, (a7).s1, (a7).s1, (a7).s1, (a7).s1, (a7).s1)) * b1));
    acc_7 = (acc_7 + (((float8)((a7).s2, (a7).s2, (a7).s2, (a7).s2, (a7).s2, (a7).s2, (a7).s2, (a7).s2)) * b2));
    acc_7 = (acc_7 + (((float8)((a7).s3, (a7).s3, (a7).s3, (a7).s3, (a7).s3, (a7).s3, (a7).s3, (a7).s3)) * b3));
  }
  vstore8((convert_half8(acc_0)), 0, C_1 + (((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20352)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)));
  vstore8((convert_half8(acc_1)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20352)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 2560));
  vstore8((convert_half8(acc_2)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20352)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 5120));
  vstore8((convert_half8(acc_3)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20352)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 7680));
  vstore8((convert_half8(acc_4)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20352)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 10240));
  vstore8((convert_half8(acc_5)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20352)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 12800));
  vstore8((convert_half8(acc_6)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20352)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 15360));
  vstore8((convert_half8(acc_7)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20352)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 17920));
}

