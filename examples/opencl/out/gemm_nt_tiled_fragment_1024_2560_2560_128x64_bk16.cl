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
  __local half As[2048];
  __local half Bs[1024];
  float8 acc_0;
  float8 acc_1;
  float8 acc_2;
  float8 acc_3;
  float8 acc_4;
  float8 acc_5;
  float8 acc_6;
  float8 acc_7;
  int tm = ((convert_int(get_local_id(0))) >> 3);
  int tn = ((convert_int(get_local_id(0))) - (((convert_int(get_local_id(0))) >> 3) * 8));
  int rbase = ((convert_int(get_group_id(1))) * 128);
  int cbase = ((convert_int(get_group_id(0))) * 64);
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
  for (int kb = 0; kb < 2560; kb += 16) {
    for (int v = (convert_int(get_local_id(0))); v < 512; v += 128) {
      int r = (v >> 2);
      int c4 = (v - ((v >> 2) * 4));
      half4 aval = vload4(0, ((__global half *)A_1 + (((((convert_int(get_group_id(1))) * 327680) + ((v >> 2) * 2544)) + (v * 4)) + kb)));
      As[((v * 512) - ((v >> 2) * 2047))] = (aval).s0;
      As[(((v * 512) + 128) - ((v >> 2) * 2047))] = (aval).s1;
      As[(((v * 512) + 256) - ((v >> 2) * 2047))] = (aval).s2;
      As[(((v * 512) + 384) - ((v >> 2) * 2047))] = (aval).s3;
    }
    for (int v_1 = (convert_int(get_local_id(0))); v_1 < 256; v_1 += 128) {
      int c = (v_1 >> 2);
      int k4 = (v_1 - ((v_1 >> 2) * 4));
      half4 bval = vload4(0, ((__global half *)B_1 + (((((convert_int(get_group_id(0))) * 163840) + ((v_1 >> 2) * 2544)) + (v_1 * 4)) + kb)));
      Bs[((v_1 * 256) - ((v_1 >> 2) * 1023))] = (bval).s0;
      Bs[(((v_1 * 256) + 64) - ((v_1 >> 2) * 1023))] = (bval).s1;
      Bs[(((v_1 * 256) + 128) - ((v_1 >> 2) * 1023))] = (bval).s2;
      Bs[(((v_1 * 256) + 192) - ((v_1 >> 2) * 1023))] = (bval).s3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kk = 0; kk < 16; ++kk) {
      float8 a8 = (convert_float8(vload8(0, As + ((kk * 128) + (((convert_int(get_local_id(0))) >> 3) * 8)))));
      float8 b8 = (convert_float8(vload8(0, Bs + (((kk * 64) + ((convert_int(get_local_id(0))) * 8)) - (((convert_int(get_local_id(0))) >> 3) * 64)))));
      acc_0 = (acc_0 + (((float8)((a8).s0, (a8).s0, (a8).s0, (a8).s0, (a8).s0, (a8).s0, (a8).s0, (a8).s0)) * b8));
      acc_1 = (acc_1 + (((float8)((a8).s1, (a8).s1, (a8).s1, (a8).s1, (a8).s1, (a8).s1, (a8).s1, (a8).s1)) * b8));
      acc_2 = (acc_2 + (((float8)((a8).s2, (a8).s2, (a8).s2, (a8).s2, (a8).s2, (a8).s2, (a8).s2, (a8).s2)) * b8));
      acc_3 = (acc_3 + (((float8)((a8).s3, (a8).s3, (a8).s3, (a8).s3, (a8).s3, (a8).s3, (a8).s3, (a8).s3)) * b8));
      acc_4 = (acc_4 + (((float8)((a8).s4, (a8).s4, (a8).s4, (a8).s4, (a8).s4, (a8).s4, (a8).s4, (a8).s4)) * b8));
      acc_5 = (acc_5 + (((float8)((a8).s5, (a8).s5, (a8).s5, (a8).s5, (a8).s5, (a8).s5, (a8).s5, (a8).s5)) * b8));
      acc_6 = (acc_6 + (((float8)((a8).s6, (a8).s6, (a8).s6, (a8).s6, (a8).s6, (a8).s6, (a8).s6, (a8).s6)) * b8));
      acc_7 = (acc_7 + (((float8)((a8).s7, (a8).s7, (a8).s7, (a8).s7, (a8).s7, (a8).s7, (a8).s7, (a8).s7)) * b8));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int r0 = (((convert_int(get_group_id(1))) * 128) + (((convert_int(get_local_id(0))) >> 3) * 8));
  int c0 = ((((convert_int(get_group_id(0))) * 64) + ((convert_int(get_local_id(0))) * 8)) - (((convert_int(get_local_id(0))) >> 3) * 64));
  vstore8((convert_half8(acc_0)), 0, C_1 + (((((convert_int(get_group_id(1))) * 327680) + (((convert_int(get_local_id(0))) >> 3) * 20416)) + ((convert_int(get_group_id(0))) * 64)) + ((convert_int(get_local_id(0))) * 8)));
  vstore8((convert_half8(acc_1)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 327680) + (((convert_int(get_local_id(0))) >> 3) * 20416)) + ((convert_int(get_group_id(0))) * 64)) + ((convert_int(get_local_id(0))) * 8)) + 2560));
  vstore8((convert_half8(acc_2)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 327680) + (((convert_int(get_local_id(0))) >> 3) * 20416)) + ((convert_int(get_group_id(0))) * 64)) + ((convert_int(get_local_id(0))) * 8)) + 5120));
  vstore8((convert_half8(acc_3)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 327680) + (((convert_int(get_local_id(0))) >> 3) * 20416)) + ((convert_int(get_group_id(0))) * 64)) + ((convert_int(get_local_id(0))) * 8)) + 7680));
  vstore8((convert_half8(acc_4)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 327680) + (((convert_int(get_local_id(0))) >> 3) * 20416)) + ((convert_int(get_group_id(0))) * 64)) + ((convert_int(get_local_id(0))) * 8)) + 10240));
  vstore8((convert_half8(acc_5)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 327680) + (((convert_int(get_local_id(0))) >> 3) * 20416)) + ((convert_int(get_group_id(0))) * 64)) + ((convert_int(get_local_id(0))) * 8)) + 12800));
  vstore8((convert_half8(acc_6)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 327680) + (((convert_int(get_local_id(0))) >> 3) * 20416)) + ((convert_int(get_group_id(0))) * 64)) + ((convert_int(get_local_id(0))) * 8)) + 15360));
  vstore8((convert_half8(acc_7)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 327680) + (((convert_int(get_local_id(0))) >> 3) * 20416)) + ((convert_int(get_group_id(0))) * 64)) + ((convert_int(get_local_id(0))) * 8)) + 17920));
}

