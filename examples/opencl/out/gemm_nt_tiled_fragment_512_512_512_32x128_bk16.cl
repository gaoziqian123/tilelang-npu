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
  __local half As[512];
  __local half Bs[2048];
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
  int rbase = ((convert_int(get_group_id(1))) * 32);
  int cbase = ((convert_int(get_group_id(0))) * 128);
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
  for (int kbb = 0; kbb < 32; ++kbb) {
    for (int v = (convert_int(get_local_id(0))); v < 128; v += 64) {
      int r = (v >> 2);
      int c4 = (v - ((v >> 2) * 4));
      half4 aval = vload4(0, A_1 + (((((convert_int(get_group_id(1))) * 16384) + ((v >> 2) * 496)) + (kbb * 16)) + (v * 4)));
      As[((v * 128) - ((v >> 2) * 511))] = (aval).s0;
      As[(((v * 128) + 32) - ((v >> 2) * 511))] = (aval).s1;
      As[(((v * 128) + 64) - ((v >> 2) * 511))] = (aval).s2;
      As[(((v * 128) + 96) - ((v >> 2) * 511))] = (aval).s3;
    }
    for (int v_1 = (convert_int(get_local_id(0))); v_1 < 512; v_1 += 64) {
      int c = (v_1 >> 2);
      int k4 = (v_1 - ((v_1 >> 2) * 4));
      half4 bval = vload4(0, B_1 + (((((convert_int(get_group_id(0))) * 65536) + ((v_1 >> 2) * 496)) + (kbb * 16)) + (v_1 * 4)));
      Bs[((v_1 * 512) - ((v_1 >> 2) * 2047))] = (bval).s0;
      Bs[(((v_1 * 512) + 128) - ((v_1 >> 2) * 2047))] = (bval).s1;
      Bs[(((v_1 * 512) + 256) - ((v_1 >> 2) * 2047))] = (bval).s2;
      Bs[(((v_1 * 512) + 384) - ((v_1 >> 2) * 2047))] = (bval).s3;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kk = 0; kk < 16; ++kk) {
      float8 a8 = (convert_float8(vload8(0, As + ((kk * 32) + (((convert_int(get_local_id(0))) >> 4) * 8)))));
      float8 b8 = (convert_float8(vload8(0, Bs + (((kk * 128) + ((convert_int(get_local_id(0))) * 8)) - (((convert_int(get_local_id(0))) >> 4) * 128)))));
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
  int r0 = (((convert_int(get_group_id(1))) * 32) + (((convert_int(get_local_id(0))) >> 4) * 8));
  int c0 = ((((convert_int(get_group_id(0))) * 128) + ((convert_int(get_local_id(0))) * 8)) - (((convert_int(get_local_id(0))) >> 4) * 128));
  vstore8((convert_half8(acc_0)), 0, C_1 + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 3968)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)));
  vstore8((convert_half8(acc_1)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 3968)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 512));
  vstore8((convert_half8(acc_2)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 3968)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 1024));
  vstore8((convert_half8(acc_3)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 3968)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 1536));
  vstore8((convert_half8(acc_4)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 3968)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 2048));
  vstore8((convert_half8(acc_5)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 3968)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 2560));
  vstore8((convert_half8(acc_6)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 3968)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 3072));
  vstore8((convert_half8(acc_7)), 0, C_1 + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 3968)) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 3584));
}

