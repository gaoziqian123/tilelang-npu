// Function: gemm_nt_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void gemm_nt_kernel_kernel(__global half* restrict A, __global half* restrict B, __global half* restrict C);
__kernel void gemm_nt_kernel_kernel(__global half* restrict A, __global half* restrict B, __global half* restrict C) {
  half C_frag[64];
  for (int ko = 0; ko < 40; ++ko) {
    half A_frag[512];
    half B_frag[512];
    for (int i = 0; i < 64; ++i) {
      vstore8(vload8(0, A + ((((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20480)) + ((i >> 3) * 2560)) + (ko * 64)) + ((i & 7) * 8))), 0, A_frag + (i * 8));
    }
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      vstore8(vload8(0, B + ((((ko * 163840) + (i_1 * 2560)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8))), 0, B_frag + (i_1 * 8));
    }
    half8 acc[8];
    if (ko == 0) {
      half broadcast_var = (half)0.000000e+00f;
      acc[0] = ((half8)(broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var, broadcast_var));
    } else {
      acc[0] = vload8(0, C_frag + 0);
    }
    if (ko == 0) {
      half broadcast_var_1 = (half)0.000000e+00f;
      acc[1] = ((half8)(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1));
    } else {
      acc[1] = vload8(0, C_frag + 8);
    }
    if (ko == 0) {
      half broadcast_var_2 = (half)0.000000e+00f;
      acc[2] = ((half8)(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2));
    } else {
      acc[2] = vload8(0, C_frag + 16);
    }
    if (ko == 0) {
      half broadcast_var_3 = (half)0.000000e+00f;
      acc[3] = ((half8)(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3));
    } else {
      acc[3] = vload8(0, C_frag + 24);
    }
    if (ko == 0) {
      half broadcast_var_4 = (half)0.000000e+00f;
      acc[4] = ((half8)(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4));
    } else {
      acc[4] = vload8(0, C_frag + 32);
    }
    if (ko == 0) {
      half broadcast_var_5 = (half)0.000000e+00f;
      acc[5] = ((half8)(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5));
    } else {
      acc[5] = vload8(0, C_frag + 40);
    }
    if (ko == 0) {
      half broadcast_var_6 = (half)0.000000e+00f;
      acc[6] = ((half8)(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6));
    } else {
      acc[6] = vload8(0, C_frag + 48);
    }
    if (ko == 0) {
      half broadcast_var_7 = (half)0.000000e+00f;
      acc[7] = ((half8)(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7));
    } else {
      acc[7] = vload8(0, C_frag + 56);
    }
    for (int pos4 = 0; pos4 < 16; ++pos4) {
      acc[0] = ((((acc[0] + (((half8)((vload4(0, A_frag + (pos4 * 4))).s0, (vload4(0, A_frag + (pos4 * 4))).s0, (vload4(0, A_frag + (pos4 * 4))).s0, (vload4(0, A_frag + (pos4 * 4))).s0, (vload4(0, A_frag + (pos4 * 4))).s0, (vload4(0, A_frag + (pos4 * 4))).s0, (vload4(0, A_frag + (pos4 * 4))).s0, (vload4(0, A_frag + (pos4 * 4))).s0)) * vload8(0, B_frag + (pos4 * 32)))) + (((half8)((vload4(0, A_frag + (pos4 * 4))).s1, (vload4(0, A_frag + (pos4 * 4))).s1, (vload4(0, A_frag + (pos4 * 4))).s1, (vload4(0, A_frag + (pos4 * 4))).s1, (vload4(0, A_frag + (pos4 * 4))).s1, (vload4(0, A_frag + (pos4 * 4))).s1, (vload4(0, A_frag + (pos4 * 4))).s1, (vload4(0, A_frag + (pos4 * 4))).s1)) * vload8(0, B_frag + ((pos4 * 32) + 8)))) + (((half8)((vload4(0, A_frag + (pos4 * 4))).s2, (vload4(0, A_frag + (pos4 * 4))).s2, (vload4(0, A_frag + (pos4 * 4))).s2, (vload4(0, A_frag + (pos4 * 4))).s2, (vload4(0, A_frag + (pos4 * 4))).s2, (vload4(0, A_frag + (pos4 * 4))).s2, (vload4(0, A_frag + (pos4 * 4))).s2, (vload4(0, A_frag + (pos4 * 4))).s2)) * vload8(0, B_frag + ((pos4 * 32) + 16)))) + (((half8)((vload4(0, A_frag + (pos4 * 4))).s3, (vload4(0, A_frag + (pos4 * 4))).s3, (vload4(0, A_frag + (pos4 * 4))).s3, (vload4(0, A_frag + (pos4 * 4))).s3, (vload4(0, A_frag + (pos4 * 4))).s3, (vload4(0, A_frag + (pos4 * 4))).s3, (vload4(0, A_frag + (pos4 * 4))).s3, (vload4(0, A_frag + (pos4 * 4))).s3)) * vload8(0, B_frag + ((pos4 * 32) + 24))));
      acc[1] = ((((acc[1] + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 64))).s0, (vload4(0, A_frag + ((pos4 * 4) + 64))).s0, (vload4(0, A_frag + ((pos4 * 4) + 64))).s0, (vload4(0, A_frag + ((pos4 * 4) + 64))).s0, (vload4(0, A_frag + ((pos4 * 4) + 64))).s0, (vload4(0, A_frag + ((pos4 * 4) + 64))).s0, (vload4(0, A_frag + ((pos4 * 4) + 64))).s0, (vload4(0, A_frag + ((pos4 * 4) + 64))).s0)) * vload8(0, B_frag + (pos4 * 32)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 64))).s1, (vload4(0, A_frag + ((pos4 * 4) + 64))).s1, (vload4(0, A_frag + ((pos4 * 4) + 64))).s1, (vload4(0, A_frag + ((pos4 * 4) + 64))).s1, (vload4(0, A_frag + ((pos4 * 4) + 64))).s1, (vload4(0, A_frag + ((pos4 * 4) + 64))).s1, (vload4(0, A_frag + ((pos4 * 4) + 64))).s1, (vload4(0, A_frag + ((pos4 * 4) + 64))).s1)) * vload8(0, B_frag + ((pos4 * 32) + 8)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 64))).s2, (vload4(0, A_frag + ((pos4 * 4) + 64))).s2, (vload4(0, A_frag + ((pos4 * 4) + 64))).s2, (vload4(0, A_frag + ((pos4 * 4) + 64))).s2, (vload4(0, A_frag + ((pos4 * 4) + 64))).s2, (vload4(0, A_frag + ((pos4 * 4) + 64))).s2, (vload4(0, A_frag + ((pos4 * 4) + 64))).s2, (vload4(0, A_frag + ((pos4 * 4) + 64))).s2)) * vload8(0, B_frag + ((pos4 * 32) + 16)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 64))).s3, (vload4(0, A_frag + ((pos4 * 4) + 64))).s3, (vload4(0, A_frag + ((pos4 * 4) + 64))).s3, (vload4(0, A_frag + ((pos4 * 4) + 64))).s3, (vload4(0, A_frag + ((pos4 * 4) + 64))).s3, (vload4(0, A_frag + ((pos4 * 4) + 64))).s3, (vload4(0, A_frag + ((pos4 * 4) + 64))).s3, (vload4(0, A_frag + ((pos4 * 4) + 64))).s3)) * vload8(0, B_frag + ((pos4 * 32) + 24))));
      acc[2] = ((((acc[2] + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 128))).s0, (vload4(0, A_frag + ((pos4 * 4) + 128))).s0, (vload4(0, A_frag + ((pos4 * 4) + 128))).s0, (vload4(0, A_frag + ((pos4 * 4) + 128))).s0, (vload4(0, A_frag + ((pos4 * 4) + 128))).s0, (vload4(0, A_frag + ((pos4 * 4) + 128))).s0, (vload4(0, A_frag + ((pos4 * 4) + 128))).s0, (vload4(0, A_frag + ((pos4 * 4) + 128))).s0)) * vload8(0, B_frag + (pos4 * 32)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 128))).s1, (vload4(0, A_frag + ((pos4 * 4) + 128))).s1, (vload4(0, A_frag + ((pos4 * 4) + 128))).s1, (vload4(0, A_frag + ((pos4 * 4) + 128))).s1, (vload4(0, A_frag + ((pos4 * 4) + 128))).s1, (vload4(0, A_frag + ((pos4 * 4) + 128))).s1, (vload4(0, A_frag + ((pos4 * 4) + 128))).s1, (vload4(0, A_frag + ((pos4 * 4) + 128))).s1)) * vload8(0, B_frag + ((pos4 * 32) + 8)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 128))).s2, (vload4(0, A_frag + ((pos4 * 4) + 128))).s2, (vload4(0, A_frag + ((pos4 * 4) + 128))).s2, (vload4(0, A_frag + ((pos4 * 4) + 128))).s2, (vload4(0, A_frag + ((pos4 * 4) + 128))).s2, (vload4(0, A_frag + ((pos4 * 4) + 128))).s2, (vload4(0, A_frag + ((pos4 * 4) + 128))).s2, (vload4(0, A_frag + ((pos4 * 4) + 128))).s2)) * vload8(0, B_frag + ((pos4 * 32) + 16)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 128))).s3, (vload4(0, A_frag + ((pos4 * 4) + 128))).s3, (vload4(0, A_frag + ((pos4 * 4) + 128))).s3, (vload4(0, A_frag + ((pos4 * 4) + 128))).s3, (vload4(0, A_frag + ((pos4 * 4) + 128))).s3, (vload4(0, A_frag + ((pos4 * 4) + 128))).s3, (vload4(0, A_frag + ((pos4 * 4) + 128))).s3, (vload4(0, A_frag + ((pos4 * 4) + 128))).s3)) * vload8(0, B_frag + ((pos4 * 32) + 24))));
      acc[3] = ((((acc[3] + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 192))).s0, (vload4(0, A_frag + ((pos4 * 4) + 192))).s0, (vload4(0, A_frag + ((pos4 * 4) + 192))).s0, (vload4(0, A_frag + ((pos4 * 4) + 192))).s0, (vload4(0, A_frag + ((pos4 * 4) + 192))).s0, (vload4(0, A_frag + ((pos4 * 4) + 192))).s0, (vload4(0, A_frag + ((pos4 * 4) + 192))).s0, (vload4(0, A_frag + ((pos4 * 4) + 192))).s0)) * vload8(0, B_frag + (pos4 * 32)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 192))).s1, (vload4(0, A_frag + ((pos4 * 4) + 192))).s1, (vload4(0, A_frag + ((pos4 * 4) + 192))).s1, (vload4(0, A_frag + ((pos4 * 4) + 192))).s1, (vload4(0, A_frag + ((pos4 * 4) + 192))).s1, (vload4(0, A_frag + ((pos4 * 4) + 192))).s1, (vload4(0, A_frag + ((pos4 * 4) + 192))).s1, (vload4(0, A_frag + ((pos4 * 4) + 192))).s1)) * vload8(0, B_frag + ((pos4 * 32) + 8)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 192))).s2, (vload4(0, A_frag + ((pos4 * 4) + 192))).s2, (vload4(0, A_frag + ((pos4 * 4) + 192))).s2, (vload4(0, A_frag + ((pos4 * 4) + 192))).s2, (vload4(0, A_frag + ((pos4 * 4) + 192))).s2, (vload4(0, A_frag + ((pos4 * 4) + 192))).s2, (vload4(0, A_frag + ((pos4 * 4) + 192))).s2, (vload4(0, A_frag + ((pos4 * 4) + 192))).s2)) * vload8(0, B_frag + ((pos4 * 32) + 16)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 192))).s3, (vload4(0, A_frag + ((pos4 * 4) + 192))).s3, (vload4(0, A_frag + ((pos4 * 4) + 192))).s3, (vload4(0, A_frag + ((pos4 * 4) + 192))).s3, (vload4(0, A_frag + ((pos4 * 4) + 192))).s3, (vload4(0, A_frag + ((pos4 * 4) + 192))).s3, (vload4(0, A_frag + ((pos4 * 4) + 192))).s3, (vload4(0, A_frag + ((pos4 * 4) + 192))).s3)) * vload8(0, B_frag + ((pos4 * 32) + 24))));
      acc[4] = ((((acc[4] + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 256))).s0, (vload4(0, A_frag + ((pos4 * 4) + 256))).s0, (vload4(0, A_frag + ((pos4 * 4) + 256))).s0, (vload4(0, A_frag + ((pos4 * 4) + 256))).s0, (vload4(0, A_frag + ((pos4 * 4) + 256))).s0, (vload4(0, A_frag + ((pos4 * 4) + 256))).s0, (vload4(0, A_frag + ((pos4 * 4) + 256))).s0, (vload4(0, A_frag + ((pos4 * 4) + 256))).s0)) * vload8(0, B_frag + (pos4 * 32)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 256))).s1, (vload4(0, A_frag + ((pos4 * 4) + 256))).s1, (vload4(0, A_frag + ((pos4 * 4) + 256))).s1, (vload4(0, A_frag + ((pos4 * 4) + 256))).s1, (vload4(0, A_frag + ((pos4 * 4) + 256))).s1, (vload4(0, A_frag + ((pos4 * 4) + 256))).s1, (vload4(0, A_frag + ((pos4 * 4) + 256))).s1, (vload4(0, A_frag + ((pos4 * 4) + 256))).s1)) * vload8(0, B_frag + ((pos4 * 32) + 8)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 256))).s2, (vload4(0, A_frag + ((pos4 * 4) + 256))).s2, (vload4(0, A_frag + ((pos4 * 4) + 256))).s2, (vload4(0, A_frag + ((pos4 * 4) + 256))).s2, (vload4(0, A_frag + ((pos4 * 4) + 256))).s2, (vload4(0, A_frag + ((pos4 * 4) + 256))).s2, (vload4(0, A_frag + ((pos4 * 4) + 256))).s2, (vload4(0, A_frag + ((pos4 * 4) + 256))).s2)) * vload8(0, B_frag + ((pos4 * 32) + 16)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 256))).s3, (vload4(0, A_frag + ((pos4 * 4) + 256))).s3, (vload4(0, A_frag + ((pos4 * 4) + 256))).s3, (vload4(0, A_frag + ((pos4 * 4) + 256))).s3, (vload4(0, A_frag + ((pos4 * 4) + 256))).s3, (vload4(0, A_frag + ((pos4 * 4) + 256))).s3, (vload4(0, A_frag + ((pos4 * 4) + 256))).s3, (vload4(0, A_frag + ((pos4 * 4) + 256))).s3)) * vload8(0, B_frag + ((pos4 * 32) + 24))));
      acc[5] = ((((acc[5] + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 320))).s0, (vload4(0, A_frag + ((pos4 * 4) + 320))).s0, (vload4(0, A_frag + ((pos4 * 4) + 320))).s0, (vload4(0, A_frag + ((pos4 * 4) + 320))).s0, (vload4(0, A_frag + ((pos4 * 4) + 320))).s0, (vload4(0, A_frag + ((pos4 * 4) + 320))).s0, (vload4(0, A_frag + ((pos4 * 4) + 320))).s0, (vload4(0, A_frag + ((pos4 * 4) + 320))).s0)) * vload8(0, B_frag + (pos4 * 32)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 320))).s1, (vload4(0, A_frag + ((pos4 * 4) + 320))).s1, (vload4(0, A_frag + ((pos4 * 4) + 320))).s1, (vload4(0, A_frag + ((pos4 * 4) + 320))).s1, (vload4(0, A_frag + ((pos4 * 4) + 320))).s1, (vload4(0, A_frag + ((pos4 * 4) + 320))).s1, (vload4(0, A_frag + ((pos4 * 4) + 320))).s1, (vload4(0, A_frag + ((pos4 * 4) + 320))).s1)) * vload8(0, B_frag + ((pos4 * 32) + 8)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 320))).s2, (vload4(0, A_frag + ((pos4 * 4) + 320))).s2, (vload4(0, A_frag + ((pos4 * 4) + 320))).s2, (vload4(0, A_frag + ((pos4 * 4) + 320))).s2, (vload4(0, A_frag + ((pos4 * 4) + 320))).s2, (vload4(0, A_frag + ((pos4 * 4) + 320))).s2, (vload4(0, A_frag + ((pos4 * 4) + 320))).s2, (vload4(0, A_frag + ((pos4 * 4) + 320))).s2)) * vload8(0, B_frag + ((pos4 * 32) + 16)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 320))).s3, (vload4(0, A_frag + ((pos4 * 4) + 320))).s3, (vload4(0, A_frag + ((pos4 * 4) + 320))).s3, (vload4(0, A_frag + ((pos4 * 4) + 320))).s3, (vload4(0, A_frag + ((pos4 * 4) + 320))).s3, (vload4(0, A_frag + ((pos4 * 4) + 320))).s3, (vload4(0, A_frag + ((pos4 * 4) + 320))).s3, (vload4(0, A_frag + ((pos4 * 4) + 320))).s3)) * vload8(0, B_frag + ((pos4 * 32) + 24))));
      acc[6] = ((((acc[6] + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 384))).s0, (vload4(0, A_frag + ((pos4 * 4) + 384))).s0, (vload4(0, A_frag + ((pos4 * 4) + 384))).s0, (vload4(0, A_frag + ((pos4 * 4) + 384))).s0, (vload4(0, A_frag + ((pos4 * 4) + 384))).s0, (vload4(0, A_frag + ((pos4 * 4) + 384))).s0, (vload4(0, A_frag + ((pos4 * 4) + 384))).s0, (vload4(0, A_frag + ((pos4 * 4) + 384))).s0)) * vload8(0, B_frag + (pos4 * 32)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 384))).s1, (vload4(0, A_frag + ((pos4 * 4) + 384))).s1, (vload4(0, A_frag + ((pos4 * 4) + 384))).s1, (vload4(0, A_frag + ((pos4 * 4) + 384))).s1, (vload4(0, A_frag + ((pos4 * 4) + 384))).s1, (vload4(0, A_frag + ((pos4 * 4) + 384))).s1, (vload4(0, A_frag + ((pos4 * 4) + 384))).s1, (vload4(0, A_frag + ((pos4 * 4) + 384))).s1)) * vload8(0, B_frag + ((pos4 * 32) + 8)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 384))).s2, (vload4(0, A_frag + ((pos4 * 4) + 384))).s2, (vload4(0, A_frag + ((pos4 * 4) + 384))).s2, (vload4(0, A_frag + ((pos4 * 4) + 384))).s2, (vload4(0, A_frag + ((pos4 * 4) + 384))).s2, (vload4(0, A_frag + ((pos4 * 4) + 384))).s2, (vload4(0, A_frag + ((pos4 * 4) + 384))).s2, (vload4(0, A_frag + ((pos4 * 4) + 384))).s2)) * vload8(0, B_frag + ((pos4 * 32) + 16)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 384))).s3, (vload4(0, A_frag + ((pos4 * 4) + 384))).s3, (vload4(0, A_frag + ((pos4 * 4) + 384))).s3, (vload4(0, A_frag + ((pos4 * 4) + 384))).s3, (vload4(0, A_frag + ((pos4 * 4) + 384))).s3, (vload4(0, A_frag + ((pos4 * 4) + 384))).s3, (vload4(0, A_frag + ((pos4 * 4) + 384))).s3, (vload4(0, A_frag + ((pos4 * 4) + 384))).s3)) * vload8(0, B_frag + ((pos4 * 32) + 24))));
      acc[7] = ((((acc[7] + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 448))).s0, (vload4(0, A_frag + ((pos4 * 4) + 448))).s0, (vload4(0, A_frag + ((pos4 * 4) + 448))).s0, (vload4(0, A_frag + ((pos4 * 4) + 448))).s0, (vload4(0, A_frag + ((pos4 * 4) + 448))).s0, (vload4(0, A_frag + ((pos4 * 4) + 448))).s0, (vload4(0, A_frag + ((pos4 * 4) + 448))).s0, (vload4(0, A_frag + ((pos4 * 4) + 448))).s0)) * vload8(0, B_frag + (pos4 * 32)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 448))).s1, (vload4(0, A_frag + ((pos4 * 4) + 448))).s1, (vload4(0, A_frag + ((pos4 * 4) + 448))).s1, (vload4(0, A_frag + ((pos4 * 4) + 448))).s1, (vload4(0, A_frag + ((pos4 * 4) + 448))).s1, (vload4(0, A_frag + ((pos4 * 4) + 448))).s1, (vload4(0, A_frag + ((pos4 * 4) + 448))).s1, (vload4(0, A_frag + ((pos4 * 4) + 448))).s1)) * vload8(0, B_frag + ((pos4 * 32) + 8)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 448))).s2, (vload4(0, A_frag + ((pos4 * 4) + 448))).s2, (vload4(0, A_frag + ((pos4 * 4) + 448))).s2, (vload4(0, A_frag + ((pos4 * 4) + 448))).s2, (vload4(0, A_frag + ((pos4 * 4) + 448))).s2, (vload4(0, A_frag + ((pos4 * 4) + 448))).s2, (vload4(0, A_frag + ((pos4 * 4) + 448))).s2, (vload4(0, A_frag + ((pos4 * 4) + 448))).s2)) * vload8(0, B_frag + ((pos4 * 32) + 16)))) + (((half8)((vload4(0, A_frag + ((pos4 * 4) + 448))).s3, (vload4(0, A_frag + ((pos4 * 4) + 448))).s3, (vload4(0, A_frag + ((pos4 * 4) + 448))).s3, (vload4(0, A_frag + ((pos4 * 4) + 448))).s3, (vload4(0, A_frag + ((pos4 * 4) + 448))).s3, (vload4(0, A_frag + ((pos4 * 4) + 448))).s3, (vload4(0, A_frag + ((pos4 * 4) + 448))).s3, (vload4(0, A_frag + ((pos4 * 4) + 448))).s3)) * vload8(0, B_frag + ((pos4 * 32) + 24))));
    }
    C_frag[0] = (acc[0]).s0;
    C_frag[1] = (acc[0]).s1;
    C_frag[2] = (acc[0]).s2;
    C_frag[3] = (acc[0]).s3;
    C_frag[4] = (acc[0]).s4;
    C_frag[5] = (acc[0]).s5;
    C_frag[6] = (acc[0]).s6;
    C_frag[7] = (acc[0]).s7;
    C_frag[8] = (acc[1]).s0;
    C_frag[9] = (acc[1]).s1;
    C_frag[10] = (acc[1]).s2;
    C_frag[11] = (acc[1]).s3;
    C_frag[12] = (acc[1]).s4;
    C_frag[13] = (acc[1]).s5;
    C_frag[14] = (acc[1]).s6;
    C_frag[15] = (acc[1]).s7;
    C_frag[16] = (acc[2]).s0;
    C_frag[17] = (acc[2]).s1;
    C_frag[18] = (acc[2]).s2;
    C_frag[19] = (acc[2]).s3;
    C_frag[20] = (acc[2]).s4;
    C_frag[21] = (acc[2]).s5;
    C_frag[22] = (acc[2]).s6;
    C_frag[23] = (acc[2]).s7;
    C_frag[24] = (acc[3]).s0;
    C_frag[25] = (acc[3]).s1;
    C_frag[26] = (acc[3]).s2;
    C_frag[27] = (acc[3]).s3;
    C_frag[28] = (acc[3]).s4;
    C_frag[29] = (acc[3]).s5;
    C_frag[30] = (acc[3]).s6;
    C_frag[31] = (acc[3]).s7;
    C_frag[32] = (acc[4]).s0;
    C_frag[33] = (acc[4]).s1;
    C_frag[34] = (acc[4]).s2;
    C_frag[35] = (acc[4]).s3;
    C_frag[36] = (acc[4]).s4;
    C_frag[37] = (acc[4]).s5;
    C_frag[38] = (acc[4]).s6;
    C_frag[39] = (acc[4]).s7;
    C_frag[40] = (acc[5]).s0;
    C_frag[41] = (acc[5]).s1;
    C_frag[42] = (acc[5]).s2;
    C_frag[43] = (acc[5]).s3;
    C_frag[44] = (acc[5]).s4;
    C_frag[45] = (acc[5]).s5;
    C_frag[46] = (acc[5]).s6;
    C_frag[47] = (acc[5]).s7;
    C_frag[48] = (acc[6]).s0;
    C_frag[49] = (acc[6]).s1;
    C_frag[50] = (acc[6]).s2;
    C_frag[51] = (acc[6]).s3;
    C_frag[52] = (acc[6]).s4;
    C_frag[53] = (acc[6]).s5;
    C_frag[54] = (acc[6]).s6;
    C_frag[55] = (acc[6]).s7;
    C_frag[56] = (acc[7]).s0;
    C_frag[57] = (acc[7]).s1;
    C_frag[58] = (acc[7]).s2;
    C_frag[59] = (acc[7]).s3;
    C_frag[60] = (acc[7]).s4;
    C_frag[61] = (acc[7]).s5;
    C_frag[62] = (acc[7]).s6;
    C_frag[63] = (acc[7]).s7;
  }
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    vstore8(vload8(0, C_frag + (i_2 * 8)), 0, C + ((((((convert_int(get_group_id(1))) * 81920) + (((convert_int(get_local_id(0))) >> 4) * 20480)) + (i_2 * 2560)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)));
  }
}

