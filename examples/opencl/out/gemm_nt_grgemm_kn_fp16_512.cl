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
  half8 acc[8];
  half broadcast_var = (half)0.000000e+00f;
  acc[0] = ((half8)(broadcast_var));
  half broadcast_var_1 = (half)0.000000e+00f;
  acc[1] = ((half8)(broadcast_var_1));
  half broadcast_var_2 = (half)0.000000e+00f;
  acc[2] = ((half8)(broadcast_var_2));
  half broadcast_var_3 = (half)0.000000e+00f;
  acc[3] = ((half8)(broadcast_var_3));
  half broadcast_var_4 = (half)0.000000e+00f;
  acc[4] = ((half8)(broadcast_var_4));
  half broadcast_var_5 = (half)0.000000e+00f;
  acc[5] = ((half8)(broadcast_var_5));
  half broadcast_var_6 = (half)0.000000e+00f;
  acc[6] = ((half8)(broadcast_var_6));
  half broadcast_var_7 = (half)0.000000e+00f;
  acc[7] = ((half8)(broadcast_var_7));
  for (int pos4 = 0; pos4 < 128; ++pos4) {
    half4 av[8];
    half8 bv[4];
    av[0] = vload4(0, A + ((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)));
    av[1] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 512));
    av[2] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 1024));
    av[3] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 1536));
    av[4] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 2048));
    av[5] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 2560));
    av[6] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 3072));
    av[7] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 3584));
    bv[0] = vload8(0, B + ((((pos4 * 2048) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) - (((convert_int(get_local_id(0))) >> 4) * 128)));
    bv[1] = vload8(0, B + (((((pos4 * 2048) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 512) - (((convert_int(get_local_id(0))) >> 4) * 128)));
    bv[2] = vload8(0, B + (((((pos4 * 2048) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 1024) - (((convert_int(get_local_id(0))) >> 4) * 128)));
    bv[3] = vload8(0, B + (((((pos4 * 2048) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 1536) - (((convert_int(get_local_id(0))) >> 4) * 128)));
    acc[0] = mad(((half8)((av[0]).s0)), bv[0], mad(((half8)((av[0]).s1)), bv[1], mad(((half8)((av[0]).s2)), bv[2], mad(((half8)((av[0]).s3)), bv[3], acc[0]))));
    acc[1] = mad(((half8)((av[1]).s0)), bv[0], mad(((half8)((av[1]).s1)), bv[1], mad(((half8)((av[1]).s2)), bv[2], mad(((half8)((av[1]).s3)), bv[3], acc[1]))));
    acc[2] = mad(((half8)((av[2]).s0)), bv[0], mad(((half8)((av[2]).s1)), bv[1], mad(((half8)((av[2]).s2)), bv[2], mad(((half8)((av[2]).s3)), bv[3], acc[2]))));
    acc[3] = mad(((half8)((av[3]).s0)), bv[0], mad(((half8)((av[3]).s1)), bv[1], mad(((half8)((av[3]).s2)), bv[2], mad(((half8)((av[3]).s3)), bv[3], acc[3]))));
    acc[4] = mad(((half8)((av[4]).s0)), bv[0], mad(((half8)((av[4]).s1)), bv[1], mad(((half8)((av[4]).s2)), bv[2], mad(((half8)((av[4]).s3)), bv[3], acc[4]))));
    acc[5] = mad(((half8)((av[5]).s0)), bv[0], mad(((half8)((av[5]).s1)), bv[1], mad(((half8)((av[5]).s2)), bv[2], mad(((half8)((av[5]).s3)), bv[3], acc[5]))));
    acc[6] = mad(((half8)((av[6]).s0)), bv[0], mad(((half8)((av[6]).s1)), bv[1], mad(((half8)((av[6]).s2)), bv[2], mad(((half8)((av[6]).s3)), bv[3], acc[6]))));
    acc[7] = mad(((half8)((av[7]).s0)), bv[0], mad(((half8)((av[7]).s1)), bv[1], mad(((half8)((av[7]).s2)), bv[2], mad(((half8)((av[7]).s3)), bv[3], acc[7]))));
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
  vstore8(vload8(0, C_frag + 0), 0, C + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)));
  vstore8(vload8(0, C_frag + 8), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 512));
  vstore8(vload8(0, C_frag + 16), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1024));
  vstore8(vload8(0, C_frag + 24), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1536));
  vstore8(vload8(0, C_frag + 32), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2048));
  vstore8(vload8(0, C_frag + 40), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2560));
  vstore8(vload8(0, C_frag + 48), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3072));
  vstore8(vload8(0, C_frag + 56), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3584));
}

