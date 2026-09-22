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
  half C_frag_1[64];
  half8 acc[8];
  half4 av[8];
  half8 bv[4];
  half8 acc_1[8];
  half broadcast_var = (half)0.000000e+00f;
  acc_1[0] = ((half8)(broadcast_var));
  half broadcast_var_1 = (half)0.000000e+00f;
  acc_1[1] = ((half8)(broadcast_var_1));
  half broadcast_var_2 = (half)0.000000e+00f;
  acc_1[2] = ((half8)(broadcast_var_2));
  half broadcast_var_3 = (half)0.000000e+00f;
  acc_1[3] = ((half8)(broadcast_var_3));
  half broadcast_var_4 = (half)0.000000e+00f;
  acc_1[4] = ((half8)(broadcast_var_4));
  half broadcast_var_5 = (half)0.000000e+00f;
  acc_1[5] = ((half8)(broadcast_var_5));
  half broadcast_var_6 = (half)0.000000e+00f;
  acc_1[6] = ((half8)(broadcast_var_6));
  half broadcast_var_7 = (half)0.000000e+00f;
  acc_1[7] = ((half8)(broadcast_var_7));
  for (int pos4 = 0; pos4 < 128; ++pos4) {
    half4 av_1[8];
    half8 bv_1[4];
    av_1[0] = vload4(0, A + ((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)));
    av_1[1] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 512));
    av_1[2] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 1024));
    av_1[3] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 1536));
    av_1[4] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 2048));
    av_1[5] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 2560));
    av_1[6] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 3072));
    av_1[7] = vload4(0, A + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + (pos4 * 4)) + 3584));
    bv_1[0] = vload8(0, B + ((((pos4 * 2048) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) - (((convert_int(get_local_id(0))) >> 4) * 128)));
    bv_1[1] = vload8(0, B + (((((pos4 * 2048) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 512) - (((convert_int(get_local_id(0))) >> 4) * 128)));
    bv_1[2] = vload8(0, B + (((((pos4 * 2048) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 1024) - (((convert_int(get_local_id(0))) >> 4) * 128)));
    bv_1[3] = vload8(0, B + (((((pos4 * 2048) + ((convert_int(get_group_id(0))) * 128)) + ((convert_int(get_local_id(0))) * 8)) + 1536) - (((convert_int(get_local_id(0))) >> 4) * 128)));
    acc_1[0] = mad(((half8)((av_1[0]).s0)), bv_1[0], mad(((half8)((av_1[0]).s1)), bv_1[1], mad(((half8)((av_1[0]).s2)), bv_1[2], mad(((half8)((av_1[0]).s3)), bv_1[3], acc_1[0]))));
    acc_1[1] = mad(((half8)((av_1[1]).s0)), bv_1[0], mad(((half8)((av_1[1]).s1)), bv_1[1], mad(((half8)((av_1[1]).s2)), bv_1[2], mad(((half8)((av_1[1]).s3)), bv_1[3], acc_1[1]))));
    acc_1[2] = mad(((half8)((av_1[2]).s0)), bv_1[0], mad(((half8)((av_1[2]).s1)), bv_1[1], mad(((half8)((av_1[2]).s2)), bv_1[2], mad(((half8)((av_1[2]).s3)), bv_1[3], acc_1[2]))));
    acc_1[3] = mad(((half8)((av_1[3]).s0)), bv_1[0], mad(((half8)((av_1[3]).s1)), bv_1[1], mad(((half8)((av_1[3]).s2)), bv_1[2], mad(((half8)((av_1[3]).s3)), bv_1[3], acc_1[3]))));
    acc_1[4] = mad(((half8)((av_1[4]).s0)), bv_1[0], mad(((half8)((av_1[4]).s1)), bv_1[1], mad(((half8)((av_1[4]).s2)), bv_1[2], mad(((half8)((av_1[4]).s3)), bv_1[3], acc_1[4]))));
    acc_1[5] = mad(((half8)((av_1[5]).s0)), bv_1[0], mad(((half8)((av_1[5]).s1)), bv_1[1], mad(((half8)((av_1[5]).s2)), bv_1[2], mad(((half8)((av_1[5]).s3)), bv_1[3], acc_1[5]))));
    acc_1[6] = mad(((half8)((av_1[6]).s0)), bv_1[0], mad(((half8)((av_1[6]).s1)), bv_1[1], mad(((half8)((av_1[6]).s2)), bv_1[2], mad(((half8)((av_1[6]).s3)), bv_1[3], acc_1[6]))));
    acc_1[7] = mad(((half8)((av_1[7]).s0)), bv_1[0], mad(((half8)((av_1[7]).s1)), bv_1[1], mad(((half8)((av_1[7]).s2)), bv_1[2], mad(((half8)((av_1[7]).s3)), bv_1[3], acc_1[7]))));
  }
  C_frag_1[0] = (acc_1[0]).s0;
  C_frag_1[1] = (acc_1[0]).s1;
  C_frag_1[2] = (acc_1[0]).s2;
  C_frag_1[3] = (acc_1[0]).s3;
  C_frag_1[4] = (acc_1[0]).s4;
  C_frag_1[5] = (acc_1[0]).s5;
  C_frag_1[6] = (acc_1[0]).s6;
  C_frag_1[7] = (acc_1[0]).s7;
  C_frag_1[8] = (acc_1[1]).s0;
  C_frag_1[9] = (acc_1[1]).s1;
  C_frag_1[10] = (acc_1[1]).s2;
  C_frag_1[11] = (acc_1[1]).s3;
  C_frag_1[12] = (acc_1[1]).s4;
  C_frag_1[13] = (acc_1[1]).s5;
  C_frag_1[14] = (acc_1[1]).s6;
  C_frag_1[15] = (acc_1[1]).s7;
  C_frag_1[16] = (acc_1[2]).s0;
  C_frag_1[17] = (acc_1[2]).s1;
  C_frag_1[18] = (acc_1[2]).s2;
  C_frag_1[19] = (acc_1[2]).s3;
  C_frag_1[20] = (acc_1[2]).s4;
  C_frag_1[21] = (acc_1[2]).s5;
  C_frag_1[22] = (acc_1[2]).s6;
  C_frag_1[23] = (acc_1[2]).s7;
  C_frag_1[24] = (acc_1[3]).s0;
  C_frag_1[25] = (acc_1[3]).s1;
  C_frag_1[26] = (acc_1[3]).s2;
  C_frag_1[27] = (acc_1[3]).s3;
  C_frag_1[28] = (acc_1[3]).s4;
  C_frag_1[29] = (acc_1[3]).s5;
  C_frag_1[30] = (acc_1[3]).s6;
  C_frag_1[31] = (acc_1[3]).s7;
  C_frag_1[32] = (acc_1[4]).s0;
  C_frag_1[33] = (acc_1[4]).s1;
  C_frag_1[34] = (acc_1[4]).s2;
  C_frag_1[35] = (acc_1[4]).s3;
  C_frag_1[36] = (acc_1[4]).s4;
  C_frag_1[37] = (acc_1[4]).s5;
  C_frag_1[38] = (acc_1[4]).s6;
  C_frag_1[39] = (acc_1[4]).s7;
  C_frag_1[40] = (acc_1[5]).s0;
  C_frag_1[41] = (acc_1[5]).s1;
  C_frag_1[42] = (acc_1[5]).s2;
  C_frag_1[43] = (acc_1[5]).s3;
  C_frag_1[44] = (acc_1[5]).s4;
  C_frag_1[45] = (acc_1[5]).s5;
  C_frag_1[46] = (acc_1[5]).s6;
  C_frag_1[47] = (acc_1[5]).s7;
  C_frag_1[48] = (acc_1[6]).s0;
  C_frag_1[49] = (acc_1[6]).s1;
  C_frag_1[50] = (acc_1[6]).s2;
  C_frag_1[51] = (acc_1[6]).s3;
  C_frag_1[52] = (acc_1[6]).s4;
  C_frag_1[53] = (acc_1[6]).s5;
  C_frag_1[54] = (acc_1[6]).s6;
  C_frag_1[55] = (acc_1[6]).s7;
  C_frag_1[56] = (acc_1[7]).s0;
  C_frag_1[57] = (acc_1[7]).s1;
  C_frag_1[58] = (acc_1[7]).s2;
  C_frag_1[59] = (acc_1[7]).s3;
  C_frag_1[60] = (acc_1[7]).s4;
  C_frag_1[61] = (acc_1[7]).s5;
  C_frag_1[62] = (acc_1[7]).s6;
  C_frag_1[63] = (acc_1[7]).s7;
  vstore8(vload8(0, C_frag_1 + 0), 0, C + (((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)));
  vstore8(vload8(0, C_frag_1 + 8), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 512));
  vstore8(vload8(0, C_frag_1 + 16), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1024));
  vstore8(vload8(0, C_frag_1 + 24), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1536));
  vstore8(vload8(0, C_frag_1 + 32), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2048));
  vstore8(vload8(0, C_frag_1 + 40), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2560));
  vstore8(vload8(0, C_frag_1 + 48), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3072));
  vstore8(vload8(0, C_frag_1 + 56), 0, C + ((((((convert_int(get_group_id(1))) * 16384) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3584));
}

