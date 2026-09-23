// Function: gemm_nt_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void gemm_nt_kernel_kernel(__global half* restrict A, __global half* restrict B, __global half* restrict C);
__kernel void gemm_nt_kernel_kernel(__global half* restrict A, __global half* restrict B, __global half* restrict C) { const int tl_gid0 = convert_int(get_group_id(0)); const int tl_gid1 = convert_int(get_group_id(1)); const int tl_lid0 = convert_int(get_local_id(0)); const int tl_wi_aff0 = ((tl_gid1 * 40960) + ((tl_lid0 >> 3) * 20480)); const int tl_wi_aff1 = ((tl_lid0 >> 3) * 64); const int tl_wi_aff2 = (tl_gid0 * 64); const int tl_wi_aff3 = (tl_lid0 * 8);
  float C_frag_1[64];
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
  for (int pos4 = 0; pos4 < 640; ++pos4) { const int tl_loop_aff0 = (tl_wi_aff0 + (pos4 * 4)); const int tl_loop_aff1 = (((pos4 * 10240) + tl_wi_aff2) + tl_wi_aff3);
    float4 av_1[8];
    float8 bv_1[4];
    av_1[0] = (convert_float4(vload4(0, A + tl_loop_aff0)));
    av_1[1] = (convert_float4(vload4(0, A + (tl_loop_aff0 + 2560))));
    av_1[2] = (convert_float4(vload4(0, A + (tl_loop_aff0 + 5120))));
    av_1[3] = (convert_float4(vload4(0, A + (tl_loop_aff0 + 7680))));
    av_1[4] = (convert_float4(vload4(0, A + (tl_loop_aff0 + 10240))));
    av_1[5] = (convert_float4(vload4(0, A + (tl_loop_aff0 + 12800))));
    av_1[6] = (convert_float4(vload4(0, A + (tl_loop_aff0 + 15360))));
    av_1[7] = (convert_float4(vload4(0, A + (tl_loop_aff0 + 17920))));
    bv_1[0] = (convert_float8(vload8(0, B + (tl_loop_aff1 - tl_wi_aff1))));
    bv_1[1] = (convert_float8(vload8(0, B + ((tl_loop_aff1 + 2560) - tl_wi_aff1))));
    bv_1[2] = (convert_float8(vload8(0, B + ((tl_loop_aff1 + 5120) - tl_wi_aff1))));
    bv_1[3] = (convert_float8(vload8(0, B + ((tl_loop_aff1 + 7680) - tl_wi_aff1))));
    acc_1_0 = mad(((float8)((av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0, (av_1[0]).s0)), bv_1[0], mad(((float8)((av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1, (av_1[0]).s1)), bv_1[1], mad(((float8)((av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2, (av_1[0]).s2)), bv_1[2], mad(((float8)((av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3, (av_1[0]).s3)), bv_1[3], acc_1_0))));
    acc_1_1 = mad(((float8)((av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0, (av_1[1]).s0)), bv_1[0], mad(((float8)((av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1, (av_1[1]).s1)), bv_1[1], mad(((float8)((av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2, (av_1[1]).s2)), bv_1[2], mad(((float8)((av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3, (av_1[1]).s3)), bv_1[3], acc_1_1))));
    acc_1_2 = mad(((float8)((av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0, (av_1[2]).s0)), bv_1[0], mad(((float8)((av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1, (av_1[2]).s1)), bv_1[1], mad(((float8)((av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2, (av_1[2]).s2)), bv_1[2], mad(((float8)((av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3, (av_1[2]).s3)), bv_1[3], acc_1_2))));
    acc_1_3 = mad(((float8)((av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0, (av_1[3]).s0)), bv_1[0], mad(((float8)((av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1, (av_1[3]).s1)), bv_1[1], mad(((float8)((av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2, (av_1[3]).s2)), bv_1[2], mad(((float8)((av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3, (av_1[3]).s3)), bv_1[3], acc_1_3))));
    acc_1_4 = mad(((float8)((av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0, (av_1[4]).s0)), bv_1[0], mad(((float8)((av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1, (av_1[4]).s1)), bv_1[1], mad(((float8)((av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2, (av_1[4]).s2)), bv_1[2], mad(((float8)((av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3, (av_1[4]).s3)), bv_1[3], acc_1_4))));
    acc_1_5 = mad(((float8)((av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0, (av_1[5]).s0)), bv_1[0], mad(((float8)((av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1, (av_1[5]).s1)), bv_1[1], mad(((float8)((av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2, (av_1[5]).s2)), bv_1[2], mad(((float8)((av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3, (av_1[5]).s3)), bv_1[3], acc_1_5))));
    acc_1_6 = mad(((float8)((av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0, (av_1[6]).s0)), bv_1[0], mad(((float8)((av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1, (av_1[6]).s1)), bv_1[1], mad(((float8)((av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2, (av_1[6]).s2)), bv_1[2], mad(((float8)((av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3, (av_1[6]).s3)), bv_1[3], acc_1_6))));
    acc_1_7 = mad(((float8)((av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0, (av_1[7]).s0)), bv_1[0], mad(((float8)((av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1, (av_1[7]).s1)), bv_1[1], mad(((float8)((av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2, (av_1[7]).s2)), bv_1[2], mad(((float8)((av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3, (av_1[7]).s3)), bv_1[3], acc_1_7))));
  }
  C_frag_1[0] = (acc_1_0).s0;
  C_frag_1[1] = (acc_1_0).s1;
  C_frag_1[2] = (acc_1_0).s2;
  C_frag_1[3] = (acc_1_0).s3;
  C_frag_1[4] = (acc_1_0).s4;
  C_frag_1[5] = (acc_1_0).s5;
  C_frag_1[6] = (acc_1_0).s6;
  C_frag_1[7] = (acc_1_0).s7;
  C_frag_1[8] = (acc_1_1).s0;
  C_frag_1[9] = (acc_1_1).s1;
  C_frag_1[10] = (acc_1_1).s2;
  C_frag_1[11] = (acc_1_1).s3;
  C_frag_1[12] = (acc_1_1).s4;
  C_frag_1[13] = (acc_1_1).s5;
  C_frag_1[14] = (acc_1_1).s6;
  C_frag_1[15] = (acc_1_1).s7;
  C_frag_1[16] = (acc_1_2).s0;
  C_frag_1[17] = (acc_1_2).s1;
  C_frag_1[18] = (acc_1_2).s2;
  C_frag_1[19] = (acc_1_2).s3;
  C_frag_1[20] = (acc_1_2).s4;
  C_frag_1[21] = (acc_1_2).s5;
  C_frag_1[22] = (acc_1_2).s6;
  C_frag_1[23] = (acc_1_2).s7;
  C_frag_1[24] = (acc_1_3).s0;
  C_frag_1[25] = (acc_1_3).s1;
  C_frag_1[26] = (acc_1_3).s2;
  C_frag_1[27] = (acc_1_3).s3;
  C_frag_1[28] = (acc_1_3).s4;
  C_frag_1[29] = (acc_1_3).s5;
  C_frag_1[30] = (acc_1_3).s6;
  C_frag_1[31] = (acc_1_3).s7;
  C_frag_1[32] = (acc_1_4).s0;
  C_frag_1[33] = (acc_1_4).s1;
  C_frag_1[34] = (acc_1_4).s2;
  C_frag_1[35] = (acc_1_4).s3;
  C_frag_1[36] = (acc_1_4).s4;
  C_frag_1[37] = (acc_1_4).s5;
  C_frag_1[38] = (acc_1_4).s6;
  C_frag_1[39] = (acc_1_4).s7;
  C_frag_1[40] = (acc_1_5).s0;
  C_frag_1[41] = (acc_1_5).s1;
  C_frag_1[42] = (acc_1_5).s2;
  C_frag_1[43] = (acc_1_5).s3;
  C_frag_1[44] = (acc_1_5).s4;
  C_frag_1[45] = (acc_1_5).s5;
  C_frag_1[46] = (acc_1_5).s6;
  C_frag_1[47] = (acc_1_5).s7;
  C_frag_1[48] = (acc_1_6).s0;
  C_frag_1[49] = (acc_1_6).s1;
  C_frag_1[50] = (acc_1_6).s2;
  C_frag_1[51] = (acc_1_6).s3;
  C_frag_1[52] = (acc_1_6).s4;
  C_frag_1[53] = (acc_1_6).s5;
  C_frag_1[54] = (acc_1_6).s6;
  C_frag_1[55] = (acc_1_6).s7;
  C_frag_1[56] = (acc_1_7).s0;
  C_frag_1[57] = (acc_1_7).s1;
  C_frag_1[58] = (acc_1_7).s2;
  C_frag_1[59] = (acc_1_7).s3;
  C_frag_1[60] = (acc_1_7).s4;
  C_frag_1[61] = (acc_1_7).s5;
  C_frag_1[62] = (acc_1_7).s6;
  C_frag_1[63] = (acc_1_7).s7;
  for (int i = 0; i < 8; ++i) {
    half C_local_cast[8];
    for (int vec = 0; vec < 2; ++vec) {
      (*(half4*)(C_local_cast + (vec * 4))) = (convert_half4((*(float4*)(C_frag_1 + ((i * 8) + (vec * 4))))));
    }
    vstore8((*(half8*)(C_local_cast + 0)), 0, C + (((tl_wi_aff0 + (i * 2560)) + tl_wi_aff2) + ((tl_lid0 & 7) * 8)));
  }
}

