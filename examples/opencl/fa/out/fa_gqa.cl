// Function: fa_gqa_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void fa_gqa_kernel_kernel(__global half* restrict K, __global half* restrict O, __global half* restrict Q, __global half* restrict V);
__kernel void fa_gqa_kernel_kernel(__global half* restrict K, __global half* restrict O, __global half* restrict Q, __global half* restrict V) { const int tl_gid0 = convert_int(get_group_id(0)); const int tl_gid1 = convert_int(get_group_id(1)); const int tl_lid0 = convert_int(get_local_id(0)); const int tl_wi_aff0 = ((tl_gid0 * 16) + ((tl_lid0 >> 5) * 2)); const int tl_wi_aff1 = ((tl_lid0 >> 4) * 256); const int tl_wi_aff2 = ((tl_lid0 >> 2) * 256); const int tl_wi_aff3 = ((tl_lid0 >> 4) * 512); const int tl_wi_aff4 = ((tl_lid0 >> 5) * 512); const int tl_wi_aff5 = ((tl_lid0 >> 4) * 2); const int tl_wi_aff6 = ((tl_lid0 & 3) * 128); const int tl_wi_aff7 = ((tl_lid0 & 15) * 16); const int tl_wi_aff8 = ((tl_lid0 & 3) * 512); const int tl_wi_aff9 = ((tl_lid0 & 31) * 2); const int tl_wi_aff10 = ((tl_lid0 & 31) * 4); const int tl_wi_aff11 = ((tl_lid0 & 3) * 4); const int tl_wi_aff12 = (tl_gid1 * 262144); const int tl_wi_aff13 = (tl_gid0 * 8192); const int tl_wi_aff14 = (tl_gid0 * 8); const int tl_wi_aff15 = (tl_lid0 * 128);
__local float alpha[32];
__local float lrun[32];
__local float mrun[32];
__local half Sh[4096];
__local half Ks[2048];
__local half Qs[512];
  float4 acc_o_1_0_lo = (float4)(0.000000e+00f);
  float4 acc_o_1_0_hi = (float4)(0.000000e+00f);
  float4 acc_o_1_1_lo = (float4)(0.000000e+00f);
  float4 acc_o_1_1_hi = (float4)(0.000000e+00f);
  float4 acc_o_1_2_lo = (float4)(0.000000e+00f);
  float4 acc_o_1_2_hi = (float4)(0.000000e+00f);
  float4 acc_o_1_3_lo = (float4)(0.000000e+00f);
  float4 acc_o_1_3_hi = (float4)(0.000000e+00f);
  if (tl_lid0 < 32) {
    mrun[tl_lid0] = -3.276800e+04f;
    lrun[tl_lid0] = 0.000000e+00f;
    alpha[tl_lid0] = 1.000000e+00f;
  }
  float broadcast_var = 0.000000e+00f;
  acc_o_1_0_lo = (((float4)(broadcast_var, broadcast_var, broadcast_var, broadcast_var)));
  float broadcast_var_1 = 0.000000e+00f;
  acc_o_1_0_hi = (((float4)(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1)));
  float broadcast_var_2 = 0.000000e+00f;
  acc_o_1_1_lo = (((float4)(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2)));
  float broadcast_var_3 = 0.000000e+00f;
  acc_o_1_1_hi = (((float4)(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3)));
  float broadcast_var_4 = 0.000000e+00f;
  acc_o_1_2_lo = (((float4)(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4)));
  float broadcast_var_5 = 0.000000e+00f;
  acc_o_1_2_hi = (((float4)(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5)));
  float broadcast_var_6 = 0.000000e+00f;
  acc_o_1_3_lo = (((float4)(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6)));
  float broadcast_var_7 = 0.000000e+00f;
  acc_o_1_3_hi = (((float4)(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7)));
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int ks = 0; ks < ((tl_gid0 >> 2) + 1); ++ks) {
    float4 acc_s_1_0_lo = (float4)(0.000000e+00f);
    float4 acc_s_1_0_hi = (float4)(0.000000e+00f);
    float4 acc_s_1_1_lo = (float4)(0.000000e+00f);
    float4 acc_s_1_1_hi = (float4)(0.000000e+00f);
    float broadcast_var_8 = 0.000000e+00f;
    acc_s_1_0_lo = (((float4)(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8)));
    float broadcast_var_9 = 0.000000e+00f;
    acc_s_1_0_hi = (((float4)(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9)));
    float broadcast_var_10 = 0.000000e+00f;
    acc_s_1_1_lo = (((float4)(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10)));
    float broadcast_var_11 = 0.000000e+00f;
    acc_s_1_1_hi = (((float4)(broadcast_var_11, broadcast_var_11, broadcast_var_11, broadcast_var_11)));
    for (int _tmp = 0; _tmp < 16; ++_tmp) {
      if (tl_lid0 < 128) {
        half4 qtmp_1_v = vload4(0, Q + ((((tl_wi_aff12 + tl_wi_aff13) + tl_wi_aff2) + (_tmp * 16)) + tl_wi_aff11));
                Qs[(tl_wi_aff6 + (tl_lid0 >> 2))] = qtmp_1_v.s0;
        Qs[((tl_wi_aff6 + (tl_lid0 >> 2)) + 32)] = qtmp_1_v.s1;
        Qs[((tl_wi_aff6 + (tl_lid0 >> 2)) + 64)] = qtmp_1_v.s2;
        Qs[((tl_wi_aff6 + (tl_lid0 >> 2)) + 96)] = qtmp_1_v.s3;
      }
      half4 ktmp_1_v = vload4(0, K + ((((((tl_gid1 >> 2) * 262144) + (ks * 32768)) + tl_wi_aff2) + (_tmp * 16)) + tl_wi_aff11));
            Ks[(tl_wi_aff8 + (tl_lid0 >> 2))] = ktmp_1_v.s0;
      Ks[((tl_wi_aff8 + (tl_lid0 >> 2)) + 128)] = ktmp_1_v.s1;
      Ks[((tl_wi_aff8 + (tl_lid0 >> 2)) + 256)] = ktmp_1_v.s2;
      Ks[((tl_wi_aff8 + (tl_lid0 >> 2)) + 384)] = ktmp_1_v.s3;
      half4 ktmp_2_v = vload4(0, K + (((((((tl_gid1 >> 2) * 262144) + (ks * 32768)) + tl_wi_aff2) + (_tmp * 16)) + tl_wi_aff11) + 16384));
            Ks[((tl_wi_aff8 + (tl_lid0 >> 2)) + 64)] = ktmp_2_v.s0;
      Ks[((tl_wi_aff8 + (tl_lid0 >> 2)) + 192)] = ktmp_2_v.s1;
      Ks[((tl_wi_aff8 + (tl_lid0 >> 2)) + 320)] = ktmp_2_v.s2;
      Ks[((tl_wi_aff8 + (tl_lid0 >> 2)) + 448)] = ktmp_2_v.s3;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k = 0; k < 16; ++k) {
        half4 q4_1_v = vload4(0, Qs + ((k * 32) + ((tl_lid0 >> 5) * 4)));
        half4 k4_1_v = vload4(0, Ks + ((k * 128) + tl_wi_aff10));
                        acc_s_1_0_lo = ((((float4)(convert_float(q4_1_v.s0)) * (convert_float4(k4_1_v))) + acc_s_1_0_lo));
        acc_s_1_0_hi = ((((float4)(convert_float(q4_1_v.s1)) * (convert_float4(k4_1_v))) + acc_s_1_0_hi));
        acc_s_1_1_lo = ((((float4)(convert_float(q4_1_v.s2)) * (convert_float4(k4_1_v))) + acc_s_1_1_lo));
        acc_s_1_1_hi = ((((float4)(convert_float(q4_1_v.s3)) * (convert_float4(k4_1_v))) + acc_s_1_1_hi));
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    float condval;
    if (((((tl_wi_aff14 + (tl_lid0 >> 5)) - (tl_lid0 & 31)) - (ks * 32)) < 0)) {
      condval = -3.276800e+04f;
    } else {
      condval = (acc_s_1_0_lo.s0 * 6.250000e-02f);
    }
    Sh[(tl_wi_aff4 + tl_wi_aff10)] = (convert_half(condval));
    float condval_1;
    if (((tl_wi_aff14 + (tl_lid0 >> 5)) <= ((ks * 32) + (tl_lid0 & 31)))) {
      condval_1 = -3.276800e+04f;
    } else {
      condval_1 = (acc_s_1_0_lo.s1 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 1)] = (convert_half(condval_1));
    float condval_2;
    if ((((tl_wi_aff0 - tl_wi_aff9) - (ks * 64)) < 1)) {
      condval_2 = -3.276800e+04f;
    } else {
      condval_2 = (acc_s_1_0_lo.s2 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 2)] = (convert_half(condval_2));
    float condval_3;
    if (((tl_wi_aff14 + (tl_lid0 >> 5)) <= ((ks * 32) + (tl_lid0 & 31)))) {
      condval_3 = -3.276800e+04f;
    } else {
      condval_3 = (acc_s_1_0_lo.s3 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 3)] = (convert_half(condval_3));
    float condval_4;
    if (((((tl_wi_aff14 + (tl_lid0 >> 5)) - (tl_lid0 & 31)) - (ks * 32)) < 0)) {
      condval_4 = -3.276800e+04f;
    } else {
      condval_4 = (acc_s_1_0_hi.s0 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 128)] = (convert_half(condval_4));
    float condval_5;
    if (((((tl_wi_aff14 + (tl_lid0 >> 5)) - (tl_lid0 & 31)) - (ks * 32)) < 0)) {
      condval_5 = -3.276800e+04f;
    } else {
      condval_5 = (acc_s_1_0_hi.s1 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 129)] = (convert_half(condval_5));
    float condval_6;
    if ((((tl_wi_aff0 - tl_wi_aff9) - (ks * 64)) < 1)) {
      condval_6 = -3.276800e+04f;
    } else {
      condval_6 = (acc_s_1_0_hi.s2 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 130)] = (convert_half(condval_6));
    float condval_7;
    if ((((tl_wi_aff0 - tl_wi_aff9) - (ks * 64)) < 1)) {
      condval_7 = -3.276800e+04f;
    } else {
      condval_7 = (acc_s_1_0_hi.s3 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 131)] = (convert_half(condval_7));
    float condval_8;
    if ((((tl_wi_aff0 - tl_wi_aff9) - (ks * 64)) < -1)) {
      condval_8 = -3.276800e+04f;
    } else {
      condval_8 = (acc_s_1_1_lo.s0 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 256)] = (convert_half(condval_8));
    float condval_9;
    if (((((tl_wi_aff14 + (tl_lid0 >> 5)) - (tl_lid0 & 31)) - (ks * 32)) < 0)) {
      condval_9 = -3.276800e+04f;
    } else {
      condval_9 = (acc_s_1_1_lo.s1 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 257)] = (convert_half(condval_9));
    float condval_10;
    if (((((tl_wi_aff14 + (tl_lid0 >> 5)) - (tl_lid0 & 31)) - (ks * 32)) < 0)) {
      condval_10 = -3.276800e+04f;
    } else {
      condval_10 = (acc_s_1_1_lo.s2 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 258)] = (convert_half(condval_10));
    float condval_11;
    if (((tl_wi_aff14 + (tl_lid0 >> 5)) <= ((ks * 32) + (tl_lid0 & 31)))) {
      condval_11 = -3.276800e+04f;
    } else {
      condval_11 = (acc_s_1_1_lo.s3 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 259)] = (convert_half(condval_11));
    float condval_12;
    if (((((tl_wi_aff14 + (tl_lid0 >> 5)) - (tl_lid0 & 31)) - (ks * 32)) < 0)) {
      condval_12 = -3.276800e+04f;
    } else {
      condval_12 = (acc_s_1_1_hi.s0 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 384)] = (convert_half(condval_12));
    float condval_13;
    if ((((tl_wi_aff0 - tl_wi_aff9) - (ks * 64)) < -1)) {
      condval_13 = -3.276800e+04f;
    } else {
      condval_13 = (acc_s_1_1_hi.s1 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 385)] = (convert_half(condval_13));
    float condval_14;
    if (((((tl_wi_aff14 + (tl_lid0 >> 5)) - (tl_lid0 & 31)) - (ks * 32)) < 0)) {
      condval_14 = -3.276800e+04f;
    } else {
      condval_14 = (acc_s_1_1_hi.s2 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 386)] = (convert_half(condval_14));
    float condval_15;
    if (((((tl_wi_aff14 + (tl_lid0 >> 5)) - (tl_lid0 & 31)) - (ks * 32)) < 0)) {
      condval_15 = -3.276800e+04f;
    } else {
      condval_15 = (acc_s_1_1_hi.s3 * 6.250000e-02f);
    }
    Sh[((tl_wi_aff4 + tl_wi_aff10) + 387)] = (convert_half(condval_15));
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tl_lid0 < 32) {
      float m_old_1[1];
      float m_new_1[1];
      float l_t_1[1];
      m_old_1[0] = mrun[tl_lid0];
      m_new_1[0] = m_old_1[0];
      for (int c = 0; c < 128; ++c) {
        m_new_1[0] = max((float)m_new_1[0], (float)(convert_float(Sh[(tl_wi_aff15 + c)])));
      }
      alpha[tl_lid0] = exp((m_old_1[0] - m_new_1[0]));
      l_t_1[0] = 0.000000e+00f;
      for (int c_1 = 0; c_1 < 128; ++c_1) {
        float p = exp(((convert_float(Sh[(tl_wi_aff15 + c_1)])) - m_new_1[0]));
        Sh[(tl_wi_aff15 + c_1)] = (convert_half(p));
        l_t_1[0] = (l_t_1[0] + p);
      }
      lrun[tl_lid0] = ((lrun[tl_lid0] * alpha[tl_lid0]) + l_t_1[0]);
      mrun[tl_lid0] = m_new_1[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    acc_o_1_0_lo = ((acc_o_1_0_lo * ((float4)(alpha[tl_wi_aff5], alpha[tl_wi_aff5], alpha[tl_wi_aff5], alpha[tl_wi_aff5]))));
    acc_o_1_0_hi = ((acc_o_1_0_hi * ((float4)(alpha[tl_wi_aff5], alpha[tl_wi_aff5], alpha[tl_wi_aff5], alpha[tl_wi_aff5]))));
    acc_o_1_1_lo = ((acc_o_1_1_lo * ((float4)(alpha[tl_wi_aff5], alpha[tl_wi_aff5], alpha[tl_wi_aff5], alpha[tl_wi_aff5]))));
    acc_o_1_1_hi = ((acc_o_1_1_hi * ((float4)(alpha[tl_wi_aff5], alpha[tl_wi_aff5], alpha[tl_wi_aff5], alpha[tl_wi_aff5]))));
    acc_o_1_2_lo = ((acc_o_1_2_lo * ((float4)(alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)]))));
    acc_o_1_2_hi = ((acc_o_1_2_hi * ((float4)(alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)]))));
    acc_o_1_3_lo = ((acc_o_1_3_lo * ((float4)(alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)]))));
    acc_o_1_3_hi = ((acc_o_1_3_hi * ((float4)(alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)], alpha[(tl_wi_aff5 + 1)]))));
    for (int j = 0; j < 128; ++j) { const int tl_loop_aff0 = (j * 256);
      half8 vtmp_1_lo = vload8(0, V + (((((tl_gid1 >> 2) * 262144) + (ks * 32768)) + tl_loop_aff0) + tl_wi_aff7));
half8 vtmp_1_hi = vload8(0, V + ((((((tl_gid1 >> 2) * 262144) + (ks * 32768)) + tl_loop_aff0) + tl_wi_aff7) + 8));
                              acc_o_1_0_lo = ((((float4)(convert_float(Sh[(tl_wi_aff1 + j)])) * (convert_float4(vtmp_1_lo.lo))) + acc_o_1_0_lo));
      acc_o_1_0_hi = ((((float4)(convert_float(Sh[(tl_wi_aff1 + j)])) * (convert_float4(vtmp_1_lo.hi))) + acc_o_1_0_hi));
      acc_o_1_1_lo = ((((float4)(convert_float(Sh[(tl_wi_aff1 + j)])) * (convert_float4(vtmp_1_hi.lo))) + acc_o_1_1_lo));
      acc_o_1_1_hi = ((((float4)(convert_float(Sh[(tl_wi_aff1 + j)])) * (convert_float4(vtmp_1_hi.hi))) + acc_o_1_1_hi));
      acc_o_1_2_lo = ((((float4)(convert_float(Sh[((tl_wi_aff1 + j) + 128)])) * (convert_float4(vtmp_1_lo.lo))) + acc_o_1_2_lo));
      acc_o_1_2_hi = ((((float4)(convert_float(Sh[((tl_wi_aff1 + j) + 128)])) * (convert_float4(vtmp_1_lo.hi))) + acc_o_1_2_hi));
      acc_o_1_3_lo = ((((float4)(convert_float(Sh[((tl_wi_aff1 + j) + 128)])) * (convert_float4(vtmp_1_hi.lo))) + acc_o_1_3_lo));
      acc_o_1_3_hi = ((((float4)(convert_float(Sh[((tl_wi_aff1 + j) + 128)])) * (convert_float4(vtmp_1_hi.hi))) + acc_o_1_3_hi));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  half O_local_cast[8];
  (*(half4*)(O_local_cast + 0)) = (convert_half4((acc_o_1_0_lo / ((float4)(lrun[tl_wi_aff5], lrun[tl_wi_aff5], lrun[tl_wi_aff5], lrun[tl_wi_aff5])))));
  (*(half4*)(O_local_cast + 4)) = (convert_half4((acc_o_1_0_hi / ((float4)(lrun[tl_wi_aff5], lrun[tl_wi_aff5], lrun[tl_wi_aff5], lrun[tl_wi_aff5])))));
  vstore8((*(half8*)(O_local_cast + 0)), 0, O + (((tl_wi_aff12 + tl_wi_aff13) + tl_wi_aff3) + tl_wi_aff7));
  half O_local_cast_1[8];
  (*(half4*)(O_local_cast_1 + 0)) = (convert_half4((acc_o_1_1_lo / ((float4)(lrun[tl_wi_aff5], lrun[tl_wi_aff5], lrun[tl_wi_aff5], lrun[tl_wi_aff5])))));
  (*(half4*)(O_local_cast_1 + 4)) = (convert_half4((acc_o_1_1_hi / ((float4)(lrun[tl_wi_aff5], lrun[tl_wi_aff5], lrun[tl_wi_aff5], lrun[tl_wi_aff5])))));
  vstore8((*(half8*)(O_local_cast_1 + 0)), 0, O + ((((tl_wi_aff12 + tl_wi_aff13) + tl_wi_aff3) + tl_wi_aff7) + 8));
  half O_local_cast_2[8];
  (*(half4*)(O_local_cast_2 + 0)) = (convert_half4((acc_o_1_2_lo / ((float4)(lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)])))));
  (*(half4*)(O_local_cast_2 + 4)) = (convert_half4((acc_o_1_2_hi / ((float4)(lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)])))));
  vstore8((*(half8*)(O_local_cast_2 + 0)), 0, O + ((((tl_wi_aff12 + tl_wi_aff13) + tl_wi_aff3) + tl_wi_aff7) + 256));
  half O_local_cast_3[8];
  (*(half4*)(O_local_cast_3 + 0)) = (convert_half4((acc_o_1_3_lo / ((float4)(lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)])))));
  (*(half4*)(O_local_cast_3 + 4)) = (convert_half4((acc_o_1_3_hi / ((float4)(lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)], lrun[(tl_wi_aff5 + 1)])))));
  vstore8((*(half8*)(O_local_cast_3 + 0)), 0, O + ((((tl_wi_aff12 + tl_wi_aff13) + tl_wi_aff3) + tl_wi_aff7) + 264));
}

