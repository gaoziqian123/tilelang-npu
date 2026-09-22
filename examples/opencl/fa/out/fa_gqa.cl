// Function: fa_gqa_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void fa_gqa_kernel_kernel(__global half* restrict K, __global half* restrict O, __global half* restrict Q, __global half* restrict V);
__kernel void fa_gqa_kernel_kernel(__global half* restrict K, __global half* restrict O, __global half* restrict Q, __global half* restrict V) {
  __local uchar buf_dyn_shmem[13696];
  void* alpha = ((void*)((char*)buf_dyn_shmem + 0));
  void* lrun = ((void*)((char*)buf_dyn_shmem + 128));
  void* mrun = ((void*)((char*)buf_dyn_shmem + 256));
  void* Sh = ((void*)((char*)buf_dyn_shmem + 384));
  void* Ks = ((void*)((char*)buf_dyn_shmem + 8576));
  void* Qs = ((void*)((char*)buf_dyn_shmem + 12672));
  float4 acc_o_0_lo = (float4)(0.000000e+00f);
  float4 acc_o_0_hi = (float4)(0.000000e+00f);
  float4 acc_o_1_lo = (float4)(0.000000e+00f);
  float4 acc_o_1_hi = (float4)(0.000000e+00f);
  float4 acc_o_2_lo = (float4)(0.000000e+00f);
  float4 acc_o_2_hi = (float4)(0.000000e+00f);
  float4 acc_o_3_lo = (float4)(0.000000e+00f);
  float4 acc_o_3_hi = (float4)(0.000000e+00f);
  float4 acc_s_0_lo = (float4)(0.000000e+00f);
  float4 acc_s_0_hi = (float4)(0.000000e+00f);
  float4 acc_s_1_lo = (float4)(0.000000e+00f);
  float4 acc_s_1_hi = (float4)(0.000000e+00f);
  half qtmp[4];
  half ktmp[4];
  half q4[4];
  half k4[4];
  float m_old[1];
  float m_new[1];
  float l_t[1];
  float4 acc_o_1_0_lo = (float4)(0.000000e+00f);
  float4 acc_o_1_0_hi = (float4)(0.000000e+00f);
  float4 acc_o_1_1_lo = (float4)(0.000000e+00f);
  float4 acc_o_1_1_hi = (float4)(0.000000e+00f);
  float4 acc_o_1_2_lo = (float4)(0.000000e+00f);
  float4 acc_o_1_2_hi = (float4)(0.000000e+00f);
  float4 acc_o_1_3_lo = (float4)(0.000000e+00f);
  float4 acc_o_1_3_hi = (float4)(0.000000e+00f);
  if ((convert_int(get_local_id(0))) < 32) {
    ((float*)mrun)[(convert_int(get_local_id(0)))] = -3.276800e+04f;
    ((float*)lrun)[(convert_int(get_local_id(0)))] = 0.000000e+00f;
    ((float*)alpha)[(convert_int(get_local_id(0)))] = 1.000000e+00f;
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
  for (int ks = 0; ks < (((convert_int(get_group_id(0))) >> 2) + 1); ++ks) {
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
      if ((convert_int(get_local_id(0))) < 128) {
        half4 qtmp_1_v = vload4(0, Q + ((((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + (((convert_int(get_local_id(0))) >> 2) * 256)) + (_tmp * 16)) + (((convert_int(get_local_id(0))) & 3) * 4)));
                ((half*)Qs)[((((convert_int(get_local_id(0))) & 3) * 128) + ((convert_int(get_local_id(0))) >> 2))] = qtmp_1_v.s0;
        ((half*)Qs)[(((((convert_int(get_local_id(0))) & 3) * 128) + ((convert_int(get_local_id(0))) >> 2)) + 32)] = qtmp_1_v.s1;
        ((half*)Qs)[(((((convert_int(get_local_id(0))) & 3) * 128) + ((convert_int(get_local_id(0))) >> 2)) + 64)] = qtmp_1_v.s2;
        ((half*)Qs)[(((((convert_int(get_local_id(0))) & 3) * 128) + ((convert_int(get_local_id(0))) >> 2)) + 96)] = qtmp_1_v.s3;
      }
      half4 ktmp_1_v = vload4(0, K + (((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (((convert_int(get_local_id(0))) >> 2) * 256)) + (_tmp * 16)) + (((convert_int(get_local_id(0))) & 3) * 4)));
            ((half*)Ks)[((((convert_int(get_local_id(0))) & 3) * 512) + ((convert_int(get_local_id(0))) >> 2))] = ktmp_1_v.s0;
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 3) * 512) + ((convert_int(get_local_id(0))) >> 2)) + 128)] = ktmp_1_v.s1;
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 3) * 512) + ((convert_int(get_local_id(0))) >> 2)) + 256)] = ktmp_1_v.s2;
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 3) * 512) + ((convert_int(get_local_id(0))) >> 2)) + 384)] = ktmp_1_v.s3;
      half4 ktmp_2_v = vload4(0, K + ((((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (((convert_int(get_local_id(0))) >> 2) * 256)) + (_tmp * 16)) + (((convert_int(get_local_id(0))) & 3) * 4)) + 16384));
            ((half*)Ks)[(((((convert_int(get_local_id(0))) & 3) * 512) + ((convert_int(get_local_id(0))) >> 2)) + 64)] = ktmp_2_v.s0;
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 3) * 512) + ((convert_int(get_local_id(0))) >> 2)) + 192)] = ktmp_2_v.s1;
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 3) * 512) + ((convert_int(get_local_id(0))) >> 2)) + 320)] = ktmp_2_v.s2;
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 3) * 512) + ((convert_int(get_local_id(0))) >> 2)) + 448)] = ktmp_2_v.s3;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k = 0; k < 16; ++k) {
        half4 q4_1_v = vload4(0, (half*)Qs + ((k * 32) + (((convert_int(get_local_id(0))) >> 5) * 4)));
        half4 k4_1_v = vload4(0, (half*)Ks + ((k * 128) + (((convert_int(get_local_id(0))) & 31) * 4)));
                        acc_s_1_0_lo = ((((float4)(convert_float(q4_1_v.s0)) * (convert_float4(k4_1_v))) + acc_s_1_0_lo));
        acc_s_1_0_hi = ((((float4)(convert_float(q4_1_v.s1)) * (convert_float4(k4_1_v))) + acc_s_1_0_hi));
        acc_s_1_1_lo = ((((float4)(convert_float(q4_1_v.s2)) * (convert_float4(k4_1_v))) + acc_s_1_1_lo));
        acc_s_1_1_hi = ((((float4)(convert_float(q4_1_v.s3)) * (convert_float4(k4_1_v))) + acc_s_1_1_hi));
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    float condval;
    if (((((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) - ((convert_int(get_local_id(0))) & 31)) - (ks * 32)) < 0)) {
      condval = -3.276800e+04f;
    } else {
      condval = (acc_s_1_0_lo.s0 * 6.250000e-02f);
    }
    ((half*)Sh)[((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4))] = (convert_half(condval));
    float condval_1;
    if (((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) <= ((ks * 32) + ((convert_int(get_local_id(0))) & 31)))) {
      condval_1 = -3.276800e+04f;
    } else {
      condval_1 = (acc_s_1_0_lo.s1 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 1)] = (convert_half(condval_1));
    float condval_2;
    if (((((((convert_int(get_group_id(0))) * 16) + (((convert_int(get_local_id(0))) >> 5) * 2)) - (((convert_int(get_local_id(0))) & 31) * 2)) - (ks * 64)) < 1)) {
      condval_2 = -3.276800e+04f;
    } else {
      condval_2 = (acc_s_1_0_lo.s2 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 2)] = (convert_half(condval_2));
    float condval_3;
    if (((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) <= ((ks * 32) + ((convert_int(get_local_id(0))) & 31)))) {
      condval_3 = -3.276800e+04f;
    } else {
      condval_3 = (acc_s_1_0_lo.s3 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 3)] = (convert_half(condval_3));
    float condval_4;
    if (((((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) - ((convert_int(get_local_id(0))) & 31)) - (ks * 32)) < 0)) {
      condval_4 = -3.276800e+04f;
    } else {
      condval_4 = (acc_s_1_0_hi.s0 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 128)] = (convert_half(condval_4));
    float condval_5;
    if (((((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) - ((convert_int(get_local_id(0))) & 31)) - (ks * 32)) < 0)) {
      condval_5 = -3.276800e+04f;
    } else {
      condval_5 = (acc_s_1_0_hi.s1 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 129)] = (convert_half(condval_5));
    float condval_6;
    if (((((((convert_int(get_group_id(0))) * 16) + (((convert_int(get_local_id(0))) >> 5) * 2)) - (((convert_int(get_local_id(0))) & 31) * 2)) - (ks * 64)) < 1)) {
      condval_6 = -3.276800e+04f;
    } else {
      condval_6 = (acc_s_1_0_hi.s2 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 130)] = (convert_half(condval_6));
    float condval_7;
    if (((((((convert_int(get_group_id(0))) * 16) + (((convert_int(get_local_id(0))) >> 5) * 2)) - (((convert_int(get_local_id(0))) & 31) * 2)) - (ks * 64)) < 1)) {
      condval_7 = -3.276800e+04f;
    } else {
      condval_7 = (acc_s_1_0_hi.s3 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 131)] = (convert_half(condval_7));
    float condval_8;
    if (((((((convert_int(get_group_id(0))) * 16) + (((convert_int(get_local_id(0))) >> 5) * 2)) - (((convert_int(get_local_id(0))) & 31) * 2)) - (ks * 64)) < -1)) {
      condval_8 = -3.276800e+04f;
    } else {
      condval_8 = (acc_s_1_1_lo.s0 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 256)] = (convert_half(condval_8));
    float condval_9;
    if (((((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) - ((convert_int(get_local_id(0))) & 31)) - (ks * 32)) < 0)) {
      condval_9 = -3.276800e+04f;
    } else {
      condval_9 = (acc_s_1_1_lo.s1 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 257)] = (convert_half(condval_9));
    float condval_10;
    if (((((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) - ((convert_int(get_local_id(0))) & 31)) - (ks * 32)) < 0)) {
      condval_10 = -3.276800e+04f;
    } else {
      condval_10 = (acc_s_1_1_lo.s2 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 258)] = (convert_half(condval_10));
    float condval_11;
    if (((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) <= ((ks * 32) + ((convert_int(get_local_id(0))) & 31)))) {
      condval_11 = -3.276800e+04f;
    } else {
      condval_11 = (acc_s_1_1_lo.s3 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 259)] = (convert_half(condval_11));
    float condval_12;
    if (((((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) - ((convert_int(get_local_id(0))) & 31)) - (ks * 32)) < 0)) {
      condval_12 = -3.276800e+04f;
    } else {
      condval_12 = (acc_s_1_1_hi.s0 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 384)] = (convert_half(condval_12));
    float condval_13;
    if (((((((convert_int(get_group_id(0))) * 16) + (((convert_int(get_local_id(0))) >> 5) * 2)) - (((convert_int(get_local_id(0))) & 31) * 2)) - (ks * 64)) < -1)) {
      condval_13 = -3.276800e+04f;
    } else {
      condval_13 = (acc_s_1_1_hi.s1 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 385)] = (convert_half(condval_13));
    float condval_14;
    if (((((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) - ((convert_int(get_local_id(0))) & 31)) - (ks * 32)) < 0)) {
      condval_14 = -3.276800e+04f;
    } else {
      condval_14 = (acc_s_1_1_hi.s2 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 386)] = (convert_half(condval_14));
    float condval_15;
    if (((((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 5)) - ((convert_int(get_local_id(0))) & 31)) - (ks * 32)) < 0)) {
      condval_15 = -3.276800e+04f;
    } else {
      condval_15 = (acc_s_1_1_hi.s3 * 6.250000e-02f);
    }
    ((half*)Sh)[(((((convert_int(get_local_id(0))) >> 5) * 512) + (((convert_int(get_local_id(0))) & 31) * 4)) + 387)] = (convert_half(condval_15));
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((convert_int(get_local_id(0))) < 32) {
      float m_old_1[1];
      float m_new_1[1];
      float l_t_1[1];
      m_old_1[0] = ((float*)mrun)[(convert_int(get_local_id(0)))];
      m_new_1[0] = m_old_1[0];
      for (int c = 0; c < 128; ++c) {
        m_new_1[0] = max((float)m_new_1[0], (float)(convert_float(((half*)Sh)[(((convert_int(get_local_id(0))) * 128) + c)])));
      }
      ((float*)alpha)[(convert_int(get_local_id(0)))] = exp((m_old_1[0] - m_new_1[0]));
      l_t_1[0] = 0.000000e+00f;
      for (int c_1 = 0; c_1 < 128; ++c_1) {
        float p = exp(((convert_float(((half*)Sh)[(((convert_int(get_local_id(0))) * 128) + c_1)])) - m_new_1[0]));
        ((half*)Sh)[(((convert_int(get_local_id(0))) * 128) + c_1)] = (convert_half(p));
        l_t_1[0] = (l_t_1[0] + p);
      }
      ((float*)lrun)[(convert_int(get_local_id(0)))] = ((((float*)lrun)[(convert_int(get_local_id(0)))] * ((float*)alpha)[(convert_int(get_local_id(0)))]) + l_t_1[0]);
      ((float*)mrun)[(convert_int(get_local_id(0)))] = m_new_1[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    acc_o_1_0_lo = ((acc_o_1_0_lo * ((float4)(((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)]))));
    acc_o_1_0_hi = ((acc_o_1_0_hi * ((float4)(((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)]))));
    acc_o_1_1_lo = ((acc_o_1_1_lo * ((float4)(((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)]))));
    acc_o_1_1_hi = ((acc_o_1_1_hi * ((float4)(((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)]))));
    acc_o_1_2_lo = ((acc_o_1_2_lo * ((float4)(((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)]))));
    acc_o_1_2_hi = ((acc_o_1_2_hi * ((float4)(((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)]))));
    acc_o_1_3_lo = ((acc_o_1_3_lo * ((float4)(((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)]))));
    acc_o_1_3_hi = ((acc_o_1_3_hi * ((float4)(((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 3)]))));
    for (int j = 0; j < 128; ++j) {
      half8 V_local_cast_v = vload8(0, V + ((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 7) * 32)));
            acc_o_1_0_lo = ((((float4)(convert_float(((half*)Sh)[((((convert_int(get_local_id(0))) >> 3) * 128) + j)])) * (convert_float4(V_local_cast_v.lo))) + acc_o_1_0_lo));
      acc_o_1_0_hi = ((((float4)(convert_float(((half*)Sh)[((((convert_int(get_local_id(0))) >> 3) * 128) + j)])) * (convert_float4(V_local_cast_v.hi))) + acc_o_1_0_hi));
      half8 V_local_cast_1_v = vload8(0, V + (((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 7) * 32)) + 8));
            acc_o_1_1_lo = ((((float4)(convert_float(((half*)Sh)[((((convert_int(get_local_id(0))) >> 3) * 128) + j)])) * (convert_float4(V_local_cast_1_v.lo))) + acc_o_1_1_lo));
      acc_o_1_1_hi = ((((float4)(convert_float(((half*)Sh)[((((convert_int(get_local_id(0))) >> 3) * 128) + j)])) * (convert_float4(V_local_cast_1_v.hi))) + acc_o_1_1_hi));
      half8 V_local_cast_2_v = vload8(0, V + (((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 7) * 32)) + 16));
            acc_o_1_2_lo = ((((float4)(convert_float(((half*)Sh)[((((convert_int(get_local_id(0))) >> 3) * 128) + j)])) * (convert_float4(V_local_cast_2_v.lo))) + acc_o_1_2_lo));
      acc_o_1_2_hi = ((((float4)(convert_float(((half*)Sh)[((((convert_int(get_local_id(0))) >> 3) * 128) + j)])) * (convert_float4(V_local_cast_2_v.hi))) + acc_o_1_2_hi));
      half8 V_local_cast_3_v = vload8(0, V + (((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 7) * 32)) + 24));
            acc_o_1_3_lo = ((((float4)(convert_float(((half*)Sh)[((((convert_int(get_local_id(0))) >> 3) * 128) + j)])) * (convert_float4(V_local_cast_3_v.lo))) + acc_o_1_3_lo));
      acc_o_1_3_hi = ((((float4)(convert_float(((half*)Sh)[((((convert_int(get_local_id(0))) >> 3) * 128) + j)])) * (convert_float4(V_local_cast_3_v.hi))) + acc_o_1_3_hi));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  half O_local_cast_1[8];
  (*(half4*)(O_local_cast_1 + 0)) = (convert_half4((acc_o_1_0_lo / ((float4)(((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)])))));
  (*(half4*)(O_local_cast_1 + 4)) = (convert_half4((acc_o_1_0_hi / ((float4)(((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)])))));
  vstore8((*(half8*)(O_local_cast_1 + 0)), 0, O + ((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 32)));
  half O_local_cast_1_1[8];
  (*(half4*)(O_local_cast_1_1 + 0)) = (convert_half4((acc_o_1_1_lo / ((float4)(((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)])))));
  (*(half4*)(O_local_cast_1_1 + 4)) = (convert_half4((acc_o_1_1_hi / ((float4)(((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)])))));
  vstore8((*(half8*)(O_local_cast_1_1 + 0)), 0, O + (((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 32)) + 8));
  half O_local_cast_1_2[8];
  (*(half4*)(O_local_cast_1_2 + 0)) = (convert_half4((acc_o_1_2_lo / ((float4)(((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)])))));
  (*(half4*)(O_local_cast_1_2 + 4)) = (convert_half4((acc_o_1_2_hi / ((float4)(((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)])))));
  vstore8((*(half8*)(O_local_cast_1_2 + 0)), 0, O + (((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 32)) + 16));
  half O_local_cast_1_3[8];
  (*(half4*)(O_local_cast_1_3 + 0)) = (convert_half4((acc_o_1_3_lo / ((float4)(((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)])))));
  (*(half4*)(O_local_cast_1_3 + 4)) = (convert_half4((acc_o_1_3_hi / ((float4)(((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 3)])))));
  vstore8((*(half8*)(O_local_cast_1_3 + 0)), 0, O + (((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 32)) + 24));
}

