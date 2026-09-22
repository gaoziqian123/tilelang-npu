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
  float8 acc_o_0 = (float8)(0.000000e+00f);
  float8 acc_o_1 = (float8)(0.000000e+00f);
  float8 acc_o_2 = (float8)(0.000000e+00f);
  float8 acc_o_3 = (float8)(0.000000e+00f);
  float8 acc_s_0 = (float8)(0.000000e+00f);
  float8 acc_s_1 = (float8)(0.000000e+00f);
  float m_old[1];
  float m_new[1];
  float l_t[1];
  float8 acc_o_1_0 = (float8)(0.000000e+00f);
  float8 acc_o_1_1 = (float8)(0.000000e+00f);
  float8 acc_o_1_2 = (float8)(0.000000e+00f);
  float8 acc_o_1_3 = (float8)(0.000000e+00f);
  if ((convert_int(get_local_id(0))) < 32) {
    ((float*)mrun)[(convert_int(get_local_id(0)))] = -3.276800e+04f;
    ((float*)lrun)[(convert_int(get_local_id(0)))] = 0.000000e+00f;
    ((float*)alpha)[(convert_int(get_local_id(0)))] = 1.000000e+00f;
  }
  float broadcast_var = 0.000000e+00f;
  acc_o_1_0.lo = (((float4)(broadcast_var, broadcast_var, broadcast_var, broadcast_var)));
  float broadcast_var_1 = 0.000000e+00f;
  acc_o_1_0.hi = (((float4)(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1)));
  float broadcast_var_2 = 0.000000e+00f;
  acc_o_1_1.lo = (((float4)(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2)));
  float broadcast_var_3 = 0.000000e+00f;
  acc_o_1_1.hi = (((float4)(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3)));
  float broadcast_var_4 = 0.000000e+00f;
  acc_o_1_2.lo = (((float4)(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4)));
  float broadcast_var_5 = 0.000000e+00f;
  acc_o_1_2.hi = (((float4)(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5)));
  float broadcast_var_6 = 0.000000e+00f;
  acc_o_1_3.lo = (((float4)(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6)));
  float broadcast_var_7 = 0.000000e+00f;
  acc_o_1_3.hi = (((float4)(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7)));
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int ks = 0; ks < (((convert_int(get_group_id(0))) >> 2) + 1); ++ks) {
    float8 acc_s_1_0 = (float8)(0.000000e+00f);
    float8 acc_s_1_1 = (float8)(0.000000e+00f);
    float broadcast_var_8 = 0.000000e+00f;
    acc_s_1_0.lo = (((float4)(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8)));
    float broadcast_var_9 = 0.000000e+00f;
    acc_s_1_0.hi = (((float4)(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9)));
    float broadcast_var_10 = 0.000000e+00f;
    acc_s_1_1.lo = (((float4)(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10)));
    float broadcast_var_11 = 0.000000e+00f;
    acc_s_1_1.hi = (((float4)(broadcast_var_11, broadcast_var_11, broadcast_var_11, broadcast_var_11)));
    for (int _tmp = 0; _tmp < 16; ++_tmp) {
      ((half*)Qs)[((((convert_int(get_local_id(0))) & 15) * 32) + ((convert_int(get_local_id(0))) >> 4))] = Q[((((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + (((convert_int(get_local_id(0))) >> 4) * 256)) + (_tmp * 16)) + ((convert_int(get_local_id(0))) & 15))];
      ((half*)Qs)[(((((convert_int(get_local_id(0))) & 15) * 32) + ((convert_int(get_local_id(0))) >> 4)) + 16)] = Q[(((((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + (((convert_int(get_local_id(0))) >> 4) * 256)) + (_tmp * 16)) + ((convert_int(get_local_id(0))) & 15)) + 4096)];
      ((half*)Ks)[((((convert_int(get_local_id(0))) & 15) * 128) + ((convert_int(get_local_id(0))) >> 4))] = K[(((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (((convert_int(get_local_id(0))) >> 4) * 256)) + (_tmp * 16)) + ((convert_int(get_local_id(0))) & 15))];
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 15) * 128) + ((convert_int(get_local_id(0))) >> 4)) + 16)] = K[((((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (((convert_int(get_local_id(0))) >> 4) * 256)) + (_tmp * 16)) + ((convert_int(get_local_id(0))) & 15)) + 4096)];
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 15) * 128) + ((convert_int(get_local_id(0))) >> 4)) + 32)] = K[((((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (((convert_int(get_local_id(0))) >> 4) * 256)) + (_tmp * 16)) + ((convert_int(get_local_id(0))) & 15)) + 8192)];
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 15) * 128) + ((convert_int(get_local_id(0))) >> 4)) + 48)] = K[((((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (((convert_int(get_local_id(0))) >> 4) * 256)) + (_tmp * 16)) + ((convert_int(get_local_id(0))) & 15)) + 12288)];
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 15) * 128) + ((convert_int(get_local_id(0))) >> 4)) + 64)] = K[((((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (((convert_int(get_local_id(0))) >> 4) * 256)) + (_tmp * 16)) + ((convert_int(get_local_id(0))) & 15)) + 16384)];
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 15) * 128) + ((convert_int(get_local_id(0))) >> 4)) + 80)] = K[((((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (((convert_int(get_local_id(0))) >> 4) * 256)) + (_tmp * 16)) + ((convert_int(get_local_id(0))) & 15)) + 20480)];
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 15) * 128) + ((convert_int(get_local_id(0))) >> 4)) + 96)] = K[((((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (((convert_int(get_local_id(0))) >> 4) * 256)) + (_tmp * 16)) + ((convert_int(get_local_id(0))) & 15)) + 24576)];
      ((half*)Ks)[(((((convert_int(get_local_id(0))) & 15) * 128) + ((convert_int(get_local_id(0))) >> 4)) + 112)] = K[((((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (((convert_int(get_local_id(0))) >> 4) * 256)) + (_tmp * 16)) + ((convert_int(get_local_id(0))) & 15)) + 28672)];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k = 0; k < 16; ++k) {
        half Qs_local_cast[4];
        half Ks_local_cast_1[4];
        (*(half4*)(Qs_local_cast + 0)) = ((half4)(((half*)Qs)[((k * 32) + ((convert_int(get_local_id(0))) >> 5))]));
        (*(half4*)(Ks_local_cast_1 + 0)) = vload4(0, (half*)Ks + ((k * 128) + (((convert_int(get_local_id(0))) & 31) * 4)));
        acc_s_1_0.lo = (((float4)(acc_s_1_0.s0, acc_s_1_0.s1, acc_s_1_0.s2, acc_s_1_0.s3) + ((convert_float4((*(half4*)(Qs_local_cast + 0)))) * (convert_float4((*(half4*)(Ks_local_cast_1 + 0)))))));
        half Qs_local_cast_1[4];
        half Ks_local_cast_1_1[4];
        (*(half4*)(Qs_local_cast_1 + 0)) = ((half4)(((half*)Qs)[(((k * 32) + ((convert_int(get_local_id(0))) >> 5)) + 8)]));
        (*(half4*)(Ks_local_cast_1_1 + 0)) = vload4(0, (half*)Ks + ((k * 128) + (((convert_int(get_local_id(0))) & 31) * 4)));
        acc_s_1_0.hi = (((float4)(acc_s_1_0.s4, acc_s_1_0.s5, acc_s_1_0.s6, acc_s_1_0.s7) + ((convert_float4((*(half4*)(Qs_local_cast_1 + 0)))) * (convert_float4((*(half4*)(Ks_local_cast_1_1 + 0)))))));
        half Qs_local_cast_2[4];
        half Ks_local_cast_1_2[4];
        (*(half4*)(Qs_local_cast_2 + 0)) = ((half4)(((half*)Qs)[(((k * 32) + ((convert_int(get_local_id(0))) >> 5)) + 16)]));
        (*(half4*)(Ks_local_cast_1_2 + 0)) = vload4(0, (half*)Ks + ((k * 128) + (((convert_int(get_local_id(0))) & 31) * 4)));
        acc_s_1_1.lo = (((float4)(acc_s_1_1.s0, acc_s_1_1.s1, acc_s_1_1.s2, acc_s_1_1.s3) + ((convert_float4((*(half4*)(Qs_local_cast_2 + 0)))) * (convert_float4((*(half4*)(Ks_local_cast_1_2 + 0)))))));
        half Qs_local_cast_3[4];
        half Ks_local_cast_1_3[4];
        (*(half4*)(Qs_local_cast_3 + 0)) = ((half4)(((half*)Qs)[(((k * 32) + ((convert_int(get_local_id(0))) >> 5)) + 24)]));
        (*(half4*)(Ks_local_cast_1_3 + 0)) = vload4(0, (half*)Ks + ((k * 128) + (((convert_int(get_local_id(0))) & 31) * 4)));
        acc_s_1_1.hi = (((float4)(acc_s_1_1.s4, acc_s_1_1.s5, acc_s_1_1.s6, acc_s_1_1.s7) + ((convert_float4((*(half4*)(Qs_local_cast_3 + 0)))) * (convert_float4((*(half4*)(Ks_local_cast_1_3 + 0)))))));
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    float condval;
    if (((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval = -3.276800e+04f;
    } else {
      condval = (acc_s_1_0.s0 * 6.250000e-02f);
    }
    ((half*)Sh)[((convert_int(get_local_id(0))) * 4)] = (convert_half(condval));
    float condval_1;
    if (((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) <= ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_1 = -3.276800e+04f;
    } else {
      condval_1 = (acc_s_1_0.s1 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 1)] = (convert_half(condval_1));
    float condval_2;
    if (((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) < (((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)) + 2))) {
      condval_2 = -3.276800e+04f;
    } else {
      condval_2 = (acc_s_1_0.s2 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 2)] = (convert_half(condval_2));
    float condval_3;
    if (((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) < (((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)) + 3))) {
      condval_3 = -3.276800e+04f;
    } else {
      condval_3 = (acc_s_1_0.s3 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 3)] = (convert_half(condval_3));
    float condval_4;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 8) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_4 = -3.276800e+04f;
    } else {
      condval_4 = (acc_s_1_0.s4 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 1024)] = (convert_half(condval_4));
    float condval_5;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 7) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_5 = -3.276800e+04f;
    } else {
      condval_5 = (acc_s_1_0.s5 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 1025)] = (convert_half(condval_5));
    float condval_6;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 6) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_6 = -3.276800e+04f;
    } else {
      condval_6 = (acc_s_1_0.s6 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 1026)] = (convert_half(condval_6));
    float condval_7;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 5) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_7 = -3.276800e+04f;
    } else {
      condval_7 = (acc_s_1_0.s7 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 1027)] = (convert_half(condval_7));
    float condval_8;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 16) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_8 = -3.276800e+04f;
    } else {
      condval_8 = (acc_s_1_1.s0 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 2048)] = (convert_half(condval_8));
    float condval_9;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 15) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_9 = -3.276800e+04f;
    } else {
      condval_9 = (acc_s_1_1.s1 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 2049)] = (convert_half(condval_9));
    float condval_10;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 14) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_10 = -3.276800e+04f;
    } else {
      condval_10 = (acc_s_1_1.s2 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 2050)] = (convert_half(condval_10));
    float condval_11;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 13) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_11 = -3.276800e+04f;
    } else {
      condval_11 = (acc_s_1_1.s3 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 2051)] = (convert_half(condval_11));
    float condval_12;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 24) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_12 = -3.276800e+04f;
    } else {
      condval_12 = (acc_s_1_1.s4 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 3072)] = (convert_half(condval_12));
    float condval_13;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 23) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_13 = -3.276800e+04f;
    } else {
      condval_13 = (acc_s_1_1.s5 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 3073)] = (convert_half(condval_13));
    float condval_14;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 22) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_14 = -3.276800e+04f;
    } else {
      condval_14 = (acc_s_1_1.s6 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 3074)] = (convert_half(condval_14));
    float condval_15;
    if ((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 5)) + 21) < ((ks * 128) + (((convert_int(get_local_id(0))) & 31) * 4)))) {
      condval_15 = -3.276800e+04f;
    } else {
      condval_15 = (acc_s_1_1.s7 * 6.250000e-02f);
    }
    ((half*)Sh)[(((convert_int(get_local_id(0))) * 4) + 3075)] = (convert_half(condval_15));
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
    acc_o_1_0.lo = (((float4)(acc_o_1_0.s0, acc_o_1_0.s1, acc_o_1_0.s2, acc_o_1_0.s3) * ((float4)(((float*)alpha)[((convert_int(get_local_id(0))) >> 6)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 6)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 6)], ((float*)alpha)[((convert_int(get_local_id(0))) >> 6)]))));
    acc_o_1_0.hi = (((float4)(acc_o_1_0.s4, acc_o_1_0.s5, acc_o_1_0.s6, acc_o_1_0.s7) * ((float4)(((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 4)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 4)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 4)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 4)]))));
    acc_o_1_1.lo = (((float4)(acc_o_1_1.s0, acc_o_1_1.s1, acc_o_1_1.s2, acc_o_1_1.s3) * ((float4)(((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 8)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 8)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 8)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 8)]))));
    acc_o_1_1.hi = (((float4)(acc_o_1_1.s4, acc_o_1_1.s5, acc_o_1_1.s6, acc_o_1_1.s7) * ((float4)(((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 12)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 12)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 12)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 12)]))));
    acc_o_1_2.lo = (((float4)(acc_o_1_2.s0, acc_o_1_2.s1, acc_o_1_2.s2, acc_o_1_2.s3) * ((float4)(((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 16)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 16)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 16)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 16)]))));
    acc_o_1_2.hi = (((float4)(acc_o_1_2.s4, acc_o_1_2.s5, acc_o_1_2.s6, acc_o_1_2.s7) * ((float4)(((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 20)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 20)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 20)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 20)]))));
    acc_o_1_3.lo = (((float4)(acc_o_1_3.s0, acc_o_1_3.s1, acc_o_1_3.s2, acc_o_1_3.s3) * ((float4)(((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 24)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 24)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 24)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 24)]))));
    acc_o_1_3.hi = (((float4)(acc_o_1_3.s4, acc_o_1_3.s5, acc_o_1_3.s6, acc_o_1_3.s7) * ((float4)(((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 28)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 28)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 28)], ((float*)alpha)[(((convert_int(get_local_id(0))) >> 6) + 28)]))));
    for (int j = 0; j < 128; ++j) {
      half Sh_local_cast_2[4];
      half V_local_cast_3[4];
      (*(half4*)(Sh_local_cast_2 + 0)) = ((half4)(((half*)Sh)[((((convert_int(get_local_id(0))) >> 6) * 128) + j)]));
      (*(half4*)(V_local_cast_3 + 0)) = vload4(0, V + ((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 63) * 4)));
      acc_o_1_0.lo = (((float4)(acc_o_1_0.s0, acc_o_1_0.s1, acc_o_1_0.s2, acc_o_1_0.s3) + ((convert_float4((*(half4*)(Sh_local_cast_2 + 0)))) * (convert_float4((*(half4*)(V_local_cast_3 + 0)))))));
      half Sh_local_cast_2_1[4];
      half V_local_cast_3_1[4];
      (*(half4*)(Sh_local_cast_2_1 + 0)) = ((half4)(((half*)Sh)[(((((convert_int(get_local_id(0))) >> 6) * 128) + j) + 512)]));
      (*(half4*)(V_local_cast_3_1 + 0)) = vload4(0, V + ((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 63) * 4)));
      acc_o_1_0.hi = (((float4)(acc_o_1_0.s4, acc_o_1_0.s5, acc_o_1_0.s6, acc_o_1_0.s7) + ((convert_float4((*(half4*)(Sh_local_cast_2_1 + 0)))) * (convert_float4((*(half4*)(V_local_cast_3_1 + 0)))))));
      half Sh_local_cast_2_2[4];
      half V_local_cast_3_2[4];
      (*(half4*)(Sh_local_cast_2_2 + 0)) = ((half4)(((half*)Sh)[(((((convert_int(get_local_id(0))) >> 6) * 128) + j) + 1024)]));
      (*(half4*)(V_local_cast_3_2 + 0)) = vload4(0, V + ((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 63) * 4)));
      acc_o_1_1.lo = (((float4)(acc_o_1_1.s0, acc_o_1_1.s1, acc_o_1_1.s2, acc_o_1_1.s3) + ((convert_float4((*(half4*)(Sh_local_cast_2_2 + 0)))) * (convert_float4((*(half4*)(V_local_cast_3_2 + 0)))))));
      half Sh_local_cast_2_3[4];
      half V_local_cast_3_3[4];
      (*(half4*)(Sh_local_cast_2_3 + 0)) = ((half4)(((half*)Sh)[(((((convert_int(get_local_id(0))) >> 6) * 128) + j) + 1536)]));
      (*(half4*)(V_local_cast_3_3 + 0)) = vload4(0, V + ((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 63) * 4)));
      acc_o_1_1.hi = (((float4)(acc_o_1_1.s4, acc_o_1_1.s5, acc_o_1_1.s6, acc_o_1_1.s7) + ((convert_float4((*(half4*)(Sh_local_cast_2_3 + 0)))) * (convert_float4((*(half4*)(V_local_cast_3_3 + 0)))))));
      half Sh_local_cast_2_4[4];
      half V_local_cast_3_4[4];
      (*(half4*)(Sh_local_cast_2_4 + 0)) = ((half4)(((half*)Sh)[(((((convert_int(get_local_id(0))) >> 6) * 128) + j) + 2048)]));
      (*(half4*)(V_local_cast_3_4 + 0)) = vload4(0, V + ((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 63) * 4)));
      acc_o_1_2.lo = (((float4)(acc_o_1_2.s0, acc_o_1_2.s1, acc_o_1_2.s2, acc_o_1_2.s3) + ((convert_float4((*(half4*)(Sh_local_cast_2_4 + 0)))) * (convert_float4((*(half4*)(V_local_cast_3_4 + 0)))))));
      half Sh_local_cast_2_5[4];
      half V_local_cast_3_5[4];
      (*(half4*)(Sh_local_cast_2_5 + 0)) = ((half4)(((half*)Sh)[(((((convert_int(get_local_id(0))) >> 6) * 128) + j) + 2560)]));
      (*(half4*)(V_local_cast_3_5 + 0)) = vload4(0, V + ((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 63) * 4)));
      acc_o_1_2.hi = (((float4)(acc_o_1_2.s4, acc_o_1_2.s5, acc_o_1_2.s6, acc_o_1_2.s7) + ((convert_float4((*(half4*)(Sh_local_cast_2_5 + 0)))) * (convert_float4((*(half4*)(V_local_cast_3_5 + 0)))))));
      half Sh_local_cast_2_6[4];
      half V_local_cast_3_6[4];
      (*(half4*)(Sh_local_cast_2_6 + 0)) = ((half4)(((half*)Sh)[(((((convert_int(get_local_id(0))) >> 6) * 128) + j) + 3072)]));
      (*(half4*)(V_local_cast_3_6 + 0)) = vload4(0, V + ((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 63) * 4)));
      acc_o_1_3.lo = (((float4)(acc_o_1_3.s0, acc_o_1_3.s1, acc_o_1_3.s2, acc_o_1_3.s3) + ((convert_float4((*(half4*)(Sh_local_cast_2_6 + 0)))) * (convert_float4((*(half4*)(V_local_cast_3_6 + 0)))))));
      half Sh_local_cast_2_7[4];
      half V_local_cast_3_7[4];
      (*(half4*)(Sh_local_cast_2_7 + 0)) = ((half4)(((half*)Sh)[(((((convert_int(get_local_id(0))) >> 6) * 128) + j) + 3584)]));
      (*(half4*)(V_local_cast_3_7 + 0)) = vload4(0, V + ((((((convert_int(get_group_id(1))) >> 2) * 262144) + (ks * 32768)) + (j * 256)) + (((convert_int(get_local_id(0))) & 63) * 4)));
      acc_o_1_3.hi = (((float4)(acc_o_1_3.s4, acc_o_1_3.s5, acc_o_1_3.s6, acc_o_1_3.s7) + ((convert_float4((*(half4*)(Sh_local_cast_2_7 + 0)))) * (convert_float4((*(half4*)(V_local_cast_3_7 + 0)))))));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float lrun_local_cast_5[4];
  half O_local_cast_4[4];
  (*(float4*)(lrun_local_cast_5 + 0)) = ((float4)(((float*)lrun)[((convert_int(get_local_id(0))) >> 6)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 6)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 6)], ((float*)lrun)[((convert_int(get_local_id(0))) >> 6)]));
  (*(half4*)(O_local_cast_4 + 0)) = (convert_half4(((float4)(acc_o_1_0.s0, acc_o_1_0.s1, acc_o_1_0.s2, acc_o_1_0.s3) / (*(float4*)(lrun_local_cast_5 + 0)))));
  vstore4((*(half4*)(O_local_cast_4 + 0)), 0, O + ((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 4)));
  float lrun_local_cast_5_1[4];
  half O_local_cast_4_1[4];
  (*(float4*)(lrun_local_cast_5_1 + 0)) = ((float4)(((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 4)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 4)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 4)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 4)]));
  (*(half4*)(O_local_cast_4_1 + 0)) = (convert_half4(((float4)(acc_o_1_0.s4, acc_o_1_0.s5, acc_o_1_0.s6, acc_o_1_0.s7) / (*(float4*)(lrun_local_cast_5_1 + 0)))));
  vstore4((*(half4*)(O_local_cast_4_1 + 0)), 0, O + (((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 4)) + 1024));
  float lrun_local_cast_5_2[4];
  half O_local_cast_4_2[4];
  (*(float4*)(lrun_local_cast_5_2 + 0)) = ((float4)(((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 8)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 8)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 8)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 8)]));
  (*(half4*)(O_local_cast_4_2 + 0)) = (convert_half4(((float4)(acc_o_1_1.s0, acc_o_1_1.s1, acc_o_1_1.s2, acc_o_1_1.s3) / (*(float4*)(lrun_local_cast_5_2 + 0)))));
  vstore4((*(half4*)(O_local_cast_4_2 + 0)), 0, O + (((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 4)) + 2048));
  float lrun_local_cast_5_3[4];
  half O_local_cast_4_3[4];
  (*(float4*)(lrun_local_cast_5_3 + 0)) = ((float4)(((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 12)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 12)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 12)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 12)]));
  (*(half4*)(O_local_cast_4_3 + 0)) = (convert_half4(((float4)(acc_o_1_1.s4, acc_o_1_1.s5, acc_o_1_1.s6, acc_o_1_1.s7) / (*(float4*)(lrun_local_cast_5_3 + 0)))));
  vstore4((*(half4*)(O_local_cast_4_3 + 0)), 0, O + (((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 4)) + 3072));
  float lrun_local_cast_5_4[4];
  half O_local_cast_4_4[4];
  (*(float4*)(lrun_local_cast_5_4 + 0)) = ((float4)(((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 16)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 16)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 16)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 16)]));
  (*(half4*)(O_local_cast_4_4 + 0)) = (convert_half4(((float4)(acc_o_1_2.s0, acc_o_1_2.s1, acc_o_1_2.s2, acc_o_1_2.s3) / (*(float4*)(lrun_local_cast_5_4 + 0)))));
  vstore4((*(half4*)(O_local_cast_4_4 + 0)), 0, O + (((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 4)) + 4096));
  float lrun_local_cast_5_5[4];
  half O_local_cast_4_5[4];
  (*(float4*)(lrun_local_cast_5_5 + 0)) = ((float4)(((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 20)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 20)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 20)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 20)]));
  (*(half4*)(O_local_cast_4_5 + 0)) = (convert_half4(((float4)(acc_o_1_2.s4, acc_o_1_2.s5, acc_o_1_2.s6, acc_o_1_2.s7) / (*(float4*)(lrun_local_cast_5_5 + 0)))));
  vstore4((*(half4*)(O_local_cast_4_5 + 0)), 0, O + (((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 4)) + 5120));
  float lrun_local_cast_5_6[4];
  half O_local_cast_4_6[4];
  (*(float4*)(lrun_local_cast_5_6 + 0)) = ((float4)(((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 24)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 24)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 24)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 24)]));
  (*(half4*)(O_local_cast_4_6 + 0)) = (convert_half4(((float4)(acc_o_1_3.s0, acc_o_1_3.s1, acc_o_1_3.s2, acc_o_1_3.s3) / (*(float4*)(lrun_local_cast_5_6 + 0)))));
  vstore4((*(half4*)(O_local_cast_4_6 + 0)), 0, O + (((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 4)) + 6144));
  float lrun_local_cast_5_7[4];
  half O_local_cast_4_7[4];
  (*(float4*)(lrun_local_cast_5_7 + 0)) = ((float4)(((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 28)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 28)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 28)], ((float*)lrun)[(((convert_int(get_local_id(0))) >> 6) + 28)]));
  (*(half4*)(O_local_cast_4_7 + 0)) = (convert_half4(((float4)(acc_o_1_3.s4, acc_o_1_3.s5, acc_o_1_3.s6, acc_o_1_3.s7) / (*(float4*)(lrun_local_cast_5_7 + 0)))));
  vstore4((*(half4*)(O_local_cast_4_7 + 0)), 0, O + (((((convert_int(get_group_id(1))) * 262144) + ((convert_int(get_group_id(0))) * 8192)) + ((convert_int(get_local_id(0))) * 4)) + 7168));
}

