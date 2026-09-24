// Function: gdn_prep_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void gdn_prep_kernel_kernel(__global half* restrict A2buf, __global float* restrict B, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global float* restrict G, __global half* restrict K, __global half* restrict KDbuf, __global half* restrict Q, __global half* restrict Ubuf, __global half* restrict V, __global half* restrict Wbuf);
__kernel void gdn_prep_kernel_kernel(__global half* restrict A2buf, __global float* restrict B, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global float* restrict G, __global half* restrict K, __global half* restrict KDbuf, __global half* restrict Q, __global half* restrict Ubuf, __global half* restrict V, __global half* restrict Wbuf) { const int tl_gid0 = convert_int(get_group_id(0)); const int tl_gid1 = convert_int(get_group_id(1)); const int tl_lid0 = convert_int(get_local_id(0)); const int tl_wi_aff0 = ((tl_lid0 >> 5) * 128); const int tl_wi_aff1 = ((tl_lid0 & 31) * 128); const int tl_wi_aff2 = (tl_gid1 * 122880); const int tl_wi_aff3 = (tl_gid1 * 30720); const int tl_wi_aff4 = (tl_gid0 * 1024); const int tl_wi_aff5 = (tl_gid0 * 4096); const int tl_wi_aff6 = (tl_gid1 * 960); const int tl_wi_aff7 = (tl_gid0 * 32);
__local float gc[32];
__local float bf[32];
__local float gf[32];
__local float egc[32];
__local float L[1024];
__local half UW[4096];
  if (tl_lid0 < 32) {
    gf[tl_lid0] = G[((tl_wi_aff6 + tl_wi_aff7) + tl_lid0)];
    bf[tl_lid0] = B[((tl_wi_aff6 + tl_wi_aff7) + tl_lid0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tl_lid0 < 32) {
    float s_1[1];
    s_1[0] = 0.000000e+00f;
    for (int r = 0; r < 32; ++r) {
      if (r <= tl_lid0) {
        s_1[0] = (s_1[0] + gf[r]);
      }
    }
    gc[tl_lid0] = s_1[0];
    egc[tl_lid0] = exp(s_1[0]);
    EgcBuf[((tl_wi_aff6 + tl_wi_aff7) + tl_lid0)] = exp(s_1[0]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  EglBuf[((tl_gid1 * 30) + tl_gid0)] = exp(gc[31]);
  float kk_1[1];
  float qk_1[1];
  kk_1[0] = 0.000000e+00f;
  qk_1[0] = 0.000000e+00f;
  for (int kd = 0; kd < 128; ++kd) {
    float kj = (convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff1) + kd)]));
    kk_1[0] = (kk_1[0] + ((convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd)])) * kj));
    qk_1[0] = (qk_1[0] + ((convert_float(Q[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd)])) * kj));
  }
  float decay = exp((gc[(tl_lid0 >> 5)] - gc[(tl_lid0 & 31)]));
  float condval;
  if (((tl_lid0 & 31) < (tl_lid0 >> 5))) {
    condval = ((bf[(tl_lid0 >> 5)] * kk_1[0]) * decay);
  } else {
    condval = 0.000000e+00f;
  }
  L[tl_lid0] = condval;
  float condval_1;
  if (((tl_lid0 & 31) <= (tl_lid0 >> 5))) {
    condval_1 = (qk_1[0] * decay);
  } else {
    condval_1 = 0.000000e+00f;
  }
  A2buf[((tl_wi_aff3 + tl_wi_aff4) + tl_lid0)] = (convert_half(condval_1));
  float kk_2[1];
  float qk_2[1];
  kk_2[0] = 0.000000e+00f;
  qk_2[0] = 0.000000e+00f;
  for (int kd_1 = 0; kd_1 < 128; ++kd_1) {
    float kj_1 = (convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff1) + kd_1)]));
    kk_2[0] = (kk_2[0] + ((convert_float(K[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_1) + 512)])) * kj_1));
    qk_2[0] = (qk_2[0] + ((convert_float(Q[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_1) + 512)])) * kj_1));
  }
  float decay_1 = exp((gc[((tl_lid0 >> 5) + 4)] - gc[(tl_lid0 & 31)]));
  float condval_2;
  if (((tl_lid0 & 31) < ((tl_lid0 >> 5) + 4))) {
    condval_2 = ((bf[((tl_lid0 >> 5) + 4)] * kk_2[0]) * decay_1);
  } else {
    condval_2 = 0.000000e+00f;
  }
  L[(tl_lid0 + 128)] = condval_2;
  float condval_3;
  if (((tl_lid0 & 31) <= ((tl_lid0 >> 5) + 4))) {
    condval_3 = (qk_2[0] * decay_1);
  } else {
    condval_3 = 0.000000e+00f;
  }
  A2buf[(((tl_wi_aff3 + tl_wi_aff4) + tl_lid0) + 128)] = (convert_half(condval_3));
  float kk_3[1];
  float qk_3[1];
  kk_3[0] = 0.000000e+00f;
  qk_3[0] = 0.000000e+00f;
  for (int kd_2 = 0; kd_2 < 128; ++kd_2) {
    float kj_2 = (convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff1) + kd_2)]));
    kk_3[0] = (kk_3[0] + ((convert_float(K[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_2) + 1024)])) * kj_2));
    qk_3[0] = (qk_3[0] + ((convert_float(Q[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_2) + 1024)])) * kj_2));
  }
  float decay_2 = exp((gc[((tl_lid0 >> 5) + 8)] - gc[(tl_lid0 & 31)]));
  float condval_4;
  if (((tl_lid0 & 31) < ((tl_lid0 >> 5) + 8))) {
    condval_4 = ((bf[((tl_lid0 >> 5) + 8)] * kk_3[0]) * decay_2);
  } else {
    condval_4 = 0.000000e+00f;
  }
  L[(tl_lid0 + 256)] = condval_4;
  float condval_5;
  if (((tl_lid0 & 31) <= ((tl_lid0 >> 5) + 8))) {
    condval_5 = (qk_3[0] * decay_2);
  } else {
    condval_5 = 0.000000e+00f;
  }
  A2buf[(((tl_wi_aff3 + tl_wi_aff4) + tl_lid0) + 256)] = (convert_half(condval_5));
  float kk_4[1];
  float qk_4[1];
  kk_4[0] = 0.000000e+00f;
  qk_4[0] = 0.000000e+00f;
  for (int kd_3 = 0; kd_3 < 128; ++kd_3) {
    float kj_3 = (convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff1) + kd_3)]));
    kk_4[0] = (kk_4[0] + ((convert_float(K[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_3) + 1536)])) * kj_3));
    qk_4[0] = (qk_4[0] + ((convert_float(Q[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_3) + 1536)])) * kj_3));
  }
  float decay_3 = exp((gc[((tl_lid0 >> 5) + 12)] - gc[(tl_lid0 & 31)]));
  float condval_6;
  if (((tl_lid0 & 31) < ((tl_lid0 >> 5) + 12))) {
    condval_6 = ((bf[((tl_lid0 >> 5) + 12)] * kk_4[0]) * decay_3);
  } else {
    condval_6 = 0.000000e+00f;
  }
  L[(tl_lid0 + 384)] = condval_6;
  float condval_7;
  if (((tl_lid0 & 31) <= ((tl_lid0 >> 5) + 12))) {
    condval_7 = (qk_4[0] * decay_3);
  } else {
    condval_7 = 0.000000e+00f;
  }
  A2buf[(((tl_wi_aff3 + tl_wi_aff4) + tl_lid0) + 384)] = (convert_half(condval_7));
  float kk_5[1];
  float qk_5[1];
  kk_5[0] = 0.000000e+00f;
  qk_5[0] = 0.000000e+00f;
  for (int kd_4 = 0; kd_4 < 128; ++kd_4) {
    float kj_4 = (convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff1) + kd_4)]));
    kk_5[0] = (kk_5[0] + ((convert_float(K[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_4) + 2048)])) * kj_4));
    qk_5[0] = (qk_5[0] + ((convert_float(Q[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_4) + 2048)])) * kj_4));
  }
  float decay_4 = exp((gc[((tl_lid0 >> 5) + 16)] - gc[(tl_lid0 & 31)]));
  float condval_8;
  if (((tl_lid0 & 31) < ((tl_lid0 >> 5) + 16))) {
    condval_8 = ((bf[((tl_lid0 >> 5) + 16)] * kk_5[0]) * decay_4);
  } else {
    condval_8 = 0.000000e+00f;
  }
  L[(tl_lid0 + 512)] = condval_8;
  float condval_9;
  if (((tl_lid0 & 31) <= ((tl_lid0 >> 5) + 16))) {
    condval_9 = (qk_5[0] * decay_4);
  } else {
    condval_9 = 0.000000e+00f;
  }
  A2buf[(((tl_wi_aff3 + tl_wi_aff4) + tl_lid0) + 512)] = (convert_half(condval_9));
  float kk_6[1];
  float qk_6[1];
  kk_6[0] = 0.000000e+00f;
  qk_6[0] = 0.000000e+00f;
  for (int kd_5 = 0; kd_5 < 128; ++kd_5) {
    float kj_5 = (convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff1) + kd_5)]));
    kk_6[0] = (kk_6[0] + ((convert_float(K[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_5) + 2560)])) * kj_5));
    qk_6[0] = (qk_6[0] + ((convert_float(Q[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_5) + 2560)])) * kj_5));
  }
  float decay_5 = exp((gc[((tl_lid0 >> 5) + 20)] - gc[(tl_lid0 & 31)]));
  float condval_10;
  if (((tl_lid0 & 31) < ((tl_lid0 >> 5) + 20))) {
    condval_10 = ((bf[((tl_lid0 >> 5) + 20)] * kk_6[0]) * decay_5);
  } else {
    condval_10 = 0.000000e+00f;
  }
  L[(tl_lid0 + 640)] = condval_10;
  float condval_11;
  if (((tl_lid0 & 31) <= ((tl_lid0 >> 5) + 20))) {
    condval_11 = (qk_6[0] * decay_5);
  } else {
    condval_11 = 0.000000e+00f;
  }
  A2buf[(((tl_wi_aff3 + tl_wi_aff4) + tl_lid0) + 640)] = (convert_half(condval_11));
  float kk_7[1];
  float qk_7[1];
  kk_7[0] = 0.000000e+00f;
  qk_7[0] = 0.000000e+00f;
  for (int kd_6 = 0; kd_6 < 128; ++kd_6) {
    float kj_6 = (convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff1) + kd_6)]));
    kk_7[0] = (kk_7[0] + ((convert_float(K[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_6) + 3072)])) * kj_6));
    qk_7[0] = (qk_7[0] + ((convert_float(Q[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_6) + 3072)])) * kj_6));
  }
  float decay_6 = exp((gc[((tl_lid0 >> 5) + 24)] - gc[(tl_lid0 & 31)]));
  float condval_12;
  if (((tl_lid0 & 31) < ((tl_lid0 >> 5) + 24))) {
    condval_12 = ((bf[((tl_lid0 >> 5) + 24)] * kk_7[0]) * decay_6);
  } else {
    condval_12 = 0.000000e+00f;
  }
  L[(tl_lid0 + 768)] = condval_12;
  float condval_13;
  if (((tl_lid0 & 31) <= ((tl_lid0 >> 5) + 24))) {
    condval_13 = (qk_7[0] * decay_6);
  } else {
    condval_13 = 0.000000e+00f;
  }
  A2buf[(((tl_wi_aff3 + tl_wi_aff4) + tl_lid0) + 768)] = (convert_half(condval_13));
  float kk_8[1];
  float qk_8[1];
  kk_8[0] = 0.000000e+00f;
  qk_8[0] = 0.000000e+00f;
  for (int kd_7 = 0; kd_7 < 128; ++kd_7) {
    float kj_7 = (convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff1) + kd_7)]));
    kk_8[0] = (kk_8[0] + ((convert_float(K[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_7) + 3584)])) * kj_7));
    qk_8[0] = (qk_8[0] + ((convert_float(Q[((((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_wi_aff0) + kd_7) + 3584)])) * kj_7));
  }
  float decay_7 = exp((gc[((tl_lid0 >> 5) + 28)] - gc[(tl_lid0 & 31)]));
  float condval_14;
  if (((tl_lid0 & 31) < ((tl_lid0 >> 5) + 28))) {
    condval_14 = ((bf[((tl_lid0 >> 5) + 28)] * kk_8[0]) * decay_7);
  } else {
    condval_14 = 0.000000e+00f;
  }
  L[(tl_lid0 + 896)] = condval_14;
  float condval_15;
  if (((tl_lid0 & 31) <= ((tl_lid0 >> 5) + 28))) {
    condval_15 = (qk_8[0] * decay_7);
  } else {
    condval_15 = 0.000000e+00f;
  }
  A2buf[(((tl_wi_aff3 + tl_wi_aff4) + tl_lid0) + 896)] = (convert_half(condval_15));
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = 0; i < 32; ++i) {
    float au_1[1];
    au_1[0] = (bf[i] * (convert_float(V[(((tl_wi_aff2 + tl_wi_aff5) + (i * 128)) + tl_lid0)])));
    for (int m = 0; m < 32; ++m) {
      if (m < i) {
        au_1[0] = (au_1[0] - (L[((i * 32) + m)] * (convert_float(UW[((m * 128) + tl_lid0)]))));
      }
    }
    UW[((i * 128) + tl_lid0)] = (convert_half(au_1[0]));
  }
  for (int i_1 = 0; i_1 < 32; ++i_1) { const int tl_loop_aff0 = (i_1 * 128);
    Ubuf[(((tl_wi_aff2 + tl_wi_aff5) + tl_loop_aff0) + tl_lid0)] = UW[(tl_loop_aff0 + tl_lid0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    float aw_1[1];
    aw_1[0] = ((bf[i_2] * (convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + (i_2 * 128)) + tl_lid0)]))) * egc[i_2]);
    for (int m_1 = 0; m_1 < 32; ++m_1) {
      if (m_1 < i_2) {
        aw_1[0] = (aw_1[0] - (L[((i_2 * 32) + m_1)] * (convert_float(UW[((m_1 * 128) + tl_lid0)]))));
      }
    }
    UW[((i_2 * 128) + tl_lid0)] = (convert_half(aw_1[0]));
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int i_3 = 0; i_3 < 32; ++i_3) { const int tl_loop_aff0 = (i_3 * 128);
    Wbuf[(((tl_wi_aff2 + tl_wi_aff5) + tl_loop_aff0) + tl_lid0)] = UW[(tl_loop_aff0 + tl_lid0)];
    KDbuf[(((tl_wi_aff2 + tl_wi_aff5) + tl_loop_aff0) + tl_lid0)] = (convert_half(((convert_float(K[(((((tl_gid1 & 15) * 122880) + tl_wi_aff5) + tl_loop_aff0) + tl_lid0)])) * exp((gc[31] - gc[i_3])))));
  }
}

