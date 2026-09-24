// Function: gdn_prep_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void gdn_prep_kernel_kernel(__global half* restrict A2buf, __global float* restrict B, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global float* restrict G, __global half* restrict K, __global half* restrict KDbuf, __global half* restrict Q, __global half* restrict Ubuf, __global half* restrict V, __global half* restrict Wbuf);
__kernel void gdn_prep_kernel_kernel(__global half* restrict A2buf, __global float* restrict B, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global float* restrict G, __global half* restrict K, __global half* restrict KDbuf, __global half* restrict Q, __global half* restrict Ubuf, __global half* restrict V, __global half* restrict Wbuf) { const int tl_gid0 = convert_int(get_group_id(0)); const int tl_gid1 = convert_int(get_group_id(1)); const int tl_lid0 = convert_int(get_local_id(0)); const int tl_wi_aff0 = ((tl_lid0 >> 5) * 128); const int tl_wi_aff1 = ((tl_lid0 & 31) * 128); const int tl_wi_aff2 = (tl_gid1 * 131072); const int tl_wi_aff3 = (tl_gid1 * 32768); const int tl_wi_aff4 = (tl_gid0 * 4096); const int tl_wi_aff5 = (tl_gid1 * 1024); const int tl_wi_aff6 = (tl_gid0 * 1024); const int tl_wi_aff7 = (tl_gid0 * 32);
__local float gc[32];
__local float bf[32];
__local float gf[32];
__local float egc[32];
__local float L[1024];
__local half UW[4096];
  if (tl_lid0 < 32) {
    gf[tl_lid0] = G[((tl_wi_aff5 + tl_wi_aff7) + tl_lid0)];
    bf[tl_lid0] = B[((tl_wi_aff5 + tl_wi_aff7) + tl_lid0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tl_lid0 < 32) {
    float s_1 = 0.000000e+00f;
    for (int r = 0; r < 32; ++r) {
      if (r <= tl_lid0) {
        s_1 = (s_1 + gf[r]);
      }
    }
    gc[tl_lid0] = s_1;
    egc[tl_lid0] = native_exp(s_1);
    EgcBuf[((tl_wi_aff5 + tl_wi_aff7) + tl_lid0)] = native_exp(s_1);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  EglBuf[((tl_gid1 * 32) + tl_gid0)] = native_exp(gc[31]);
  for (int t = tl_lid0; t < 1024; t += 128) {
    int i = t >> 5;
    int j = t & 31;
    float lv = 0.000000e+00f;
    float av = 0.000000e+00f;
    if (j <= i) {
      float decay = native_exp(gc[i] - gc[j]);
      half8 acc8 = (half8)(0.0h);
      half8 qacc = (half8)(0.0h);
      const int base_k = (((tl_gid1 & 15) * 131072) + (tl_gid0 * 4096));
      #pragma unroll
      for (int kd = 0; kd < 128; kd += 8) {
        half8 kj = vload8(0, K + base_k + (j * 128) + kd);
        acc8 += vload8(0, K + base_k + (i * 128) + kd) * kj;
        qacc += vload8(0, Q + base_k + (i * 128) + kd) * kj;
      }
      float kk = convert_float(acc8.s0) + convert_float(acc8.s1) + convert_float(acc8.s2) + convert_float(acc8.s3)
               + convert_float(acc8.s4) + convert_float(acc8.s5) + convert_float(acc8.s6) + convert_float(acc8.s7);
      if (j < i) {
        lv = bf[i] * kk * decay;
      }
      av = (convert_float(qacc.s0) + convert_float(qacc.s1) + convert_float(qacc.s2) + convert_float(qacc.s3)
          + convert_float(qacc.s4) + convert_float(qacc.s5) + convert_float(qacc.s6) + convert_float(qacc.s7)) * decay;
    }
    L[t] = lv;
    A2buf[(((tl_gid1 * 32768) + (tl_gid0 * 1024)) + t)] = convert_half(av);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = 0; i < 32; ++i) {
    float au_1[1];
    au_1[0] = (bf[i] * (convert_float(V[(((tl_wi_aff2 + tl_wi_aff4) + (i * 128)) + tl_lid0)])));
    for (int m = 0; m < 32; ++m) {
      if (m < i) {
        au_1[0] = (au_1[0] - (L[((i * 32) + m)] * (convert_float(UW[((m * 128) + tl_lid0)]))));
      }
    }
    UW[((i * 128) + tl_lid0)] = (convert_half(au_1[0]));
  }
  for (int i_1 = 0; i_1 < 32; ++i_1) { const int tl_loop_aff0 = (i_1 * 128);
    Ubuf[(((tl_wi_aff2 + tl_wi_aff4) + tl_loop_aff0) + tl_lid0)] = UW[(tl_loop_aff0 + tl_lid0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    float aw_1[1];
    aw_1[0] = ((bf[i_2] * (convert_float(K[(((((tl_gid1 & 15) * 131072) + tl_wi_aff4) + (i_2 * 128)) + tl_lid0)]))) * egc[i_2]);
    for (int m_1 = 0; m_1 < 32; ++m_1) {
      if (m_1 < i_2) {
        aw_1[0] = (aw_1[0] - (L[((i_2 * 32) + m_1)] * (convert_float(UW[((m_1 * 128) + tl_lid0)]))));
      }
    }
    UW[((i_2 * 128) + tl_lid0)] = (convert_half(aw_1[0]));
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int i_3 = 0; i_3 < 32; ++i_3) { const int tl_loop_aff0 = (i_3 * 128);
    Wbuf[(((tl_wi_aff2 + tl_wi_aff4) + tl_loop_aff0) + tl_lid0)] = UW[(tl_loop_aff0 + tl_lid0)];
    KDbuf[(((tl_wi_aff2 + tl_wi_aff4) + tl_loop_aff0) + tl_lid0)] = (convert_half(((convert_float(K[(((((tl_gid1 & 15) * 131072) + tl_wi_aff4) + tl_loop_aff0) + tl_lid0)])) * native_exp((gc[31] - gc[i_3])))));
  }
}

