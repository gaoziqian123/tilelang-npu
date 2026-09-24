// Function: gdn_seq_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void gdn_seq_kernel_kernel(__global half* restrict A2buf, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global half* restrict KDbuf, __global half* restrict O, __global half* restrict Q, __global float* restrict S, __global half* restrict Ubuf, __global half* restrict Wbuf);
__kernel void gdn_seq_kernel_kernel(__global half* restrict A2buf, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global half* restrict KDbuf, __global half* restrict O, __global half* restrict Q, __global float* restrict S, __global half* restrict Ubuf, __global half* restrict Wbuf) { const int tl_gid0 = convert_int(get_group_id(0)); const int tl_gid1 = convert_int(get_group_id(1)); const int tl_lid0 = convert_int(get_local_id(0)); const int tl_wi_aff0 = ((tl_lid0 >> 5) * 1024); const int tl_wi_aff1 = ((tl_lid0 >> 2) * 128); const int tl_wi_aff2 = ((tl_lid0 >> 2) * 32); const int tl_wi_aff3 = ((tl_lid0 & 3) * 8); const int tl_wi_aff4 = (tl_gid1 * 131072); const int tl_wi_aff5 = (tl_gid1 * 16384); const int tl_wi_aff6 = (tl_gid1 * 32768); const int tl_wi_aff7 = (tl_gid1 * 1024); const int tl_wi_aff8 = (tl_gid1 * 32); const int tl_wi_aff9 = (tl_gid0 * 32); const int tl_wi_aff10 = (tl_lid0 * 8);
__local half kdl[4096];
__local half vn[1024];
  for (int c = 0; c < 32; ++c) { const int tl_loop_aff0 = (tl_wi_aff4 + (c * 4096));
    vstore8(vload8(0, KDbuf + (tl_loop_aff0 + tl_wi_aff10)), 0, kdl + tl_wi_aff10);
    vstore8(vload8(0, KDbuf + ((tl_loop_aff0 + tl_wi_aff10) + 1024)), 0, kdl + (tl_wi_aff10 + 1024));
    vstore8(vload8(0, KDbuf + ((tl_loop_aff0 + tl_wi_aff10) + 2048)), 0, kdl + (tl_wi_aff10 + 2048));
    vstore8(vload8(0, KDbuf + ((tl_loop_aff0 + tl_wi_aff10) + 3072)), 0, kdl + (tl_wi_aff10 + 3072));
    barrier(CLK_LOCAL_MEM_FENCE);
    int t = (tl_lid0 >> 2);
    int dvg = (tl_lid0 & 3);
    int dv0 = ((tl_gid0 * 32) + (dvg * 8));
    int cb = (((tl_gid1 * 32) + c) * 4096);
    int qbase = (((tl_gid1 & 15) * 131072) + (c * 4096) + (t * 128));
    float8 acc8 = (float8)(0.000000e+00f);
    float8 acco8 = (float8)(0.000000e+00f);
    for (int dk = 0; dk < 128; ++dk) {
      float8 sv = vload8(0, S + (((tl_gid1 * 16384) + (dk * 128)) + dv0));
      float wv = convert_float(Wbuf[(cb + (t * 128) + dk)]);
      float qv = convert_float(Q[(qbase + dk)]);
      acc8 = (acc8 + (wv * sv));
      acco8 = (acco8 + (qv * sv));
    }
    half8 u8 = vload8(0, Ubuf + (cb + (t * 128) + dv0));
    vstore8(convert_half8(convert_float8(u8) - acc8), 0, vn + (tl_lid0 * 8));
    float4 acco_1_0_lo = acco8.lo;
    float4 acco_1_0_hi = acco8.hi;
    float4 out8_1_lo;
    float4 out8_1_hi;
    barrier(CLK_LOCAL_MEM_FENCE);
    float broadcast_var_4 = 0.000000e+00f;
    out8_1_lo = ((float4)(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4));
    float broadcast_var_5 = 0.000000e+00f;
    out8_1_hi = ((float4)(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5));
    for (int j = 0; j < 32; ++j) {
      if (j <= (tl_lid0 >> 2)) {
        half8 vn_local_cast_2_v = vload8(0, vn + ((j * 32) + tl_wi_aff3));
                out8_1_lo = (((float4)(convert_float(A2buf[(((tl_wi_aff6 + (c * 1024)) + tl_wi_aff2) + j)])) * (convert_float4(vn_local_cast_2_v.lo))) + out8_1_lo);
        out8_1_hi = (((float4)(convert_float(A2buf[(((tl_wi_aff6 + (c * 1024)) + tl_wi_aff2) + j)])) * (convert_float4(vn_local_cast_2_v.hi))) + out8_1_hi);
      }
    }
    vstore8(convert_half8((float8)((out8_1_lo + (((float4)(EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))], EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))], EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))], EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))])) * acco_1_0_lo)), (out8_1_hi + (((float4)(EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))], EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))], EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))], EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))])) * acco_1_0_hi)))), 0, O + ((((tl_wi_aff4 + (c * 4096)) + tl_wi_aff1) + tl_wi_aff9) + tl_wi_aff3));
    barrier(CLK_LOCAL_MEM_FENCE);
    float egl = EglBuf[((tl_gid1 * 32) + c)];
    for (int sb = 0; sb < 4; ++sb) {
      int dk0 = (sb * 32) + ((tl_lid0 >> 5) * 8);
      half8 m8 = (half8)(0.0h);
      #pragma unroll
      for (int tt = 0; tt < 32; ++tt) {
        m8 += vload8(0, kdl + (tt * 128) + dk0) * (half8)vn[(tt * 32) + (tl_lid0 & 31)];
      }
      float8 mf = convert_float8(m8);
      int sbase = (((tl_gid1 * 16384) + (sb * 4096)) + ((tl_lid0 >> 5) * 1024) + (tl_gid0 * 32) + (tl_lid0 & 31));
      S[sbase] = (egl * S[sbase]) + mf.s0;
      S[(sbase + 128)] = (egl * S[(sbase + 128)]) + mf.s1;
      S[(sbase + 256)] = (egl * S[(sbase + 256)]) + mf.s2;
      S[(sbase + 384)] = (egl * S[(sbase + 384)]) + mf.s3;
      S[(sbase + 512)] = (egl * S[(sbase + 512)]) + mf.s4;
      S[(sbase + 640)] = (egl * S[(sbase + 640)]) + mf.s5;
      S[(sbase + 768)] = (egl * S[(sbase + 768)]) + mf.s6;
      S[(sbase + 896)] = (egl * S[(sbase + 896)]) + mf.s7;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

