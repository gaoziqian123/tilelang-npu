// Function: gdn_seq_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void gdn_seq_kernel_kernel(__global half* restrict A2buf, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global half* restrict KDbuf, __global half* restrict O, __global half* restrict Q, __global float* restrict S, __global half* restrict Ubuf, __global half* restrict Wbuf);
__kernel void gdn_seq_kernel_kernel(__global half* restrict A2buf, __global float* restrict EgcBuf, __global float* restrict EglBuf, __global half* restrict KDbuf, __global half* restrict O, __global half* restrict Q, __global float* restrict S, __global half* restrict Ubuf, __global half* restrict Wbuf) { const int tl_gid0 = convert_int(get_group_id(0)); const int tl_gid1 = convert_int(get_group_id(1)); const int tl_lid0 = convert_int(get_local_id(0)); const int tl_wi_aff0 = ((tl_lid0 >> 5) * 1024); const int tl_wi_aff1 = ((tl_lid0 >> 2) * 128); const int tl_wi_aff2 = ((tl_lid0 >> 2) * 32); const int tl_wi_aff3 = ((tl_lid0 & 3) * 8); const int tl_wi_aff4 = (tl_gid1 * 122880); const int tl_wi_aff5 = (tl_gid1 * 30720); const int tl_wi_aff6 = (tl_gid1 * 16384); const int tl_wi_aff7 = (tl_gid1 * 960); const int tl_wi_aff8 = (tl_gid0 * 32); const int tl_wi_aff9 = (tl_gid1 * 30); const int tl_wi_aff10 = (tl_lid0 * 8);
__local half kdl[4096];
__local half vn[1024];
  for (int c = 0; c < 30; ++c) { const int tl_loop_aff0 = (tl_wi_aff4 + (c * 4096));
    vstore8(vload8(0, KDbuf + (tl_loop_aff0 + tl_wi_aff10)), 0, kdl + tl_wi_aff10);
    vstore8(vload8(0, KDbuf + ((tl_loop_aff0 + tl_wi_aff10) + 1024)), 0, kdl + (tl_wi_aff10 + 1024));
    vstore8(vload8(0, KDbuf + ((tl_loop_aff0 + tl_wi_aff10) + 2048)), 0, kdl + (tl_wi_aff10 + 2048));
    vstore8(vload8(0, KDbuf + ((tl_loop_aff0 + tl_wi_aff10) + 3072)), 0, kdl + (tl_wi_aff10 + 3072));
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 acc_1_0_lo = (float4)(0.000000e+00f);
    float4 acc_1_0_hi = (float4)(0.000000e+00f);
    float4 acco_1_0_lo = (float4)(0.000000e+00f);
    float4 acco_1_0_hi = (float4)(0.000000e+00f);
    float broadcast_var = 0.000000e+00f;
    acc_1_0_lo = (((float4)(broadcast_var, broadcast_var, broadcast_var, broadcast_var)));
    float broadcast_var_1 = 0.000000e+00f;
    acco_1_0_lo = (((float4)(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1)));
    float broadcast_var_2 = 0.000000e+00f;
    acc_1_0_hi = (((float4)(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2)));
    float broadcast_var_3 = 0.000000e+00f;
    acco_1_0_hi = (((float4)(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3)));
    for (int dk = 0; dk < 128; ++dk) {
      float wv = (convert_float(Wbuf[((tl_loop_aff0 + tl_wi_aff1) + dk)]));
      float qv = (convert_float(Q[(((((tl_gid1 & 15) * 122880) + (c * 4096)) + tl_wi_aff1) + dk)]));
      float4 sv = vload4(0, S + (((tl_wi_aff6 + (dk * 128)) + tl_wi_aff8) + tl_wi_aff3));
      acc_1_0_lo = ((acc_1_0_lo + (((float4)(wv, wv, wv, wv)) * sv)));
      acco_1_0_lo = ((acco_1_0_lo + (((float4)(qv, qv, qv, qv)) * sv)));
      float4 sv_1 = vload4(0, S + ((((tl_wi_aff6 + (dk * 128)) + tl_wi_aff8) + tl_wi_aff3) + 4));
      acc_1_0_hi = ((acc_1_0_hi + (((float4)(wv, wv, wv, wv)) * sv_1)));
      acco_1_0_hi = ((acco_1_0_hi + (((float4)(qv, qv, qv, qv)) * sv_1)));
    }
    half4 Ubuf_local_cast_1_v = vload4(0, Ubuf + ((((tl_wi_aff4 + (c * 4096)) + tl_wi_aff1) + tl_wi_aff8) + tl_wi_aff3));
    half vn_local_cast[4];
        (*(half4*)(vn_local_cast + 0)) = (convert_half4(((convert_float4(Ubuf_local_cast_1_v)) - acc_1_0_lo)));
    vstore4((*(half4*)(vn_local_cast + 0)), 0, vn + tl_wi_aff10);
    half4 Ubuf_local_cast_1_1_v = vload4(0, Ubuf + (((((tl_wi_aff4 + (c * 4096)) + tl_wi_aff1) + tl_wi_aff8) + tl_wi_aff3) + 4));
    half vn_local_cast_1[4];
        (*(half4*)(vn_local_cast_1 + 0)) = (convert_half4(((convert_float4(Ubuf_local_cast_1_1_v)) - acc_1_0_hi)));
    vstore4((*(half4*)(vn_local_cast_1 + 0)), 0, vn + (tl_wi_aff10 + 4));
    barrier(CLK_LOCAL_MEM_FENCE);
    float out8_1[8];
    float broadcast_var_4 = 0.000000e+00f;
    (*(float4*)(out8_1 + 0)) = ((float4)(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4));
    float broadcast_var_5 = 0.000000e+00f;
    (*(float4*)(out8_1 + 4)) = ((float4)(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5));
    for (int j = 0; j < 32; ++j) {
      if (j <= (tl_lid0 >> 2)) {
        half8 vn_local_cast_2_v = vload8(0, vn + ((j * 32) + tl_wi_aff3));
                (*(float4*)(out8_1 + 0)) = (((float4)(convert_float(A2buf[(((tl_wi_aff5 + (c * 1024)) + tl_wi_aff2) + j)])) * (convert_float4(vn_local_cast_2_v.lo))) + (*(float4*)(out8_1 + 0)));
        (*(float4*)(out8_1 + 4)) = (((float4)(convert_float(A2buf[(((tl_wi_aff5 + (c * 1024)) + tl_wi_aff2) + j)])) * (convert_float4(vn_local_cast_2_v.hi))) + (*(float4*)(out8_1 + 4)));
      }
    }
    for (int dk_1 = 0; dk_1 < 128; ++dk_1) { const int tl_loop_aff0 = (tl_wi_aff6 + (dk_1 * 128));
      float qv_1 = (convert_float(Q[(((((tl_gid1 & 15) * 122880) + (c * 4096)) + tl_wi_aff1) + dk_1)]));
      (*(float4*)(out8_1 + 0)) = ((*(float4*)(out8_1 + 0)) + (((float4)((EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))] * qv_1), (EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))] * qv_1), (EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))] * qv_1), (EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))] * qv_1))) * vload4(0, S + ((tl_loop_aff0 + tl_wi_aff8) + tl_wi_aff3))));
      (*(float4*)(out8_1 + 4)) = ((*(float4*)(out8_1 + 4)) + (((float4)((EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))] * qv_1), (EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))] * qv_1), (EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))] * qv_1), (EgcBuf[((tl_wi_aff7 + (c * 32)) + (tl_lid0 >> 2))] * qv_1))) * vload4(0, S + (((tl_loop_aff0 + tl_wi_aff8) + tl_wi_aff3) + 4))));
    }
    half O_local_cast_3[8];
    (*(half4*)(O_local_cast_3 + 0)) = (convert_half4((*(float4*)(out8_1 + 0))));
    (*(half4*)(O_local_cast_3 + 4)) = (convert_half4((*(float4*)(out8_1 + 4))));
    vstore8((*(half8*)(O_local_cast_3 + 0)), 0, O + ((((tl_wi_aff4 + (c * 4096)) + tl_wi_aff1) + tl_wi_aff8) + tl_wi_aff3));
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int sb = 0; sb < 4; ++sb) {
      float m8_1[8];
      float broadcast_var_6 = 0.000000e+00f;
      (*(float4*)(m8_1 + 0)) = ((float4)(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6));
      float broadcast_var_7 = 0.000000e+00f;
      (*(float4*)(m8_1 + 4)) = ((float4)(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7));
      for (int tt = 0; tt < 32; ++tt) {
        float vnv = (convert_float(vn[((tt * 32) + (tl_lid0 & 31))]));
        half8 kdl_local_cast_4_v = vload8(0, kdl + (((tt * 128) + (sb * 32)) + ((tl_lid0 >> 5) * 8)));
                (*(float4*)(m8_1 + 0)) = ((*(float4*)(m8_1 + 0)) + ((convert_float4(kdl_local_cast_4_v.lo)) * ((float4)(vnv, vnv, vnv, vnv))));
        (*(float4*)(m8_1 + 4)) = ((*(float4*)(m8_1 + 4)) + ((convert_float4(kdl_local_cast_4_v.hi)) * ((float4)(vnv, vnv, vnv, vnv))));
      }
      S[((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31))] = ((EglBuf[(tl_wi_aff9 + c)] * S[((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31))]) + m8_1[0]);
      S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 128)] = ((EglBuf[(tl_wi_aff9 + c)] * S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 128)]) + m8_1[1]);
      S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 256)] = ((EglBuf[(tl_wi_aff9 + c)] * S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 256)]) + m8_1[2]);
      S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 384)] = ((EglBuf[(tl_wi_aff9 + c)] * S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 384)]) + m8_1[3]);
      S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 512)] = ((EglBuf[(tl_wi_aff9 + c)] * S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 512)]) + m8_1[4]);
      S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 640)] = ((EglBuf[(tl_wi_aff9 + c)] * S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 640)]) + m8_1[5]);
      S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 768)] = ((EglBuf[(tl_wi_aff9 + c)] * S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 768)]) + m8_1[6]);
      S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 896)] = ((EglBuf[(tl_wi_aff9 + c)] * S[(((((tl_wi_aff6 + (sb * 4096)) + tl_wi_aff0) + tl_wi_aff8) + (tl_lid0 & 31)) + 896)]) + m8_1[7]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

