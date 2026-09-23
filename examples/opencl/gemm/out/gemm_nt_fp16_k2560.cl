// Function: gemm_nt_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void gemm_nt_kernel_kernel(__global half* restrict A, __global half* restrict B, __global half* restrict C);
__kernel void gemm_nt_kernel_kernel(__global half* restrict A, __global half* restrict B, __global half* restrict C) { const int tl_gid0 = convert_int(get_group_id(0)); const int tl_gid1 = convert_int(get_group_id(1)); const int tl_lid0 = convert_int(get_local_id(0)); const int tl_wi_aff0 = ((tl_gid1 * 81920) + ((tl_lid0 >> 4) * 20480)); const int tl_wi_aff1 = ((tl_wi_aff0 + (tl_gid0 * 128)) + ((tl_lid0 & 15) * 8)); const int tl_wi_aff2 = ((tl_lid0 >> 4) * 128); const int tl_wi_aff3 = (tl_gid0 * 128); const int tl_wi_aff4 = (tl_lid0 * 8); __global half* tl_C_base = C + tl_wi_aff1;
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
  for (int pos4 = 0; pos4 < 640; ++pos4) { const int tl_loop_aff0 = (tl_wi_aff0 + (pos4 * 4)); const int tl_loop_aff1 = (((pos4 * 10240) + tl_wi_aff3) + tl_wi_aff4); __global half* tl_Ap = A + tl_loop_aff0; __global half* tl_Bp = B + (tl_loop_aff1 - tl_wi_aff2);
    half4 av_1[8];
    half8 bv_1[4];
    av_1[0] = vload4(0, tl_Ap);
    av_1[1] = vload4(0, tl_Ap + ( 2560));
    av_1[2] = vload4(0, tl_Ap + ( 5120));
    av_1[3] = vload4(0, tl_Ap + ( 7680));
    av_1[4] = vload4(0, tl_Ap + ( 10240));
    av_1[5] = vload4(0, tl_Ap + ( 12800));
    av_1[6] = vload4(0, tl_Ap + ( 15360));
    av_1[7] = vload4(0, tl_Ap + ( 17920));
    bv_1[0] = vload8(0, tl_Bp);
    bv_1[1] = vload8(0, tl_Bp + 2560);
    bv_1[2] = vload8(0, tl_Bp + 5120);
    bv_1[3] = vload8(0, tl_Bp + 7680);
    acc_1[0] = mad(((half8)((av_1[0]).s0)), bv_1[0], mad(((half8)((av_1[0]).s1)), bv_1[1], mad(((half8)((av_1[0]).s2)), bv_1[2], mad(((half8)((av_1[0]).s3)), bv_1[3], acc_1[0]))));
    acc_1[1] = mad(((half8)((av_1[1]).s0)), bv_1[0], mad(((half8)((av_1[1]).s1)), bv_1[1], mad(((half8)((av_1[1]).s2)), bv_1[2], mad(((half8)((av_1[1]).s3)), bv_1[3], acc_1[1]))));
    acc_1[2] = mad(((half8)((av_1[2]).s0)), bv_1[0], mad(((half8)((av_1[2]).s1)), bv_1[1], mad(((half8)((av_1[2]).s2)), bv_1[2], mad(((half8)((av_1[2]).s3)), bv_1[3], acc_1[2]))));
    acc_1[3] = mad(((half8)((av_1[3]).s0)), bv_1[0], mad(((half8)((av_1[3]).s1)), bv_1[1], mad(((half8)((av_1[3]).s2)), bv_1[2], mad(((half8)((av_1[3]).s3)), bv_1[3], acc_1[3]))));
    acc_1[4] = mad(((half8)((av_1[4]).s0)), bv_1[0], mad(((half8)((av_1[4]).s1)), bv_1[1], mad(((half8)((av_1[4]).s2)), bv_1[2], mad(((half8)((av_1[4]).s3)), bv_1[3], acc_1[4]))));
    acc_1[5] = mad(((half8)((av_1[5]).s0)), bv_1[0], mad(((half8)((av_1[5]).s1)), bv_1[1], mad(((half8)((av_1[5]).s2)), bv_1[2], mad(((half8)((av_1[5]).s3)), bv_1[3], acc_1[5]))));
    acc_1[6] = mad(((half8)((av_1[6]).s0)), bv_1[0], mad(((half8)((av_1[6]).s1)), bv_1[1], mad(((half8)((av_1[6]).s2)), bv_1[2], mad(((half8)((av_1[6]).s3)), bv_1[3], acc_1[6]))));
    acc_1[7] = mad(((half8)((av_1[7]).s0)), bv_1[0], mad(((half8)((av_1[7]).s1)), bv_1[1], mad(((half8)((av_1[7]).s2)), bv_1[2], mad(((half8)((av_1[7]).s3)), bv_1[3], acc_1[7]))));
  }
  vstore8(acc_1[0], 0, C + tl_wi_aff1);

  vstore8(acc_1[1], 0, tl_C_base + ( + 2560));

  vstore8(acc_1[2], 0, tl_C_base + ( + 5120));

  vstore8(acc_1[3], 0, tl_C_base + ( + 7680));

  vstore8(acc_1[4], 0, tl_C_base + ( + 10240));

  vstore8(acc_1[5], 0, tl_C_base + ( + 12800));

  vstore8(acc_1[6], 0, tl_C_base + ( + 15360));

  vstore8(acc_1[7], 0, tl_C_base + ( + 17920));

}

