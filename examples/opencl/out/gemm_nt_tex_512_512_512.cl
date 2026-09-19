// Function: gemm_nt_tex_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

#ifdef __OPENCL_VERSION__
#if __OPENCL_VERSION__ == CL_VERSION_2_0 || __OPENCL_VERSION__ == CL_VERSION_3_0 
#define READ_IMAGEH(image, sampler, coord) read_imageh(image, sampler, coord)
#define READ_IMAGEF(image, sampler, coord) read_imagef(image, sampler, coord)
#else
#define READ_IMAGEH(image, sampler, coord) read_imageh(image, coord)
#define READ_IMAGEF(image, sampler, coord) read_imagef(image, coord)
#endif
#endif

__kernel void gemm_nt_tex_kernel_kernel(__global half* restrict A, __read_only image2d_array_t B, __global half* restrict C);
__kernel void gemm_nt_tex_kernel_kernel(__global half* restrict A, __read_only image2d_array_t B, __global half* restrict C) {
  const sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  float8 acc_0 = (float8)(0.000000e+00f);
  float8 acc_1 = (float8)(0.000000e+00f);
  float8 acc_2 = (float8)(0.000000e+00f);
  float8 acc_3 = (float8)(0.000000e+00f);
  float8 acc_4 = (float8)(0.000000e+00f);
  float8 acc_5 = (float8)(0.000000e+00f);
  float8 acc_6 = (float8)(0.000000e+00f);
  float8 acc_7 = (float8)(0.000000e+00f);
  acc_0.s0 = 0.000000e+00f;
  acc_0.s1 = 0.000000e+00f;
  acc_0.s2 = 0.000000e+00f;
  acc_0.s3 = 0.000000e+00f;
  acc_0.s4 = 0.000000e+00f;
  acc_0.s5 = 0.000000e+00f;
  acc_0.s6 = 0.000000e+00f;
  acc_0.s7 = 0.000000e+00f;
  acc_1.s0 = 0.000000e+00f;
  acc_1.s1 = 0.000000e+00f;
  acc_1.s2 = 0.000000e+00f;
  acc_1.s3 = 0.000000e+00f;
  acc_1.s4 = 0.000000e+00f;
  acc_1.s5 = 0.000000e+00f;
  acc_1.s6 = 0.000000e+00f;
  acc_1.s7 = 0.000000e+00f;
  acc_2.s0 = 0.000000e+00f;
  acc_2.s1 = 0.000000e+00f;
  acc_2.s2 = 0.000000e+00f;
  acc_2.s3 = 0.000000e+00f;
  acc_2.s4 = 0.000000e+00f;
  acc_2.s5 = 0.000000e+00f;
  acc_2.s6 = 0.000000e+00f;
  acc_2.s7 = 0.000000e+00f;
  acc_3.s0 = 0.000000e+00f;
  acc_3.s1 = 0.000000e+00f;
  acc_3.s2 = 0.000000e+00f;
  acc_3.s3 = 0.000000e+00f;
  acc_3.s4 = 0.000000e+00f;
  acc_3.s5 = 0.000000e+00f;
  acc_3.s6 = 0.000000e+00f;
  acc_3.s7 = 0.000000e+00f;
  acc_4.s0 = 0.000000e+00f;
  acc_4.s1 = 0.000000e+00f;
  acc_4.s2 = 0.000000e+00f;
  acc_4.s3 = 0.000000e+00f;
  acc_4.s4 = 0.000000e+00f;
  acc_4.s5 = 0.000000e+00f;
  acc_4.s6 = 0.000000e+00f;
  acc_4.s7 = 0.000000e+00f;
  acc_5.s0 = 0.000000e+00f;
  acc_5.s1 = 0.000000e+00f;
  acc_5.s2 = 0.000000e+00f;
  acc_5.s3 = 0.000000e+00f;
  acc_5.s4 = 0.000000e+00f;
  acc_5.s5 = 0.000000e+00f;
  acc_5.s6 = 0.000000e+00f;
  acc_5.s7 = 0.000000e+00f;
  acc_6.s0 = 0.000000e+00f;
  acc_6.s1 = 0.000000e+00f;
  acc_6.s2 = 0.000000e+00f;
  acc_6.s3 = 0.000000e+00f;
  acc_6.s4 = 0.000000e+00f;
  acc_6.s5 = 0.000000e+00f;
  acc_6.s6 = 0.000000e+00f;
  acc_6.s7 = 0.000000e+00f;
  acc_7.s0 = 0.000000e+00f;
  acc_7.s1 = 0.000000e+00f;
  acc_7.s2 = 0.000000e+00f;
  acc_7.s3 = 0.000000e+00f;
  acc_7.s4 = 0.000000e+00f;
  acc_7.s5 = 0.000000e+00f;
  acc_7.s6 = 0.000000e+00f;
  acc_7.s7 = 0.000000e+00f;
  for (int _tmp = 0; _tmp < 128; ++_tmp) {
    float avec[32];
    avec[0] = (convert_float(A[(((convert_int(get_group_id(1))) * 4096) + (_tmp * 4))]));
    avec[1] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 1)]));
    avec[2] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 2)]));
    avec[3] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 3)]));
    avec[4] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 512)]));
    avec[5] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 513)]));
    avec[6] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 514)]));
    avec[7] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 515)]));
    avec[8] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 1024)]));
    avec[9] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 1025)]));
    avec[10] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 1026)]));
    avec[11] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 1027)]));
    avec[12] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 1536)]));
    avec[13] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 1537)]));
    avec[14] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 1538)]));
    avec[15] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 1539)]));
    avec[16] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 2048)]));
    avec[17] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 2049)]));
    avec[18] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 2050)]));
    avec[19] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 2051)]));
    avec[20] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 2560)]));
    avec[21] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 2561)]));
    avec[22] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 2562)]));
    avec[23] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 2563)]));
    avec[24] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 3072)]));
    avec[25] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 3073)]));
    avec[26] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 3074)]));
    avec[27] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 3075)]));
    avec[28] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 3584)]));
    avec[29] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 3585)]));
    avec[30] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 3586)]));
    avec[31] = (convert_float(A[((((convert_int(get_group_id(1))) * 4096) + (_tmp * 4)) + 3587)]));
    for (int kk = 0; kk < 4; ++kk) {
      half4 v_ = as_half4(READ_IMAGEH(B, image_sampler, ((int4)(((convert_int(get_group_id(0))) * 2), ((_tmp * 4) + kk), 0, 0))));
      acc_0.s0 = (acc_0.s0 + (avec[kk] * (convert_float(v_.s0))));
      acc_0.s1 = (acc_0.s1 + (avec[kk] * (convert_float(v_.s1))));
      acc_0.s2 = (acc_0.s2 + (avec[kk] * (convert_float(v_.s2))));
      acc_0.s3 = (acc_0.s3 + (avec[kk] * (convert_float(v_.s3))));
      half4 v__1 = as_half4(READ_IMAGEH(B, image_sampler, ((int4)((((convert_int(get_group_id(0))) * 2) + 1), ((_tmp * 4) + kk), 0, 0))));
      acc_0.s4 = (acc_0.s4 + (avec[kk] * (convert_float(v__1.s0))));
      acc_0.s5 = (acc_0.s5 + (avec[kk] * (convert_float(v__1.s1))));
      acc_0.s6 = (acc_0.s6 + (avec[kk] * (convert_float(v__1.s2))));
      acc_0.s7 = (acc_0.s7 + (avec[kk] * (convert_float(v__1.s3))));
      acc_1.s0 = (acc_1.s0 + (avec[(kk + 4)] * (convert_float(v_.s0))));
      acc_1.s1 = (acc_1.s1 + (avec[(kk + 4)] * (convert_float(v_.s1))));
      acc_1.s2 = (acc_1.s2 + (avec[(kk + 4)] * (convert_float(v_.s2))));
      acc_1.s3 = (acc_1.s3 + (avec[(kk + 4)] * (convert_float(v_.s3))));
      acc_1.s4 = (acc_1.s4 + (avec[(kk + 4)] * (convert_float(v__1.s0))));
      acc_1.s5 = (acc_1.s5 + (avec[(kk + 4)] * (convert_float(v__1.s1))));
      acc_1.s6 = (acc_1.s6 + (avec[(kk + 4)] * (convert_float(v__1.s2))));
      acc_1.s7 = (acc_1.s7 + (avec[(kk + 4)] * (convert_float(v__1.s3))));
      acc_2.s0 = (acc_2.s0 + (avec[(kk + 8)] * (convert_float(v_.s0))));
      acc_2.s1 = (acc_2.s1 + (avec[(kk + 8)] * (convert_float(v_.s1))));
      acc_2.s2 = (acc_2.s2 + (avec[(kk + 8)] * (convert_float(v_.s2))));
      acc_2.s3 = (acc_2.s3 + (avec[(kk + 8)] * (convert_float(v_.s3))));
      acc_2.s4 = (acc_2.s4 + (avec[(kk + 8)] * (convert_float(v__1.s0))));
      acc_2.s5 = (acc_2.s5 + (avec[(kk + 8)] * (convert_float(v__1.s1))));
      acc_2.s6 = (acc_2.s6 + (avec[(kk + 8)] * (convert_float(v__1.s2))));
      acc_2.s7 = (acc_2.s7 + (avec[(kk + 8)] * (convert_float(v__1.s3))));
      acc_3.s0 = (acc_3.s0 + (avec[(kk + 12)] * (convert_float(v_.s0))));
      acc_3.s1 = (acc_3.s1 + (avec[(kk + 12)] * (convert_float(v_.s1))));
      acc_3.s2 = (acc_3.s2 + (avec[(kk + 12)] * (convert_float(v_.s2))));
      acc_3.s3 = (acc_3.s3 + (avec[(kk + 12)] * (convert_float(v_.s3))));
      acc_3.s4 = (acc_3.s4 + (avec[(kk + 12)] * (convert_float(v__1.s0))));
      acc_3.s5 = (acc_3.s5 + (avec[(kk + 12)] * (convert_float(v__1.s1))));
      acc_3.s6 = (acc_3.s6 + (avec[(kk + 12)] * (convert_float(v__1.s2))));
      acc_3.s7 = (acc_3.s7 + (avec[(kk + 12)] * (convert_float(v__1.s3))));
      acc_4.s0 = (acc_4.s0 + (avec[(kk + 16)] * (convert_float(v_.s0))));
      acc_4.s1 = (acc_4.s1 + (avec[(kk + 16)] * (convert_float(v_.s1))));
      acc_4.s2 = (acc_4.s2 + (avec[(kk + 16)] * (convert_float(v_.s2))));
      acc_4.s3 = (acc_4.s3 + (avec[(kk + 16)] * (convert_float(v_.s3))));
      acc_4.s4 = (acc_4.s4 + (avec[(kk + 16)] * (convert_float(v__1.s0))));
      acc_4.s5 = (acc_4.s5 + (avec[(kk + 16)] * (convert_float(v__1.s1))));
      acc_4.s6 = (acc_4.s6 + (avec[(kk + 16)] * (convert_float(v__1.s2))));
      acc_4.s7 = (acc_4.s7 + (avec[(kk + 16)] * (convert_float(v__1.s3))));
      acc_5.s0 = (acc_5.s0 + (avec[(kk + 20)] * (convert_float(v_.s0))));
      acc_5.s1 = (acc_5.s1 + (avec[(kk + 20)] * (convert_float(v_.s1))));
      acc_5.s2 = (acc_5.s2 + (avec[(kk + 20)] * (convert_float(v_.s2))));
      acc_5.s3 = (acc_5.s3 + (avec[(kk + 20)] * (convert_float(v_.s3))));
      acc_5.s4 = (acc_5.s4 + (avec[(kk + 20)] * (convert_float(v__1.s0))));
      acc_5.s5 = (acc_5.s5 + (avec[(kk + 20)] * (convert_float(v__1.s1))));
      acc_5.s6 = (acc_5.s6 + (avec[(kk + 20)] * (convert_float(v__1.s2))));
      acc_5.s7 = (acc_5.s7 + (avec[(kk + 20)] * (convert_float(v__1.s3))));
      acc_6.s0 = (acc_6.s0 + (avec[(kk + 24)] * (convert_float(v_.s0))));
      acc_6.s1 = (acc_6.s1 + (avec[(kk + 24)] * (convert_float(v_.s1))));
      acc_6.s2 = (acc_6.s2 + (avec[(kk + 24)] * (convert_float(v_.s2))));
      acc_6.s3 = (acc_6.s3 + (avec[(kk + 24)] * (convert_float(v_.s3))));
      acc_6.s4 = (acc_6.s4 + (avec[(kk + 24)] * (convert_float(v__1.s0))));
      acc_6.s5 = (acc_6.s5 + (avec[(kk + 24)] * (convert_float(v__1.s1))));
      acc_6.s6 = (acc_6.s6 + (avec[(kk + 24)] * (convert_float(v__1.s2))));
      acc_6.s7 = (acc_6.s7 + (avec[(kk + 24)] * (convert_float(v__1.s3))));
      acc_7.s0 = (acc_7.s0 + (avec[(kk + 28)] * (convert_float(v_.s0))));
      acc_7.s1 = (acc_7.s1 + (avec[(kk + 28)] * (convert_float(v_.s1))));
      acc_7.s2 = (acc_7.s2 + (avec[(kk + 28)] * (convert_float(v_.s2))));
      acc_7.s3 = (acc_7.s3 + (avec[(kk + 28)] * (convert_float(v_.s3))));
      acc_7.s4 = (acc_7.s4 + (avec[(kk + 28)] * (convert_float(v__1.s0))));
      acc_7.s5 = (acc_7.s5 + (avec[(kk + 28)] * (convert_float(v__1.s1))));
      acc_7.s6 = (acc_7.s6 + (avec[(kk + 28)] * (convert_float(v__1.s2))));
      acc_7.s7 = (acc_7.s7 + (avec[(kk + 28)] * (convert_float(v__1.s3))));
    }
  }
  C[(((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8))] = (convert_half(acc_0.s0));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1)] = (convert_half(acc_0.s1));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2)] = (convert_half(acc_0.s2));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3)] = (convert_half(acc_0.s3));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 4)] = (convert_half(acc_0.s4));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 5)] = (convert_half(acc_0.s5));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 6)] = (convert_half(acc_0.s6));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 7)] = (convert_half(acc_0.s7));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 512)] = (convert_half(acc_1.s0));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 513)] = (convert_half(acc_1.s1));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 514)] = (convert_half(acc_1.s2));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 515)] = (convert_half(acc_1.s3));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 516)] = (convert_half(acc_1.s4));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 517)] = (convert_half(acc_1.s5));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 518)] = (convert_half(acc_1.s6));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 519)] = (convert_half(acc_1.s7));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1024)] = (convert_half(acc_2.s0));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1025)] = (convert_half(acc_2.s1));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1026)] = (convert_half(acc_2.s2));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1027)] = (convert_half(acc_2.s3));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1028)] = (convert_half(acc_2.s4));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1029)] = (convert_half(acc_2.s5));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1030)] = (convert_half(acc_2.s6));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1031)] = (convert_half(acc_2.s7));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1536)] = (convert_half(acc_3.s0));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1537)] = (convert_half(acc_3.s1));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1538)] = (convert_half(acc_3.s2));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1539)] = (convert_half(acc_3.s3));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1540)] = (convert_half(acc_3.s4));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1541)] = (convert_half(acc_3.s5));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1542)] = (convert_half(acc_3.s6));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 1543)] = (convert_half(acc_3.s7));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2048)] = (convert_half(acc_4.s0));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2049)] = (convert_half(acc_4.s1));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2050)] = (convert_half(acc_4.s2));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2051)] = (convert_half(acc_4.s3));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2052)] = (convert_half(acc_4.s4));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2053)] = (convert_half(acc_4.s5));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2054)] = (convert_half(acc_4.s6));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2055)] = (convert_half(acc_4.s7));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2560)] = (convert_half(acc_5.s0));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2561)] = (convert_half(acc_5.s1));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2562)] = (convert_half(acc_5.s2));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2563)] = (convert_half(acc_5.s3));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2564)] = (convert_half(acc_5.s4));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2565)] = (convert_half(acc_5.s5));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2566)] = (convert_half(acc_5.s6));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 2567)] = (convert_half(acc_5.s7));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3072)] = (convert_half(acc_6.s0));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3073)] = (convert_half(acc_6.s1));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3074)] = (convert_half(acc_6.s2));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3075)] = (convert_half(acc_6.s3));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3076)] = (convert_half(acc_6.s4));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3077)] = (convert_half(acc_6.s5));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3078)] = (convert_half(acc_6.s6));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3079)] = (convert_half(acc_6.s7));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3584)] = (convert_half(acc_7.s0));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3585)] = (convert_half(acc_7.s1));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3586)] = (convert_half(acc_7.s2));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3587)] = (convert_half(acc_7.s3));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3588)] = (convert_half(acc_7.s4));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3589)] = (convert_half(acc_7.s5));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3590)] = (convert_half(acc_7.s6));
  C[((((convert_int(get_group_id(1))) * 4096) + ((convert_int(get_group_id(0))) * 8)) + 3591)] = (convert_half(acc_7.s7));
}

