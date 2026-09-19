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
  for (int _tmp = 0; _tmp < 640; ++_tmp) {
    float avec[32];
    avec[0] = (convert_float(A[(((convert_int(get_group_id(1))) * 20480) + (_tmp * 4))]));
    avec[1] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 1)]));
    avec[2] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 2)]));
    avec[3] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 3)]));
    avec[4] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 2560)]));
    avec[5] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 2561)]));
    avec[6] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 2562)]));
    avec[7] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 2563)]));
    avec[8] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 5120)]));
    avec[9] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 5121)]));
    avec[10] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 5122)]));
    avec[11] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 5123)]));
    avec[12] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 7680)]));
    avec[13] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 7681)]));
    avec[14] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 7682)]));
    avec[15] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 7683)]));
    avec[16] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 10240)]));
    avec[17] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 10241)]));
    avec[18] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 10242)]));
    avec[19] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 10243)]));
    avec[20] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 12800)]));
    avec[21] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 12801)]));
    avec[22] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 12802)]));
    avec[23] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 12803)]));
    avec[24] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 15360)]));
    avec[25] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 15361)]));
    avec[26] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 15362)]));
    avec[27] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 15363)]));
    avec[28] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 17920)]));
    avec[29] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 17921)]));
    avec[30] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 17922)]));
    avec[31] = (convert_float(A[((((convert_int(get_group_id(1))) * 20480) + (_tmp * 4)) + 17923)]));
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
  C[(((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8))] = (convert_half(acc_0.s0));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 1)] = (convert_half(acc_0.s1));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 2)] = (convert_half(acc_0.s2));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 3)] = (convert_half(acc_0.s3));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 4)] = (convert_half(acc_0.s4));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 5)] = (convert_half(acc_0.s5));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 6)] = (convert_half(acc_0.s6));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 7)] = (convert_half(acc_0.s7));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 2560)] = (convert_half(acc_1.s0));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 2561)] = (convert_half(acc_1.s1));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 2562)] = (convert_half(acc_1.s2));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 2563)] = (convert_half(acc_1.s3));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 2564)] = (convert_half(acc_1.s4));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 2565)] = (convert_half(acc_1.s5));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 2566)] = (convert_half(acc_1.s6));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 2567)] = (convert_half(acc_1.s7));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 5120)] = (convert_half(acc_2.s0));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 5121)] = (convert_half(acc_2.s1));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 5122)] = (convert_half(acc_2.s2));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 5123)] = (convert_half(acc_2.s3));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 5124)] = (convert_half(acc_2.s4));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 5125)] = (convert_half(acc_2.s5));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 5126)] = (convert_half(acc_2.s6));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 5127)] = (convert_half(acc_2.s7));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 7680)] = (convert_half(acc_3.s0));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 7681)] = (convert_half(acc_3.s1));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 7682)] = (convert_half(acc_3.s2));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 7683)] = (convert_half(acc_3.s3));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 7684)] = (convert_half(acc_3.s4));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 7685)] = (convert_half(acc_3.s5));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 7686)] = (convert_half(acc_3.s6));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 7687)] = (convert_half(acc_3.s7));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 10240)] = (convert_half(acc_4.s0));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 10241)] = (convert_half(acc_4.s1));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 10242)] = (convert_half(acc_4.s2));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 10243)] = (convert_half(acc_4.s3));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 10244)] = (convert_half(acc_4.s4));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 10245)] = (convert_half(acc_4.s5));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 10246)] = (convert_half(acc_4.s6));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 10247)] = (convert_half(acc_4.s7));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 12800)] = (convert_half(acc_5.s0));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 12801)] = (convert_half(acc_5.s1));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 12802)] = (convert_half(acc_5.s2));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 12803)] = (convert_half(acc_5.s3));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 12804)] = (convert_half(acc_5.s4));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 12805)] = (convert_half(acc_5.s5));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 12806)] = (convert_half(acc_5.s6));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 12807)] = (convert_half(acc_5.s7));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 15360)] = (convert_half(acc_6.s0));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 15361)] = (convert_half(acc_6.s1));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 15362)] = (convert_half(acc_6.s2));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 15363)] = (convert_half(acc_6.s3));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 15364)] = (convert_half(acc_6.s4));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 15365)] = (convert_half(acc_6.s5));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 15366)] = (convert_half(acc_6.s6));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 15367)] = (convert_half(acc_6.s7));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 17920)] = (convert_half(acc_7.s0));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 17921)] = (convert_half(acc_7.s1));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 17922)] = (convert_half(acc_7.s2));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 17923)] = (convert_half(acc_7.s3));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 17924)] = (convert_half(acc_7.s4));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 17925)] = (convert_half(acc_7.s5));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 17926)] = (convert_half(acc_7.s6));
  C[((((convert_int(get_group_id(1))) * 20480) + ((convert_int(get_group_id(0))) * 8)) + 17927)] = (convert_half(acc_7.s7));
}

