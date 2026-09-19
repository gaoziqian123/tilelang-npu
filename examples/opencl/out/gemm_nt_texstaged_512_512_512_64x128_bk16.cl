// Function: gemm_nt_texstaged_kernel_kernel
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

__kernel void gemm_nt_texstaged_kernel_kernel(__global half* restrict A, __read_only image2d_array_t B, __global half* restrict C);
__kernel void gemm_nt_texstaged_kernel_kernel(__global half* restrict A, __read_only image2d_array_t B, __global half* restrict C) {
  const sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  __local uchar buf_dyn_shmem[6144];
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 4096));
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
  for (int ko = 0; ko < 32; ++ko) {
    for (int vi = 0; vi < 2; ++vi) {
      ((half*)A_shared)[(((((convert_int(get_local_id(0))) & 3) * 256) + (vi * 32)) + ((convert_int(get_local_id(0))) >> 2))] = A[((((((convert_int(get_group_id(1))) * 32768) + (vi * 16384)) + (((convert_int(get_local_id(0))) >> 2) * 512)) + (ko * 16)) + (((convert_int(get_local_id(0))) & 3) * 4))];
      ((half*)A_shared)[((((((convert_int(get_local_id(0))) & 3) * 256) + (vi * 32)) + ((convert_int(get_local_id(0))) >> 2)) + 64)] = A[(((((((convert_int(get_group_id(1))) * 32768) + (vi * 16384)) + (((convert_int(get_local_id(0))) >> 2) * 512)) + (ko * 16)) + (((convert_int(get_local_id(0))) & 3) * 4)) + 1)];
      ((half*)A_shared)[((((((convert_int(get_local_id(0))) & 3) * 256) + (vi * 32)) + ((convert_int(get_local_id(0))) >> 2)) + 128)] = A[(((((((convert_int(get_group_id(1))) * 32768) + (vi * 16384)) + (((convert_int(get_local_id(0))) >> 2) * 512)) + (ko * 16)) + (((convert_int(get_local_id(0))) & 3) * 4)) + 2)];
      ((half*)A_shared)[((((((convert_int(get_local_id(0))) & 3) * 256) + (vi * 32)) + ((convert_int(get_local_id(0))) >> 2)) + 192)] = A[(((((((convert_int(get_group_id(1))) * 32768) + (vi * 16384)) + (((convert_int(get_local_id(0))) >> 2) * 512)) + (ko * 16)) + (((convert_int(get_local_id(0))) & 3) * 4)) + 3)];
    }
    for (int tex_i = 0; tex_i < 4; ++tex_i) {
      half4 v_ = as_half4(READ_IMAGEH(B, image_sampler, ((int4)((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) & 31)), (((ko * 16) + (tex_i * 4)) + ((convert_int(get_local_id(0))) >> 5)), 0, 0))));
      vstore4(v_, 0, (half*)B_shared + ((tex_i * 512) + ((convert_int(get_local_id(0))) * 4)));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  for (int kk = 0; kk < 16; ++kk) {
    const int tm = (convert_int(get_local_id(0))) / 16;
    const int tn = (convert_int(get_local_id(0))) - tm * 16;
    float8 a8 = convert_float8(vload8(0, (half*)A_shared + kk * 64 + tm * 8));
    float8 b8 = convert_float8(vload8(0, (half*)B_shared + kk * 128 + tn * 8));
    acc_0 += (float8)(a8.s0) * b8;
    acc_1 += (float8)(a8.s1) * b8;
    acc_2 += (float8)(a8.s2) * b8;
    acc_3 += (float8)(a8.s3) * b8;
    acc_4 += (float8)(a8.s4) * b8;
    acc_5 += (float8)(a8.s5) * b8;
    acc_6 += (float8)(a8.s6) * b8;
    acc_7 += (float8)(a8.s7) * b8;
  }
barrier(CLK_LOCAL_MEM_FENCE);
  }
  C[(((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8))] = (convert_half(acc_0.s0));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1)] = (convert_half(acc_0.s1));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2)] = (convert_half(acc_0.s2));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3)] = (convert_half(acc_0.s3));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 4)] = (convert_half(acc_0.s4));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 5)] = (convert_half(acc_0.s5));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 6)] = (convert_half(acc_0.s6));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 7)] = (convert_half(acc_0.s7));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 512)] = (convert_half(acc_1.s0));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 513)] = (convert_half(acc_1.s1));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 514)] = (convert_half(acc_1.s2));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 515)] = (convert_half(acc_1.s3));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 516)] = (convert_half(acc_1.s4));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 517)] = (convert_half(acc_1.s5));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 518)] = (convert_half(acc_1.s6));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 519)] = (convert_half(acc_1.s7));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1024)] = (convert_half(acc_2.s0));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1025)] = (convert_half(acc_2.s1));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1026)] = (convert_half(acc_2.s2));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1027)] = (convert_half(acc_2.s3));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1028)] = (convert_half(acc_2.s4));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1029)] = (convert_half(acc_2.s5));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1030)] = (convert_half(acc_2.s6));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1031)] = (convert_half(acc_2.s7));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1536)] = (convert_half(acc_3.s0));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1537)] = (convert_half(acc_3.s1));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1538)] = (convert_half(acc_3.s2));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1539)] = (convert_half(acc_3.s3));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1540)] = (convert_half(acc_3.s4));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1541)] = (convert_half(acc_3.s5));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1542)] = (convert_half(acc_3.s6));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 1543)] = (convert_half(acc_3.s7));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2048)] = (convert_half(acc_4.s0));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2049)] = (convert_half(acc_4.s1));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2050)] = (convert_half(acc_4.s2));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2051)] = (convert_half(acc_4.s3));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2052)] = (convert_half(acc_4.s4));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2053)] = (convert_half(acc_4.s5));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2054)] = (convert_half(acc_4.s6));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2055)] = (convert_half(acc_4.s7));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2560)] = (convert_half(acc_5.s0));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2561)] = (convert_half(acc_5.s1));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2562)] = (convert_half(acc_5.s2));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2563)] = (convert_half(acc_5.s3));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2564)] = (convert_half(acc_5.s4));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2565)] = (convert_half(acc_5.s5));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2566)] = (convert_half(acc_5.s6));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 2567)] = (convert_half(acc_5.s7));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3072)] = (convert_half(acc_6.s0));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3073)] = (convert_half(acc_6.s1));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3074)] = (convert_half(acc_6.s2));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3075)] = (convert_half(acc_6.s3));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3076)] = (convert_half(acc_6.s4));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3077)] = (convert_half(acc_6.s5));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3078)] = (convert_half(acc_6.s6));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3079)] = (convert_half(acc_6.s7));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3584)] = (convert_half(acc_7.s0));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3585)] = (convert_half(acc_7.s1));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3586)] = (convert_half(acc_7.s2));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3587)] = (convert_half(acc_7.s3));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3588)] = (convert_half(acc_7.s4));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3589)] = (convert_half(acc_7.s5));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3590)] = (convert_half(acc_7.s6));
  C[((((((convert_int(get_group_id(1))) * 32768) + (((convert_int(get_local_id(0))) >> 4) * 4096)) + ((convert_int(get_group_id(0))) * 128)) + (((convert_int(get_local_id(0))) & 15) * 8)) + 3591)] = (convert_half(acc_7.s7));
}

