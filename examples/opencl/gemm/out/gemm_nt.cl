// Function: gemm_nt_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void gemm_nt_kernel_kernel(__global half* restrict A, __global half* restrict B, __global half* restrict C);
__kernel void gemm_nt_kernel_kernel(__global half* restrict A, __global half* restrict B, __global half* restrict C) {
  __local uchar buf_dyn_shmem[2048];
  void* C_accum = ((void*)((char*)buf_dyn_shmem + 0));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 1024));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 1536));
  ((float*)C_accum)[(convert_int(get_local_id(0)))] = 0.000000e+00f;
  for (int ko = 0; ko < 32; ++ko) {
    ((half*)A_shared)[(convert_int(get_local_id(0)))] = A[(((((convert_int(get_group_id(1))) * 8192) + (((convert_int(get_local_id(0))) >> 4) * 512)) + (ko * 16)) + ((convert_int(get_local_id(0))) & 15))];
    ((half*)B_shared)[(convert_int(get_local_id(0)))] = B[(((((convert_int(get_group_id(0))) * 8192) + (((convert_int(get_local_id(0))) >> 4) * 512)) + (ko * 16)) + ((convert_int(get_local_id(0))) & 15))];
    barrier(CLK_LOCAL_MEM_FENCE);
    float accum[1];
    accum[0] = ((float*)C_accum)[(convert_int(get_local_id(0)))];
    for (int b_j = 0; b_j < 16; ++b_j) {
      accum[0] = (accum[0] + ((convert_float(((half*)A_shared)[((((convert_int(get_local_id(0))) >> 4) * 16) + b_j)])) * (convert_float(((half*)B_shared)[((((convert_int(get_local_id(0))) & 15) * 16) + b_j)]))));
    }
    ((float*)C_accum)[(convert_int(get_local_id(0)))] = accum[0];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  C[(((((convert_int(get_group_id(1))) * 8192) + (((convert_int(get_local_id(0))) >> 4) * 512)) + ((convert_int(get_group_id(0))) * 16)) + ((convert_int(get_local_id(0))) & 15))] = (convert_half(((float*)C_accum)[(convert_int(get_local_id(0)))]));
}

