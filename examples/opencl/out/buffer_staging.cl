// Function: buffer_staging_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void buffer_staging_kernel_kernel(__global float* restrict O, __global half* restrict X);
__kernel void buffer_staging_kernel_kernel(__global float* restrict O, __global half* restrict X) {
  __local uchar buf_dyn_shmem[2048];
  void* staged = ((void*)((char*)buf_dyn_shmem + 0));
  void* partial = ((void*)((char*)buf_dyn_shmem + 1024));
  if ((convert_int(get_local_id(0))) < 64) {
    vstore8(vload8(0, X + (((convert_int(get_group_id(0))) * 512) + ((convert_int(get_local_id(0))) * 8))), 0, (half*)staged + ((convert_int(get_local_id(0))) * 8));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  ((float*)partial)[(convert_int(get_local_id(0)))] = 0.000000e+00f;
  for (int i = 0; i < 2; ++i) {
    ((float*)partial)[(convert_int(get_local_id(0)))] = (((float*)partial)[(convert_int(get_local_id(0)))] + (convert_float(((half*)staged)[((i * 256) + (convert_int(get_local_id(0))))])));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int step = 0; step < 8; ++step) {
    if ((convert_int(get_local_id(0))) < (128 >> step)) {
      ((float*)partial)[(convert_int(get_local_id(0)))] = (((float*)partial)[(convert_int(get_local_id(0)))] + ((float*)partial)[((convert_int(get_local_id(0))) + (128 >> step))]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((convert_int(get_local_id(0))) == 0) {
    O[(convert_int(get_group_id(0)))] = ((float*)partial)[0];
  }
}

