// Function: texture_staging_kernel_kernel
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

__kernel void texture_staging_kernel_kernel(__global float* restrict O, __read_only image2d_array_t X);
__kernel void texture_staging_kernel_kernel(__global float* restrict O, __read_only image2d_array_t X) {
  const sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  __local uchar buf_dyn_shmem[2048];
  void* staged = ((void*)((char*)buf_dyn_shmem + 0));
  void* partial = ((void*)((char*)buf_dyn_shmem + 1024));
  if ((convert_int(get_local_id(0))) < 128) {
    half4 v_ = as_half4(READ_IMAGEH(X, image_sampler, ((int4)((convert_int(get_local_id(0))), (convert_int(get_group_id(0))), 0, 0))));
    vstore4(v_, 0, (half*)staged + ((convert_int(get_local_id(0))) * 4));
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

