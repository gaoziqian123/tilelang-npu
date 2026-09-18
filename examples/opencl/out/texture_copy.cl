// Function: texture_copy_kernel_kernel
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

__kernel void texture_copy_kernel_kernel(__read_only image2d_array_t B, __global half* restrict O);
__kernel void texture_copy_kernel_kernel(__read_only image2d_array_t B, __global half* restrict O) {
  const sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  half4 v_ = as_half4(READ_IMAGEH(B, image_sampler, ((int4)((convert_int(get_group_id(0))), (convert_int(get_group_id(1))), 0, 0))));
  O[((((convert_int(get_group_id(1))) * 512) + ((convert_int(get_group_id(0))) * 4)) + (convert_int(get_local_id(0))))] = (((half*)&v_)[(convert_int(get_local_id(0)))] * (half)2.000000e+00f);
}

