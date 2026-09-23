// Function: silu_mul_kernel_kernel
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device." 
#endif

__kernel void silu_mul_kernel_kernel(__global half* restrict G, __global half* restrict H, __global half* restrict U);
__kernel void silu_mul_kernel_kernel(__global half* restrict G, __global half* restrict H, __global half* restrict U) {
  for (int i = 0; i < 8; ++i) {
    float g = (convert_float(G[((((convert_int(get_group_id(0))) * 2048) + ((convert_int(get_local_id(0))) * 8)) + i)]));
    H[((((convert_int(get_group_id(0))) * 2048) + ((convert_int(get_local_id(0))) * 8)) + i)] = (convert_half(((g / (1.000000e+00f + exp((g * -1.000000e+00f)))) * (convert_float(U[((((convert_int(get_group_id(0))) * 2048) + ((convert_int(get_local_id(0))) * 8)) + i)])))));
  }
}

