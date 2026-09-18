// Function: silu_kernel_kernel
__kernel void silu_kernel_kernel(__global float* restrict A, __global float* restrict O);
__kernel void silu_kernel_kernel(__global float* restrict A, __global float* restrict O) {
  float x = A[(((convert_int(get_group_id(0))) * 256) + (convert_int(get_local_id(0))))];
  O[(((convert_int(get_group_id(0))) * 256) + (convert_int(get_local_id(0))))] = (x / (1.000000e+00f + exp((x * -1.000000e+00f))));
}

