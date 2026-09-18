// Function: rmsnorm_kernel_kernel
__kernel void rmsnorm_kernel_kernel(__global float* restrict A, __global float* restrict O, int ncols, float eps);
__kernel void rmsnorm_kernel_kernel(__global float* restrict A, __global float* restrict O, int ncols, float eps) {
  __local float smem[256];
  float smem_1[256];
  smem[(convert_int(get_local_id(0)))] = 0.000000e+00f;
  for (int i = 0; i < 10; ++i) {
    float x = A[((((convert_int(get_group_id(0))) * 2560) + (i * 256)) + (convert_int(get_local_id(0))))];
    smem[(convert_int(get_local_id(0)))] = (smem[(convert_int(get_local_id(0)))] + (x * x));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int step = 0; step < 8; ++step) {
    if ((convert_int(get_local_id(0))) < (128 >> step)) {
      smem[(convert_int(get_local_id(0)))] = (smem[(convert_int(get_local_id(0)))] + smem[((convert_int(get_local_id(0))) + (128 >> step))]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float scale = (1.000000e+00f / sqrt(((smem[0] / (convert_float(ncols))) + eps)));
  for (int i_1 = 0; i_1 < 10; ++i_1) {
    O[((((convert_int(get_group_id(0))) * 2560) + (i_1 * 256)) + (convert_int(get_local_id(0))))] = (A[((((convert_int(get_group_id(0))) * 2560) + (i_1 * 256)) + (convert_int(get_local_id(0))))] * scale);
  }
}

