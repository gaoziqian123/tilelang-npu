# OpenCL backend: pure C++ operator lowering registrations.

file(GLOB TILE_LANG_OPENCL_SRCS
  src/opencl/op/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_OPENCL_SRCS})
