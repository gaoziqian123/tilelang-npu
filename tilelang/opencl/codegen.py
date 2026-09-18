from __future__ import annotations

from tilelang.backend.device_codegen import global_func_device_codegen

build_opencl = global_func_device_codegen("target.build.opencl")
