from tilelang.backend.device_codegen import global_func_device_codegen

build_hexagon_without_compile = global_func_device_codegen("target.build.tilelang_hexagon_without_compile")
# FastRPC compilation and deployment are external, not host-side JIT execution.
build_hexagon = build_hexagon_without_compile
