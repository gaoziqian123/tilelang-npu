"""OpenCL backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.execution_backend import ExecutionBackendSpec
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, pipeline

BACKEND = register_backend(
    BackendModule(
        name="opencl",
        target_kinds=("opencl",),
        pipelines={"opencl": PassPipeline("opencl", pipeline.OpenCLPassPipelineBody)},
        device_codegens={
            "opencl": DeviceCodegen(
                "opencl",
                build=codegen.build_opencl,
                build_without_compile=codegen.build_opencl,
            )
        },
        execution_backends=(ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),),
        host_codegens=STANDARD_HOST_CODEGENS,
    )
)
