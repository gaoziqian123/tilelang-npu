from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend
from tilelang.backend.pass_pipeline import PassPipeline

from . import codegen, execution_backend, pipeline
from .target import target_is_hexagon

HEXAGON_PIPELINE = PassPipeline("hexagon", pipeline.HexagonPassPipelineBody)

BACKEND = register_backend(
    BackendModule(
        name="hexagon",
        target_kinds=("hexagon",),
        supports_target=target_is_hexagon,
        pipelines={"hexagon": HEXAGON_PIPELINE},
        device_codegens={
            "hexagon": DeviceCodegen(
                "hexagon",
                build=codegen.build_hexagon,
                build_without_compile=codegen.build_hexagon_without_compile,
            )
        },
        execution_backends=execution_backend.EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={},
    )
)
