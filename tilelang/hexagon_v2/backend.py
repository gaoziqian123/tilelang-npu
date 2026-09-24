"""Hexagon-v2 backend manifest."""

from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend
from tilelang.backend.pass_pipeline import PassPipeline

from . import codegen, execution_backend, pipeline
from .target import target_is_hexagon_v2


BACKEND = register_backend(
    BackendModule(
        name="hexagon_v2",
        target_kinds=("hexagon",),
        supports_target=target_is_hexagon_v2,
        pipelines={"hexagon": PassPipeline("hexagon", pipeline.HexagonV2PassPipelineBody)},
        device_codegens={
            "hexagon": DeviceCodegen(
                "hexagon_v2",
                build=codegen.build_hexagon_v2_without_compile,
                build_without_compile=codegen.build_hexagon_v2_without_compile,
            )
        },
        execution_backends=execution_backend.EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
    )
)
