"""Hexagon-v2 AOT execution backend declaration."""

from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec

from .target import target_is_hexagon_v2


EXECUTION_BACKENDS = (
    ExecutionBackendSpec(
        "aot",
        supports_target=target_is_hexagon_v2,
        enable_host_codegen=True,
        enable_device_compile=True,
    ),
)
