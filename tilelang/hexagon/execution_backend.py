"""Hexagon v1 执行后端声明。"""

from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec


EXECUTION_BACKENDS = (
    # AOT 是目标形态；当前仅用于解析/建 context，真正 codegen 会在 codegen.py
    # 中以 NotImplementedError 报出边界。
    ExecutionBackendSpec("aot", enable_host_codegen=True, enable_device_compile=True),
)
