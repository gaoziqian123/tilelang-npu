"""Hexagon NPU backend Python 侧入口。"""

from __future__ import annotations

# target 必须最先导入: 这里注册 ``target="hexagon"`` 的 Python 侧归一化。
from . import target as target  # noqa: F401
from . import language as language  # noqa: F401
from . import op as op  # noqa: F401
from . import codegen as codegen  # noqa: F401
from . import execution_backend as execution_backend  # noqa: F401
from . import pipeline as pipeline  # noqa: F401
from . import backend as backend  # noqa: F401
