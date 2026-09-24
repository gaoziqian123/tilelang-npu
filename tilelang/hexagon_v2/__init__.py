"""Hexagon v2 backend prototype.

This package is deliberately separate from :mod:`tilelang.hexagon` (the
production statement emitter).  Phase 0 exposes a small backend skeleton and a
source-producing GEMM path used by ``examples/hexagon/gemm/gemm_nt_v2.py``.
"""

from . import target as target  # noqa: F401
from . import language as language  # noqa: F401
from . import intrinsics as intrinsics  # noqa: F401
from . import op as op  # noqa: F401
from . import codegen as codegen  # noqa: F401
from . import execution_backend as execution_backend  # noqa: F401
from . import pipeline as pipeline  # noqa: F401
from . import backend as backend  # noqa: F401
from .codegen import build_hexagon_v2_without_compile, emit_gemm_nt_c  # noqa: F401

__all__ = [
    "build_hexagon_v2_without_compile",
    "emit_gemm_nt_c",
]
