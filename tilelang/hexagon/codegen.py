"""Hexagon device codegen entry.

The backend is Python/AOT-only in v1: produce Hexagon intrinsic C text and wrap
it in a TVM CSourceModule so existing ``engine.lower`` users can inspect the
generated source.  It intentionally does not invoke TVM target codegen.
"""

from __future__ import annotations

import os

from tvm import IRModule
import tvm
from tvm.target import Target

from .emitter import emit_hexagon_c
from .pipeline import default_emit_path


def _source_module(source: str):
    create = tvm.ffi.get_global_func("runtime.CSourceModuleCreate")
    return create(source, "c", ["attnops_gemm_nt"], [])


def build_hexagon_without_compile(mod: IRModule, target: Target):
    out = os.environ.get("TILELANG_HEXAGON_EMIT_C", default_emit_path())
    if len(mod.functions) == 0 and out and os.path.exists(out):
        source = open(out, encoding="utf-8").read()
    else:
        source = emit_hexagon_c(mod, target, out)
    return _source_module(source)


def build_hexagon(mod: IRModule, target: Target):
    # Compilation into a FastRPC skel is an external AOT step; keep both paths
    # source-producing so callers can run hexagon-clang -fsyntax-only directly.
    return build_hexagon_without_compile(mod, target)
