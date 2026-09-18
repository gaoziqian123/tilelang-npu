"""Hexagon device codegen entry.

The backend is Python/AOT-only in v1: produce Hexagon intrinsic C text and wrap
it in a TVM CSourceModule so existing ``engine.lower`` users can inspect the
generated source.  It intentionally does not invoke TVM target codegen.
"""

from __future__ import annotations

import os
from pathlib import Path

from tvm import IRModule
import tvm
import tvm_ffi
from tvm.target import Target

from .emitter import emit_hexagon_c


def _source_module(source: str):
    create = tvm.ffi.get_global_func("runtime.CSourceModuleCreate")
    return create(source, "c", ["attnops_gemm_nt"], [])


def build_hexagon_without_compile(mod: IRModule, target: Target):
    source = emit_hexagon_c(mod, target)
    if out := os.environ.get("TILELANG_HEXAGON_EMIT_C"):
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return _source_module(source)


def build_hexagon(mod: IRModule, target: Target):
    # Compilation into a FastRPC skel is an external AOT step; keep both paths
    # source-producing so callers can run hexagon-clang -fsyntax-only directly.
    return build_hexagon_without_compile(mod, target)


tvm_ffi.register_global_func("target.build.tilelang_hexagon", f=build_hexagon, override=True)
tvm_ffi.register_global_func(
    "target.build.tilelang_hexagon_without_compile",
    f=build_hexagon_without_compile,
    override=True,
)
