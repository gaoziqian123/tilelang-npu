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


_LOWERED_MOD: IRModule | None = None


def remember_lowered_mod(mod: IRModule) -> IRModule:
    """Keep the Hexagon pipeline output for the source emitter.

    The shared engine runs generic device-codegen preparation passes after
    host/device splitting.  Hexagon's Python emitter is intentionally placed in
    codegen, but it still consumes the Hexagon pipeline IR before those generic
    source-level cleanup passes rewrite expressions.  The codegen entry below
    selects the matching functions from this cached module.
    """

    global _LOWERED_MOD
    _LOWERED_MOD = mod
    return mod


def _emit_mod_for_codegen(mod: IRModule) -> IRModule:
    cached = _LOWERED_MOD
    if cached is None or not mod.functions:
        return mod

    funcs = {}
    for gv in mod.functions:
        if gv not in cached.functions:
            return mod
        funcs[gv] = cached.functions[gv]
    selected = IRModule(funcs)
    if cached.attrs:
        selected = selected.with_attrs(cached.attrs)
    return selected


def _source_module(source: str):
    create = tvm.ffi.get_global_func("runtime.CSourceModuleCreate")
    return create(source, "c", ["attnops_gemm_nt"], [])


def build_hexagon_without_compile(mod: IRModule, target: Target):
    source = emit_hexagon_c(_emit_mod_for_codegen(mod), target)
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
