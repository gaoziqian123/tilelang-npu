from __future__ import annotations

import re

from tilelang.backend.device_codegen import global_func_device_codegen

_raw_build_opencl = global_func_device_codegen("target.build.opencl")


class _SourceModule:
    def __init__(self, mod, source: str):
        self._mod = mod
        self._source = source

    def inspect_source(self, *args, **kwargs):
        return self._source

    def __getattr__(self, name):
        return getattr(self._mod, name)


def _patch_opencl_private_accumulators(source: str) -> str:
    """Fallback SROA for legacy texture-GEMM accumulator arrays.

    The formal OpenCL fragment pass below covers the direct-global and
    shared-staged GEMM_NT fragment kernels.  Texture GEMM still reaches legacy
    OpenCL codegen with a fully-unrolled private ``float acc[64]`` whose indices
    are constant but whose texture loads are not part of the fragment matcher.
    Keep this narrow fallback until texture-scope fragment TIR is normalized to
    the same shape as buffer GEMM; it is deliberately limited to acc* arrays and
    reverts if any non-constant access survives.
    """

    original_source = source
    patched_accumulators: set[str] = set()

    def repl_decl(match: re.Match[str]) -> str:
        indent = match.group(1)
        name = match.group(2)
        n = int(match.group(3))
        if n <= 1 or n % 8 != 0 or n > 128:
            return match.group(0)
        patched_accumulators.add(name)
        return "\n".join(f"{indent}float8 {name}_{i} = (float8)(0.000000e+00f);" for i in range(n // 8))

    source = re.sub(r"(?m)^(\s*)float\s+(acc\w*)\[(\d+)\];\s*$", repl_decl, source)

    def repl_acc(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in patched_accumulators:
            return match.group(0)
        idx = int(match.group(2))
        return f"{name}_{idx // 8}.s{idx % 8}"

    source = re.sub(r"\b(acc\w*)\[(\d+)\]", repl_acc, source)

    # The TIR vector fragment pass emits vector-typed private buffers
    # (float8 acc[8]) and unrolls every use to a constant index.  Legacy OpenCL
    # C prints those as arrays; Adreno is much more reliable when they are
    # scalar private variables, so scalar-replace this already-vector TIR form
    # without relying on fragment metadata or whole-kernel custom rendering.
    vector_accumulators: set[str] = set()

    def repl_vec_decl(match: re.Match[str]) -> str:
        indent, name, n_s = match.group(1), match.group(2), match.group(3)
        n = int(n_s)
        if n <= 1 or n > 32:
            return match.group(0)
        vector_accumulators.add(name)
        return "\n".join(f"{indent}float8 {name}_{i};" for i in range(n))

    source = re.sub(r"(?m)^(\s*)float8\s+(acc\w*)\[(\d+)\];\s*$", repl_vec_decl, source)

    def repl_vec_acc(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in vector_accumulators:
            return match.group(0)
        return f"{name}_{int(match.group(2))}"

    source = re.sub(r"\b(acc\w*)\[(\d+)\]", repl_vec_acc, source)

    for name in vector_accumulators:
        if re.search(rf"\b{re.escape(name)}\[", source):
            return original_source

    for name in patched_accumulators:
        if re.search(rf"\b{re.escape(name)}\[", source):
            return original_source

    source = re.sub(
        r"\(\(half\s*\*\)&([A-Za-z_]\w*)\)\[(\d)\]",
        lambda m: f"{m.group(1)}.s{m.group(2)}",
        source,
    )
    return source

def build_opencl(mod, target):
    built = _raw_build_opencl(mod, target)
    source = built.inspect_source()
    patched = _patch_opencl_private_accumulators(source)
    if patched != source:
        return _SourceModule(built, patched)
    return built
