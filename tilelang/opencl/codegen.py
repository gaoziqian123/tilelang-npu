from __future__ import annotations

import re

from tilelang.backend.device_codegen import global_func_device_codegen

_raw_build_opencl = global_func_device_codegen("target.build.opencl")


class _SourcePatchedModule:
    def __init__(self, mod, source: str):
        self._mod = mod
        self._source = source

    def inspect_source(self, *args, **kwargs):
        return self._source

    def __getattr__(self, name):
        return getattr(self._mod, name)


def _patch_opencl_private_accumulators(source: str) -> str:
    """Keep small private OpenCL accumulator arrays in vector registers.

    TVM's legacy OpenCL printer emits TileLang ``local.fragment`` buffers as C
    private arrays (e.g. ``float acc[64]``).  Adreno's OpenCL compiler does not
    reliably scalar-replace such arrays in the texture GEMM inner loop, so the
    generated kernel spills the accumulator tile.  This OpenCL-only source fixup
    rewrites the recognized fully-unrolled accumulator array into ``float8``
    vector variables while leaving the TIR and non-OpenCL backends untouched.
    """

    if "READ_IMAGEH" not in source and "read_imageh" not in source:
        return source

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

    # Only patch arrays whose base name starts with "acc".  Other private arrays
    # such as avec[32] may still use variable indices (kk) and are not safe to
    # vector-registerize by textual replacement.
    source = re.sub(r"(?m)^(\s*)float\s+(acc\w*)\[(\d+)\];\s*$", repl_decl, source)

    def repl_acc(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in patched_accumulators:
            return match.group(0)
        idx = int(match.group(2))
        return f"{name}_{idx // 8}.s{idx % 8}"

    source = re.sub(r"\b(acc\w*)\[(\d+)\]", repl_acc, source)

    # Safety: if any access to a patched accumulator survived with a
    # non-constant index (e.g. acc[kk]), the array declaration is gone but the
    # reference remains, producing uncompilable source.  Revert everything in
    # that case.
    for name in patched_accumulators:
        if re.search(rf"\b{re.escape(name)}\[", source):
            return original_source

    # Legacy OpenCL vector element extraction is printed as address-taking of a
    # half4 temporary, e.g. ((half*)&v_)[2].  That address escape is another
    # Adreno SROA blocker.  Use OpenCL swizzles instead.
    source = re.sub(r"\(\(half\s*\*\)&([A-Za-z_]\w*)\)\[(\d)\]",
                    lambda m: f"{m.group(1)}.s{m.group(2)}", source)
    return source


def build_opencl(mod, target):
    built = _raw_build_opencl(mod, target)
    source = built.inspect_source()
    patched = _patch_opencl_private_accumulators(source)
    if patched == source:
        return built
    return _SourcePatchedModule(built, patched)
