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

    # Pointer-form accesses.  The OpenCL C printer emits vector accesses to
    # private arrays as (*(float4*)(acc_x + K)) (or vload/vstore on generic
    # casts), not acc_x[K], so the [CONST] rewrites above leave them dangling
    # against the removed array declaration.  Rewrite those forms onto the
    # promoted float8 vars too: stores first (the store LHS contains the load
    # pattern textually), then loads, then scalar *(acc + K) derefs.
    promoted = patched_accumulators | vector_accumulators

    def _lane(name: str, k: int) -> str:
        return f"{name}_{k // 8}.s{k % 8}"

    def _ctor(vec_ty: str, name: str, base: int, lanes: int) -> str:
        return "(" + vec_ty + ")(" + ", ".join(_lane(name, base + i) for i in range(lanes)) + ")"

    def _whole_or_half(name: str, base: int, lanes: int, value: str) -> str | None:
        """Assign a whole float8 var or a .lo/.hi half when the store is
        aligned - one assignment, no per-lane value duplication."""
        if lanes == 8 and base % 8 == 0:
            return f"{name}_{base // 8} = ({value});"
        if lanes == 4 and base % 4 == 0:
            half = "lo" if (base % 8) == 0 else "hi"
            return f"{name}_{base // 8}.{half} = ({value});"
        return None

    def repl_ptr_store(match: re.Match[str]) -> str:
        indent, vec_ty, name, base_s, value = match.groups()
        if name not in promoted:
            return match.group(0)
        lanes = int(vec_ty[-1])
        base = int(base_s)
        whole = _whole_or_half(name, base, lanes, value)
        if whole is not None:
            return f"{indent}{whole}"
        return "".join(
            f"{indent}{_lane(name, base + i)} = (({value}).s{i});\n" for i in range(lanes)
        ).rstrip("\n")

    source = re.sub(
        r"(?m)^(\s*)\(\*\((float[48])\*\)\((acc\w*) \+ (\d+)\)\) = (.*);$",
        repl_ptr_store,
        source,
    )

    def repl_vstore(match: re.Match[str]) -> str:
        indent, lanes_s, value, name, base_s = match.groups()
        if name not in promoted:
            return match.group(0)
        lanes = int(lanes_s)
        base = int(base_s)
        whole = _whole_or_half(name, base, lanes, value)
        if whole is not None:
            return f"{indent}{whole}"
        return "".join(
            f"{indent}{_lane(name, base + i)} = (({value}).s{i});\n" for i in range(lanes)
        ).rstrip("\n")

    source = re.sub(
        r"(?m)^(\s*)vstore([48])\((.*), 0, (acc\w*) \+ (\d+)\);$",
        repl_vstore,
        source,
    )

    def repl_ptr_load(match: re.Match[str]) -> str:
        vec_ty, name, base_s = match.groups()
        if name not in promoted:
            return match.group(0)
        return _ctor(vec_ty, name, int(base_s), int(vec_ty[-1]))

    source = re.sub(r"\(\*\((float[48])\*\)\((acc\w*) \+ (\d+)\)\)", repl_ptr_load, source)

    def repl_vload(match: re.Match[str]) -> str:
        lanes, name, base_s = match.groups()
        if name not in promoted:
            return match.group(0)
        return _ctor(f"float{lanes}", name, int(base_s), int(lanes))

    source = re.sub(r"vload([48])\(0, (acc\w*) \+ (\d+)\)", repl_vload, source)

    def repl_scalar_deref(match: re.Match[str]) -> str:
        name, k_s = match.groups()
        if name not in promoted:
            return match.group(0)
        return _lane(name, int(k_s))

    source = re.sub(r"\*\((acc\w*) \+ (\d+)\)", repl_scalar_deref, source)

    for name in vector_accumulators:
        if re.search(rf"\b{re.escape(name)}\[", source) or re.search(rf"\b{re.escape(name)} \+", source):
            return original_source

    for name in patched_accumulators:
        if re.search(rf"\b{re.escape(name)}\[", source) or re.search(rf"\b{re.escape(name)} \+", source):
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
