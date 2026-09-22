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


def _patch_opencl_half_staging(source: str) -> str:
    """Promote small half staging arrays to vector SSA variables.

    The T.vectorized-into-local pattern emits

        half q4_1[4];
        (*(half4*)(q4_1 + 0)) = vload4(0, (half*)Qs + (addr));
        ... convert_float(q4_1[0]) ...

    which relies on the device compiler to SROA the array.  Adreno's
    compiler does not always do so (and does not CSE the repeated vector
    loads into it), leaving a private-memory round trip in hot loops.
    Rewrite to `half4 q4_1_v = vload4(...)` plus .sN component reads, the
    form the handwritten production kernels use.  Only arrays whose every
    use is the single full-vector store or a constant-index scalar/deref
    read are rewritten; anything else is left untouched.
    """

    original = source
    decl_re = re.compile(r"(?m)^(\s*)half\s+([A-Za-z_]\w*)\[([48])\];\s*$")
    candidates = []  # (name, n, rhs, store_line)

    for m in decl_re.finditer(source):
        name, n_s = m.group(2), m.group(3)
        n = int(n_s)
        vec = f"half{n}"
        store_re = re.compile(
            r"\(\*\(" + vec + r"\*\)\(" + re.escape(name) + r" \+ 0\)\) = ([^;]+);"
        )
        sm = store_re.search(source)
        if not sm:
            continue
        # The only `name +` occurrences must be the store itself plus
        # (for half8) half4-deref reads at +0/+4 (rewritten to .lo/.hi below).
        scrubbed = re.sub(
            r"\(\*\(half" + str(n) + r"\*\)\(" + re.escape(name) + r" \+ 0\)\)(?!\s*=)",
            "",
            source,
        )
        scrubbed = re.sub(
            r"\(\*\(half4\*\)\(" + re.escape(name) + r" \+ [0-9]+\)\)",
            "",
            scrubbed,
        )
        if len(re.findall(re.escape(name) + r" \+", scrubbed)) != 1:
            continue
        # No other decl of the same name, no address-of, no vload/vstore
        # naming the array directly.
        if len(decl_re.findall(source)) != len(
            [d for d in decl_re.findall(source) if d[1] != name]
        ) + 1:
            continue
        if re.search(r"&" + re.escape(name) + r"\b", source):
            continue
        if re.search(r"v(?:load|store)\d*\([^;]*\b" + re.escape(name) + r"\b", source):
            continue
        candidates.append((name, n, sm.group(1), sm.group(0)))

    for name, n, rhs, store_line in candidates:
        vec = f"half{n}"
        # drop the store line (it becomes the initializer)
        source = source.replace(store_line + "\n", "", 1)
        # replace the declaration with the vector variable definition
        source = re.sub(
            r"(?m)^(\s*)half\s+" + re.escape(name) + r"\[" + str(n) + r"\];\s*$",
            lambda m: f"{m.group(1)}{vec} {name}_v = {rhs};",
            source,
            count=1,
        )
        # scalar reads -> component reads
        source = re.sub(
            r"\b" + re.escape(name) + r"\[(\d)\]",
            lambda m, _n=name: f"{_n}_v.s{m.group(1)}",
            source,
        )
        # whole-vector deref reads -> the variable itself
        source = source.replace(
            f"(*(half{n}*)({name} + 0))", f"{name}_v"
        )
        # half4-deref reads of a half8 staging array -> .lo/.hi
        if n == 8:
            source = source.replace(
                f"(*(half4*)({name} + 0))", f"{name}_v.lo"
            ).replace(f"(*(half4*)({name} + 4))", f"{name}_v.hi")
        # revert if any array-form use survives
        if re.search(r"\b" + re.escape(name) + r"\[", source) or re.search(
            r"\b" + re.escape(name) + r" \+", source
        ):
            return original
    return source


def _patch_opencl_float8_split(source: str) -> str:
    """Split float8 SSA accumulator vars into float4 lo/hi pairs.

    The vectorizer emits 4-wide fp32 ops on the halves of promoted float8
    accumulators, which the C printer renders as component reconstructions:

        acc_1_0.lo = ((float4)(acc_1_0.s0, acc_1_0.s1, acc_1_0.s2, acc_1_0.s3) + fma...);

    Adreno compiles the clean form `acc_1_0_lo = (acc_1_0_lo + fma...)` much
    better (measured on the FA kernel: PV phase 54.5ms -> 28ms).  Only vars
    whose every use is a .lo/.hi/.sN component access are split; whole-var
    uses (mad, vload8/vstore8, constructors as values) disqualify the var.
    """

    decl_re = re.compile(r"(?m)^(\s*)float8\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")
    out = source
    for m in list(decl_re.finditer(source)):
        name, init = m.group(2), m.group(3)
        # every use must be a component access
        uses = re.findall(r"\b" + re.escape(name) + r"\b(?!\s*\.(?:lo|hi|s[0-7]))", out)
        # the declaration itself matches once; allow only that
        if len(uses) > 1:
            continue
        cm = re.match(r"\(float8\)\((.*)\)\s*$", init.strip())
        if not cm:
            continue
        args = [a.strip() for a in cm.group(1).split(",")]
        if len(args) == 1:
            lo_init = hi_init = f"(float4)({args[0]})"
        elif len(args) == 8:
            lo_init = "(float4)(" + ", ".join(args[:4]) + ")"
            hi_init = "(float4)(" + ", ".join(args[4:]) + ")"
        else:
            continue
        indent = m.group(1)
        new_decl = (
            f"{indent}float4 {name}_lo = {lo_init};\n"
            f"{indent}float4 {name}_hi = {hi_init};"
        )
        out = out.replace(m.group(0), new_decl, 1)
        # component accesses (longest first: .lo/.hi before .sN)
        out = re.sub(r"\b" + re.escape(name) + r"\.lo\b", f"{name}_lo", out)
        out = re.sub(r"\b" + re.escape(name) + r"\.hi\b", f"{name}_hi", out)

        def lane(mm, _n=name):
            i = int(mm.group(1))
            if i < 4:
                return f"{_n}_lo.s{i}"
            return f"{_n}_hi.s{i - 4}"

        out = re.sub(r"\b" + re.escape(name) + r"\.s([0-7])\b", lane, out)
        # revert if any whole-var use survives
        if re.search(r"\b" + re.escape(name) + r"\b", out):
            return source
    return out


def _patch_opencl_splat_convert(source: str) -> str:
    """Fold broadcast-cast chains into scalar-convert + splat.

    Two forms are handled:

    1. ``half4 X = ((half4)(EXPR));`` where EXPR is scalar and X is only
       consumed by ``convert_float4(X)``  ->  ``float4 X = (float4)
       (convert_float(EXPR));`` with uses rewritten to plain ``X``.
    2. inline ``convert_float4(((half4)(EXPR)))``  ->
       ``(float4)(convert_float(EXPR))``.

    The vectorizer's broadcast cast otherwise prints as a vector constructor
    plus a vector convert (4 convert instructions per splat).
    """

    out = source
    decl_re = re.compile(r"(?m)^(\s*)half4\s+([A-Za-z_]\w*)\s*=\s*\(\(half4\)\((.+)\)\);$")
    for m in list(decl_re.finditer(out)):
        name, expr = m.group(2), m.group(3)
        if "," in expr:
            continue  # genuine vector value, not a scalar splat
        uses = re.findall(r"\b" + re.escape(name) + r"\b", out)
        conv_uses = re.findall(r"convert_float4\(\s*" + re.escape(name) + r"\s*\)", out)
        if len(uses) - 1 != len(conv_uses) or not conv_uses:
            continue  # declaration itself counts as one use
        indent = m.group(1)
        out = out.replace(m.group(0), f"{indent}float4 {name} = (float4)(convert_float({expr}));", 1)
        out = re.sub(r"convert_float4\(\s*" + re.escape(name) + r"\s*\)", name, out)
    out = re.sub(
        r"convert_float4\(\(\(half4\)\(([^()]+)\)\)\)",
        r"(float4)(convert_float(\1))",
        out,
    )
    return out


def _patch_opencl_identity_splat(source: str) -> str:
    """((float4)(X.s0, X.s1, X.s2, X.s3))  ->  X.

    The C printer reconstructs a float4 from the lanes of a var; when the
    lanes are exactly that same var's lanes the constructor is the identity
    and can be dropped wherever it appears (assignment target position or
    addend position).
    """

    return re.sub(
        r"\(float4\)\((\w+)\.s0, \1\.s1, \1\.s2, \1\.s3\)",
        r"\1",
        source,
    )


def _patch_opencl_inline_splat(source: str) -> str:
    """((float4)(convert_float(E), convert_float(E), ...x4))  ->  (float4)(convert_float(E)).

    The vectorizer's broadcast cast prints as a 4-lane constructor with the
    scalar convert repeated per lane; one convert plus a splat is cheaper.
    """

    return re.sub(
        r"\(\(float4\)\(\(convert_float\((.*?)\)\)(?:, \(convert_float\(\1\)\)){3}\)\)",
        r"(float4)(convert_float(\1))",
        source,
    )


def build_opencl(mod, target):
    built = _raw_build_opencl(mod, target)
    source = built.inspect_source()
    patched = _patch_opencl_private_accumulators(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_half_staging(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_float8_split(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_splat_convert(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_identity_splat(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_inline_splat(source)
    if patched != source:
        source = patched
    if source != built.inspect_source():
        return _SourceModule(built, source)
    return built
