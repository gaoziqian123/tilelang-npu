from __future__ import annotations

import os
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
    decl_re = re.compile(r"(?m)^(\s*)half\s+([A-Za-z_]\w*)\[(\d+)\];\s*$")
    candidates = []  # (name, n, [(off, rhs, store_line), ...])

    for m in decl_re.finditer(source):
        name, n = m.group(2), int(m.group(3))
        if n not in (4, 8, 16):
            continue
        store_width = n
        if n == 16:
            # two accepted store forms: 2x half8, or 4x half4
            stores = []
            for off in (0, 8):
                sm = re.search(
                    r"\(\*\(half8\*\)\(" + re.escape(name) + r" \+ "
                    + str(off) + r"\)\) = ([^;]+);",
                    source,
                )
                if not sm:
                    break
                stores.append((off, sm.group(1), sm.group(0)))
            if len(stores) == 2:
                store_width = 8
            else:
                stores = []
                for off in (0, 4, 8, 12):
                    sm = re.search(
                        r"\(\*\(half4\*\)\(" + re.escape(name) + r" \+ "
                        + str(off) + r"\)\) = ([^;]+);",
                        source,
                    )
                    if not sm:
                        break
                    stores.append((off, sm.group(1), sm.group(0)))
                if len(stores) != 4:
                    continue
                store_width = 4
        else:
            vec = f"half{n}"
            stores = []
            sm = re.search(
                r"\(\*\(" + vec + r"\*\)\(" + re.escape(name) + r" \+ 0\)\) = ([^;]+);",
                source,
            )
            if sm:
                stores.append((0, sm.group(1), sm.group(0)))
            if len(stores) != 1:
                continue
        # The only `name +` occurrences must be the stores themselves plus
        # (for half8/16) half4-deref reads (rewritten to .lo/.hi below).
        scrubbed = re.sub(
            r"\(\*\(half\d+\*\)\(" + re.escape(name) + r" \+ \d+\)\)(?!\s*=)",
            "",
            source,
        )
        if len(re.findall(re.escape(name) + r" \+", scrubbed)) != len(stores):
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
        candidates.append((name, n, stores, store_width))

    for name, n, stores, store_width in candidates:
        # drop the store lines (they become the initializers)
        for _, _, store_line in stores:
            source = source.replace(store_line + "\n", "", 1)
        if n == 16:
            rhs_by_off = {off: rhs for off, rhs, _ in stores}
            if store_width == 8:
                decl_new = (
                    f"half8 {name}_lo = {rhs_by_off[0]};\n"
                    f"half8 {name}_hi = {rhs_by_off[8]};"
                )
            else:
                # 4x vload4 stores: if each pair covers contiguous halfs
                # (E(+4) == E(+0) + " + 4"), fuse into one vload8 per pair
                # (the fa.cl form; 2x fewer load instructions).
                def vload4_arg(rhs):
                    mm = re.match(r"vload4\(0, (.*)\)$", rhs)
                    return mm.group(1) if mm else None

                e0, e4 = vload4_arg(rhs_by_off[0]), vload4_arg(rhs_by_off[4])
                e8, e12 = vload4_arg(rhs_by_off[8]), vload4_arg(rhs_by_off[12])

                def contiguous(a, b):
                    # b must be exactly a + 4 halfs; the codegen prints
                    # absolute trailing offsets (+4/+8/+12), so compare the
                    # prefix modulo parens and the trailing constant.
                    if not (a and b):
                        return False
                    fa = re.sub(r"[\s()]", "", a)
                    fb = re.sub(r"[\s()]", "", b)
                    ma = re.match(r"^(.*)\+(\d+)$", fa)
                    mb = re.match(r"^(.*)\+(\d+)$", fb)
                    if ma and mb:
                        return ma.group(1) == mb.group(1) and int(mb.group(2)) - int(ma.group(2)) == 4
                    return fb == fa + "+4"

                if contiguous(e0, e4) and contiguous(e8, e12):
                    decl_new = (
                        f"half8 {name}_lo = vload8(0, {e0});\n"
                        f"half8 {name}_hi = vload8(0, {e8});"
                    )
                else:
                    decl_new = (
                        f"half8 {name}_lo;\n"
                        f"{name}_lo.lo = {rhs_by_off[0]};\n"
                        f"{name}_lo.hi = {rhs_by_off[4]};\n"
                        f"half8 {name}_hi;\n"
                        f"{name}_hi.lo = {rhs_by_off[8]};\n"
                        f"{name}_hi.hi = {rhs_by_off[12]};"
                    )
        else:
            decl_new = f"half{n} {name}_v = {stores[0][1]};"
        source = re.sub(
            r"(?m)^(\s*)half\s+" + re.escape(name) + r"\[" + str(n) + r"\];\s*$",
            lambda m, _d=decl_new: f"{m.group(1)}{_d}",
            source,
            count=1,
        )
        if n == 16:
            # scalar reads -> component reads of the lo/hi half8 vars
            def lane16(mm, _n=name):
                i = int(mm.group(1))
                if i < 8:
                    return f"{_n}_lo.s{i}"
                return f"{_n}_hi.s{i - 8}"

            source = re.sub(
                r"\b" + re.escape(name) + r"\[(\d+)\]", lane16, source
            )
            source = source.replace(
                f"(*(half8*)({name} + 0))", f"{name}_lo"
            ).replace(f"(*(half8*)({name} + 8))", f"{name}_hi")
            source = source.replace(
                f"(*(half4*)({name} + 0))", f"{name}_lo.lo"
            ).replace(f"(*(half4*)({name} + 4))", f"{name}_lo.hi"
            ).replace(f"(*(half4*)({name} + 8))", f"{name}_hi.lo"
            ).replace(f"(*(half4*)({name} + 12))", f"{name}_hi.hi")
        else:
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


def _patch_opencl_splat_mul(source: str) -> str:
    """((float4)(convert_float(E))) * (convert_float4(V))  ->  (convert_float(E) * convert_float4(V)).

    Scalar-times-vector instead of splat-constructor-times-vector: drops one
    constructor per FMA group (measured on the FA PV loop, part of the
    33.6ms -> 27.6ms step).  Semantically identical.
    """

    return re.sub(
        r"\(\(float4\)\(convert_float\((.*?)\)\)\) \* \(convert_float4\((.*?)\)\)",
        r"(convert_float(\1) * convert_float4(\2))",
        source,
    )


def _patch_opencl_typed_shared(source: str) -> str:
    """Replace the merged uchar shared blob + void* aliases with typed __local arrays.

    TVM's MergeSharedMemoryAllocations packs every shared buffer into one
    `__local uchar buf_dyn_shmem[N]` blob and hands out `void*` aliases;
    every access then goes through a generic-address-space cast
    (`((half*)Sh)[...]`).  On Adreno these generic shared accesses are much
    slower than __local-qualified ones (measured on the FA kernel:
    71.0ms -> 38.5ms just by re-declaring the same buffers as typed
    `__local half Sh[...]` etc.).

    Guards (any failure -> source unchanged):
    - the blob declaration and the consecutive `void* X = blob + OFF;`
      aliases must match the exact printed form;
    - all alias offsets must be distinct (aliased offsets mean the merger
      reused storage; separate arrays would inflate shared usage);
    - each alias may only be used through `(<T>*)X` / `((<T>*)X)[...]`
      casts with a single element type (half/float), and never in pointer
      arithmetic (`X + ...`) or as a bare value;
    - the extent of each buffer is bounded by the next alias offset (or
      the blob size for the last one), so the total is unchanged.
    """

    m = re.search(r"(?m)^(\s*__local uchar buf_dyn_shmem\[(\d+)\];\n((?:\s*void\* \w+ = \(\(void\*\)\(\(char\*\)buf_dyn_shmem \+ \d+\)\);\n)+))", source)
    if not m:
        return source
    blob_size = int(m.group(2))
    alias_re = re.compile(r"void\* (\w+) = \(\(void\*\)\(\(char\*\)buf_dyn_shmem \+ (\d+)\)\);")
    aliases = [(name, int(off)) for name, off in alias_re.findall(m.group(3))]
    if not aliases or len({off for _, off in aliases}) != len(aliases):
        return source
    body = source[: m.start()] + source[m.end():]

    def cast_uses(name):
        # `((T*)X)` (double-paren, indexing form) and `(T*)X` (single)
        double = re.findall(r"\(\((\w+)\*\)" + re.escape(name) + r"\)", body)
        single = re.findall(r"(?<!\()\((\w+)\*\)" + re.escape(name) + r"\b", body)
        return double, single

    def scrub(name):
        out = re.sub(r"\(\(\w+\*\)" + re.escape(name) + r"\)", "", body)
        out = re.sub(r"(?<!\()\(\w+\*\)" + re.escape(name) + r"\b", "", out)
        return out

    decls = []
    offsets = sorted(off for _, off in aliases) + [blob_size]
    for name, off in aliases:
        double, single = cast_uses(name)
        types = {t for t in double + single if t in ("half", "float")}
        if len(types) != 1 or len(double) + len(single) == 0:
            return source
        if re.search(r"\b" + re.escape(name) + r"\b", scrub(name)):
            return source  # pointer arithmetic or bare use
        nxt = min(o for o in offsets if o > off)
        extent = nxt - off
        elem = types.pop()
        esize = 2 if elem == "half" else 4
        if extent % esize != 0:
            return source
        decls.append(f"__local {elem} {name}[{extent // esize}];")
    out = body
    for name, _ in aliases:
        out = re.sub(r"\(\(\w+\*\)" + re.escape(name) + r"\)", name, out)
        out = re.sub(r"(?<!\()\(\w+\*\)" + re.escape(name) + r"\b", name, out)
    decl_block = "\n".join(decls) + "\n"
    out = out[: m.start()] + decl_block + out[m.start():]
    return out


def _patch_opencl_gdn_dot8(source: str) -> str:
    """Use half8 vector dot accumulators for the GDN prep Q/K dot block.

    Generic lowering expands the 32x32 prep dot phase into eight scalar fp32
    128-wide dot loops per work-item.  The production OpenCL GDN kernel uses a
    half8 accumulator for both K.K and Q.K and only horizontally reduces the
    final eight lanes; on Adreno this is much faster and still matches the
    intended fp16-dot numerical envelope.  Keep the peephole narrowly guarded to
    the generated GDN prep kernel shape.
    """

    if "gdn_prep_kernel_kernel" not in source or "float kk_1[1];" not in source:
        return source
    start = source.find("  float kk_1[1];\n")
    end = source.find("  barrier(CLK_LOCAL_MEM_FENCE);\n", start)
    if start < 0 or end < 0:
        return source

    repl = r'''  for (int t = tl_lid0; t < 1024; t += 128) {
    int i = t >> 5;
    int j = t & 31;
    float lv = 0.000000e+00f;
    float av = 0.000000e+00f;
    if (j <= i) {
      float decay = exp(gc[i] - gc[j]);
      half8 acc8 = (half8)(0.0h);
      half8 qacc = (half8)(0.0h);
      const int base_k = (((tl_gid1 & 15) * 131072) + (tl_gid0 * 4096));
      #pragma unroll
      for (int kd = 0; kd < 128; kd += 8) {
        half8 kj = vload8(0, K + base_k + (j * 128) + kd);
        acc8 += vload8(0, K + base_k + (i * 128) + kd) * kj;
        qacc += vload8(0, Q + base_k + (i * 128) + kd) * kj;
      }
      float kk = convert_float(acc8.s0) + convert_float(acc8.s1) + convert_float(acc8.s2) + convert_float(acc8.s3)
               + convert_float(acc8.s4) + convert_float(acc8.s5) + convert_float(acc8.s6) + convert_float(acc8.s7);
      if (j < i) {
        lv = bf[i] * kk * decay;
      }
      av = (convert_float(qacc.s0) + convert_float(qacc.s1) + convert_float(qacc.s2) + convert_float(qacc.s3)
          + convert_float(qacc.s4) + convert_float(qacc.s5) + convert_float(qacc.s6) + convert_float(qacc.s7)) * decay;
    }
    L[t] = lv;
    A2buf[(((tl_gid1 * 32768) + (tl_gid0 * 1024)) + t)] = convert_half(av);
  }
'''
    out = source[:start] + repl + source[end:]
    if "float kk_1[1];" in out or "qk_1[1]" in out:
        return source
    return out


def _patch_opencl_gdn_seq_state_stream(source: str) -> str:
    """Stream the GDN seq state update through a half8 accumulator.

    This folds the generic pass2 pattern from private ``float m8_1[8]`` plus two
    float4 reductions into the handwritten shape: one half8 accumulator, one
    float8 conversion, a chunk-level Egl load hoisted outside the lane stores,
    and an explicit unroll hint.  Egl indexing is expressed from ``tl_gid1``
    (value head) instead of relying on fragile ``tl_wi_affN`` numbering.
    """

    if "gdn_seq_kernel_kernel" not in source or "float m8_1[8];" not in source:
        return source
    pass2 = r'''    float egl = EglBuf[((tl_gid1 * 32) + c)];
    for (int sb = 0; sb < 4; ++sb) {
      int dk0 = (sb * 32) + ((tl_lid0 >> 5) * 8);
      half8 m8 = (half8)(0.0h);
      #pragma unroll
      for (int tt = 0; tt < 32; ++tt) {
        m8 += vload8(0, kdl + (tt * 128) + dk0) * (half8)vn[(tt * 32) + (tl_lid0 & 31)];
      }
      float8 mf = convert_float8(m8);
      int sbase = (((tl_gid1 * 16384) + (sb * 4096)) + ((tl_lid0 >> 5) * 1024) + (tl_gid0 * 32) + (tl_lid0 & 31));
      S[sbase] = (egl * S[sbase]) + mf.s0;
      S[(sbase + 128)] = (egl * S[(sbase + 128)]) + mf.s1;
      S[(sbase + 256)] = (egl * S[(sbase + 256)]) + mf.s2;
      S[(sbase + 384)] = (egl * S[(sbase + 384)]) + mf.s3;
      S[(sbase + 512)] = (egl * S[(sbase + 512)]) + mf.s4;
      S[(sbase + 640)] = (egl * S[(sbase + 640)]) + mf.s5;
      S[(sbase + 768)] = (egl * S[(sbase + 768)]) + mf.s6;
      S[(sbase + 896)] = (egl * S[(sbase + 896)]) + mf.s7;
    }
    barrier(CLK_LOCAL_MEM_FENCE);'''
    out = re.sub(
        r"    for \(int sb = 0; sb < 4; \+\+sb\) \{\n      float m8_1\[8\];.*?\n    barrier\(CLK_LOCAL_MEM_FENCE\);",
        pass2,
        source,
        count=1,
        flags=re.S,
    )
    if out == source or "float m8_1[8];" in out:
        return source
    return out


def _patch_opencl_gdn_seq_out_private_arrays(source: str) -> str:
    """Promote the GDN seq output float[8] accumulator to two float4 regs."""

    if "gdn_seq_kernel_kernel" not in source or "float out8_1[8];" not in source:
        return source
    out = source.replace("    float out8_1[8];", "    float4 out8_1_lo;\n    float4 out8_1_hi;", 1)
    out = out.replace("(*(float4*)(out8_1 + 0))", "out8_1_lo")
    out = out.replace("(*(float4*)(out8_1 + 4))", "out8_1_hi")
    if "out8_1[" in out or "out8_1 +" in out:
        return source
    return out


def _patch_opencl_gdn_seq_pass1_float8(source: str) -> str:
    """Rewrite GDN seq pass1 to the handwritten float8 stream shape.

    Generic lowering (after float8 splitting) prints the pass1 state stream as
    two float4 accumulators, two vload4s from S, and explicit scalar splats for
    w/q.  The handwritten GDN kernel uses one float8 load and scalar*vector
    FMAs per dk.  Keep this narrowly scoped to the current GDN seq lowering and
    express addresses from group/local ids rather than tl_wi_affN meanings.
    """

    if "gdn_seq_kernel_kernel" not in source or "float4 acc_1_0_lo" not in source:
        return source
    repl = r'''    int t = (tl_lid0 >> 2);
    int dvg = (tl_lid0 & 3);
    int dv0 = ((tl_gid0 * 32) + (dvg * 8));
    int cb = (((tl_gid1 * 32) + c) * 4096);
    int qbase = (((tl_gid1 & 15) * 131072) + (c * 4096) + (t * 128));
    float8 acc8 = (float8)(0.000000e+00f);
    float8 acco8 = (float8)(0.000000e+00f);
    for (int dk = 0; dk < 128; ++dk) {
      float8 sv = vload8(0, S + (((tl_gid1 * 16384) + (dk * 128)) + dv0));
      float wv = convert_float(Wbuf[(cb + (t * 128) + dk)]);
      float qv = convert_float(Q[(qbase + dk)]);
      acc8 = (acc8 + (wv * sv));
      acco8 = (acco8 + (qv * sv));
    }
    half8 u8 = vload8(0, Ubuf + (cb + (t * 128) + dv0));
    vstore8(convert_half8(convert_float8(u8) - acc8), 0, vn + (tl_lid0 * 8));'''
    out = re.sub(
        r"    float4 acc_1_0_lo = .*?\n    vstore4\(\(\*\(half4\*\)\(vn_local_cast_1 \+ 0\)\), 0, vn \+ \(tl_wi_aff10 \+ 4\)\);",
        repl,
        source,
        count=1,
        flags=re.S,
    )
    if out == source or "float4 acc_1_0_lo" in out or "vn_local_cast_1" in out:
        return source
    return out


def _patch_opencl_gdn_seq_output_vstore8(source: str) -> str:
    """Drop the half[8] staging array on the GDN seq output store."""

    if "gdn_seq_kernel_kernel" not in source or "half O_local_cast_3[8];" not in source:
        return source
    out = re.sub(
        r"    half O_local_cast_3\[8\];\n"
        r"    \(\*\(half4\*\)\(O_local_cast_3 \+ 0\)\) = \(convert_half4\((.*?)\)\);\n"
        r"    \(\*\(half4\*\)\(O_local_cast_3 \+ 4\)\) = \(convert_half4\((.*?)\)\);\n"
        r"    vstore8\(\(\*\(half8\*\)\(O_local_cast_3 \+ 0\)\), 0, O \+ ([^;]+)\);",
        r"    vstore8(convert_half8((float8)(\1, \2)), 0, O + \3);",
        source,
        count=1,
        flags=re.S,
    )
    if out == source or "O_local_cast_3" in out:
        return source
    return out


def _patch_opencl_gdn_native_exp(source: str) -> str:
    """Use OpenCL native_exp for GDN gate exponentials, matching gdn.cl."""

    if "gdn_prep_kernel_kernel" not in source and "gdn_seq_kernel_kernel" not in source:
        return source
    return source.replace("exp(", "native_exp(")


def _patch_opencl_gdn_scalar1_arrays(source: str) -> str:
    """Scalar-replace GDN lowering's one-element private float arrays."""

    if "gdn_prep_kernel_kernel" not in source and "gdn_seq_kernel_kernel" not in source:
        return source
    decl_re = re.compile(r"(?m)^(\s*)float\s+([A-Za-z_]\w*)\[1\];\s*\n\1\2\[0\]\s*=\s*([^;]+);$")
    out = source
    for m in list(decl_re.finditer(source)):
        name = m.group(2)
        # Only rewrite pure scalar private temporaries with constant index 0.
        rest = out[: m.start()] + out[m.end():]
        if re.search(r"\b" + re.escape(name) + r"\[(?!0\])", rest):
            continue
        old = m.group(0)
        new = f"{m.group(1)}float {name} = {m.group(3)};"
        out = out.replace(old, new, 1)
        out = re.sub(r"\b" + re.escape(name) + r"\[0\]", name, out)
    return out

def _patch_opencl_workitem_id_hoist(source: str) -> str:
    """Hoist repeated OpenCL work-item id builtins into kernel-local consts.

    TileLang's generated address expressions can repeat
    ``convert_int(get_group_id(N))`` / ``convert_int(get_local_id(N))`` in every
    vector load.  Adreno does not reliably CSE those builtin calls inside large
    unrolled GEMM loops, so keep this intentionally small textual peephole: one
    const int per used id at kernel entry, then replace exact occurrences.
    """

    gid_re = re.compile(r"\(convert_int\(get_(group|local)_id\(([0-2])\)\)\)")
    kernel_re = re.compile(r"(?m)^(__kernel\s+void\s+\w+\s*\([^\n]*\)\s*)\{")

    pieces: list[str] = []
    pos = 0
    while True:
        km = kernel_re.search(source, pos)
        if not km:
            pieces.append(source[pos:])
            break
        next_km = kernel_re.search(source, km.end())
        end = next_km.start() if next_km else len(source)
        chunk = source[km.start():end]
        used = sorted({(kind, axis) for kind, axis in gid_re.findall(chunk)})
        pieces.append(source[pos:km.start()])
        if not used:
            pieces.append(chunk)
        else:
            decls = " ".join(
                f"const int tl_{'gid' if kind == 'group' else 'lid'}{axis} = convert_int(get_{kind}_id({axis}));"
                for kind, axis in used
            )
            patched = chunk[: km.end() - km.start()] + " " + decls + chunk[km.end() - km.start():]
            for kind, axis in used:
                name = f"tl_{'gid' if kind == 'group' else 'lid'}{axis}"
                patched = patched.replace(f"(convert_int(get_{kind}_id({axis})))", name)
            pieces.append(patched)
        pos = end
    return "".join(pieces)


def _patch_opencl_workitem_affine_hoist(source: str) -> str:
    """Hoist repeated affine work-item address bases.

    The id-only hoist above removes repeated builtin calls, but GR GEMM still
    prints the same workgroup/lane affine bases in every unrolled vector load,
    e.g. ``((tl_gid1 * 81920) + ((tl_lid0 >> 4) * 20480))``.  Keep this pass
    textual and conservative: only exact integer expressions composed of the
    already-hoisted ``tl_gid*``/``tl_lid*`` variables are considered, and only
    if they occur at least twice within one kernel.
    """

    kernel_re = re.compile(r"(?m)^(__kernel\s+void\s+\w+\s*\([^\n]*\)\s*)\{")
    # Longest/most valuable forms first; shorter subterms are then counted on
    # the already-rewritten body.  The patterns intentionally match the legacy
    # OpenCL printer's spaced form instead of trying to parse C.
    patterns = [
        re.compile(r"\(\(tl_gid\d+ \* \d+\) \+ \(\(tl_lid\d+ >> \d+\) \* \d+\)\)"),
        re.compile(r"\(\(tl_wi_aff\d+ \+ \(tl_gid\d+ \* \d+\)\) \+ \(\(tl_lid\d+ & \d+\) \* \d+\)\)"),
        re.compile(r"\(\(tl_lid\d+ >> \d+\) \* \d+\)"),
        re.compile(r"\(\(tl_lid\d+ & \d+\) \* \d+\)"),
        re.compile(r"\(tl_gid\d+ \* \d+\)"),
        re.compile(r"\(tl_lid\d+ \* \d+\)"),
    ]

    pieces: list[str] = []
    pos = 0
    while True:
        km = kernel_re.search(source, pos)
        if not km:
            pieces.append(source[pos:])
            break
        next_km = kernel_re.search(source, km.end())
        end = next_km.start() if next_km else len(source)
        chunk = source[km.start():end]
        pieces.append(source[pos:km.start()])

        insert = km.end() - km.start()
        id_decls = re.match(
            r"(?:\s*const int tl_(?:g|l)id[0-2] = convert_int\(get_(?:group|local)_id\([0-2]\)\);)*",
            chunk[insert:],
        )
        if id_decls is not None:
            insert += id_decls.end()
        prefix, body = chunk[:insert], chunk[insert:]
        decls: list[str] = []
        cse_idx = 0
        for pat in patterns:
            exprs = sorted(set(pat.findall(body)), key=len, reverse=True)
            for expr in exprs:
                if body.count(expr) < 2:
                    continue
                name = f"tl_wi_aff{cse_idx}"
                cse_idx += 1
                decls.append(f" const int {name} = {expr};")
                body = body.replace(expr, name)
        if decls:
            pieces.append(prefix + "".join(decls) + body)
        else:
            pieces.append(chunk)
        pos = end
    return "".join(pieces)


def _patch_opencl_loop_affine_hoist(source: str) -> str:
    """Hoist repeated induction-variable address bases inside simple loops."""

    loop_re = re.compile(r"(?ms)(?P<head>^\s*for \(int (?P<iv>\w+) = [^\n]+\) \{\n)(?P<body>.*?)(?P<tail>^\s*\})")
    out: list[str] = []
    pos = 0
    for lm in loop_re.finditer(source):
        out.append(source[pos:lm.start()])
        iv = lm.group("iv")
        body = lm.group("body")
        patterns = [
            re.compile(r"\(tl_wi_aff\d+ \+ \(" + re.escape(iv) + r" \* \d+\)\)"),
            re.compile(r"\(\(\(" + re.escape(iv) + r" \* \d+\) \+ tl_wi_aff\d+\) \+ tl_wi_aff\d+\)"),
            re.compile(r"\(" + re.escape(iv) + r" \* \d+\)"),
        ]
        decls: list[str] = []
        cse_idx = 0
        for pat in patterns:
            exprs = sorted(set(pat.findall(body)), key=len, reverse=True)
            for expr in exprs:
                if body.count(expr) < 2:
                    continue
                name = f"tl_loop_aff{cse_idx}"
                cse_idx += 1
                decls.append(f" const int {name} = {expr};")
                body = body.replace(expr, name)
        if decls:
            head = lm.group("head")
            head = head[:-1] + "".join(decls) + "\n"
            out.append(head + body + lm.group("tail"))
        else:
            out.append(lm.group(0))
        pos = lm.end()
    out.append(source[pos:])
    return "".join(out)


def _patch_opencl_grgemm_base_pointers(source: str) -> str:
    """Rewrite GR GEMM address expressions through per-work-item base pointers."""

    if "tl_wi_aff0" not in source or "tl_wi_aff1" not in source or "A_frag" in source or "B_frag" in source:
        return source
    kernel_re = re.compile(r"(?m)^(__kernel\s+void\s+\w+\s*\([^\n]*\)\s*)\{")
    pieces: list[str] = []
    pos = 0
    while True:
        km = kernel_re.search(source, pos)
        if not km:
            pieces.append(source[pos:])
            break
        next_km = kernel_re.search(source, km.end())
        end = next_km.start() if next_km else len(source)
        chunk = source[km.start():end]
        pieces.append(source[pos:km.start()])

        insert = km.end() - km.start()
        decl_match = re.match(r"(?:\s*const int tl_[A-Za-z0-9_]+ = [^;]+;)*", chunk[insert:])
        if decl_match is not None:
            insert += decl_match.end()
        prefix, body = chunk[:insert], chunk[insert:]
        decls: list[str] = []
        if "A + (tl_wi_aff0 +" in body:
            decls.append(" __global half* tl_A_base = A + tl_wi_aff0;")
            body = body.replace("A + (tl_wi_aff0 +", "tl_A_base + (")
        if "A + tl_loop_aff0" in body:
            body = re.sub(
                r"(const int tl_loop_aff1 = [^;]+;)",
                r"\1 __global half* tl_Ap = A + tl_loop_aff0; __global half* tl_Bp = B + (tl_loop_aff1 - tl_wi_aff2);",
                body,
                count=1,
            )
            body = body.replace("A + tl_loop_aff0", "tl_Ap")
            body = body.replace("A + (tl_loop_aff0 +", "tl_Ap + (")
            body = body.replace("B + (tl_loop_aff1 - tl_wi_aff2)", "tl_Bp")
            body = re.sub(r"B \+ \(\(tl_loop_aff1 \+ (\d+)\) - tl_wi_aff2\)", r"tl_Bp + \1", body)
            body = body.replace("__global half* tl_Ap = tl_Ap;", "__global half* tl_Ap = A + tl_loop_aff0;")
            body = body.replace("__global half* tl_Bp = tl_Bp;", "__global half* tl_Bp = B + (tl_loop_aff1 - tl_wi_aff2);")
        b_base_expr = "((tl_wi_aff3 + tl_wi_aff4) - tl_wi_aff2)"
        if re.search(r"B \+ \(\(\(\(pos4 \* \d+\) \+ tl_wi_aff3\) \+ tl_wi_aff4\)", body):
            decls.append(f" __global half* tl_B_base = B + {b_base_expr};")
            body = re.sub(
                r"B \+ \(\(\(\((pos4 \* \d+)\) \+ tl_wi_aff3\) \+ tl_wi_aff4\)\) - tl_wi_aff2\)",
                r"tl_B_base + (\1)",
                body,
            )
            body = re.sub(
                r"B \+ \(\(\(\(\((pos4 \* \d+)\) \+ tl_wi_aff3\) \+ tl_wi_aff4\) \+ (\d+)\) - tl_wi_aff2\)",
                r"tl_B_base + ((\1) + \2)",
                body,
            )
        if "C + (tl_wi_aff1" in body:
            decls.append(" __global half* tl_C_base = C + tl_wi_aff1;")
            body = body.replace("C + (tl_wi_aff1", "tl_C_base + (")
        if decls:
            pieces.append(prefix + "".join(decls) + body)
        else:
            pieces.append(chunk)
        pos = end
    return "".join(pieces)


def _patch_opencl_grgemm_epilogue_direct_store(source: str) -> str:
    """Bypass GR-GEMM private C fragment arrays in the writeback epilogue.

    GR GEMM lowering already keeps one vector accumulator per output row in
    private SSA form, but the generic T.copy epilogue may spill those vectors
    lane-by-lane into a private ``C_frag_*[64]`` and immediately read them back
    for the final global ``vstore8``.  On Adreno that private-memory round trip
    is expensive.  Fold only the exact, fully-unrolled pattern

        C_frag[k*8+lane] = (acc).sL;
        ... optional half C_local_cast + convert_half4(float4 C_frag) ...
        vstore8(..., 0, C + row_addr);

    into ``vstore8(convert_half8(acc), 0, addr)`` for fp32 accumulators or
    ``vstore8(acc, 0, addr)`` for fp16 accumulators.

    Guards are intentionally strict: this pass is limited to the GR GEMM
    scaffold, requires all 64 fragment lanes to be written exactly once from
    ordered vector accumulator lanes, accepts only the two known readback
    shapes, and reverts unless the fragment/cast arrays have no other uses once
    the replacement is applied.  Dead declarations are removed later by
    _patch_opencl_unused_local_decls.
    """

    if "tl_wi_aff" not in source or "tl_loop_aff" not in source or "A_frag" in source or "B_frag" in source:
        return source

    frag_decls = re.findall(r"(?m)^\s*(float|half)\s+(C_frag_\w*)\[64\];\s*$", source)
    if len(frag_decls) != 1:
        return source
    frag_ty, frag = frag_decls[0]

    write_re = re.compile(
        r"(?m)^(?P<line>\s*" + re.escape(frag) +
        r"\[(?P<idx>\d+)\]\s*=\s*\((?P<acc>[^;]+)\)\.s(?P<lane>[0-7]);\s*)$"
    )
    writes = list(write_re.finditer(source))
    if len(writes) != 64:
        return source
    by_idx: dict[int, tuple[str, int, str]] = {}
    for m in writes:
        idx = int(m.group("idx"))
        lane = int(m.group("lane"))
        acc = m.group("acc").strip()
        if idx in by_idx:
            return source
        by_idx[idx] = (acc, lane, m.group("line"))
    if set(by_idx) != set(range(64)):
        return source

    acc_by_off: dict[int, str] = {}
    for off in range(0, 64, 8):
        acc0 = by_idx[off][0]
        for lane in range(8):
            acc, lane_seen, _ = by_idx[off + lane]
            if acc != acc0 or lane_seen != lane:
                return source
        acc_by_off[off] = acc0

    used_offsets: set[int] = set()
    replacements: list[tuple[str, str]] = []

    if frag_ty == "half":
        store_re = re.compile(
            r"(?m)^(?P<line>\s*vstore8\(\(\*\(half8\*\)\(" + re.escape(frag) +
            r" \+ (?P<off>\d+)\)\), 0, (?P<addr>[^;]+)\);\s*)$"
        )
        stores = list(store_re.finditer(source))
        if len(stores) != 8:
            return source
        for m in stores:
            off = int(m.group("off"))
            if off not in acc_by_off or off in used_offsets:
                return source
            used_offsets.add(off)
            replacements.append((m.group("line"), f"  vstore8({acc_by_off[off]}, 0, {m.group('addr')});"))
    else:
        # float fragment: each output row is converted through a short-lived
        # half[8] cast array written by two convert_half4 stores then vstore8'd.
        cast_re = re.compile(
            r"(?ms)^(?P<block>\s*half\s+(?P<cast>C_local_cast\w*)\[8\];\s*\n"
            r"\s*\(\*\(half4\*\)\((?P=cast) \+ 0\)\) = \(convert_half4\(\(\*\(float4\*\)\(" + re.escape(frag) + r" \+ (?P<off0>\d+)\)\)\)\);\s*\n"
            r"\s*\(\*\(half4\*\)\((?P=cast) \+ 4\)\) = \(convert_half4\(\(\*\(float4\*\)\(" + re.escape(frag) + r" \+ (?P<off4>\d+)\)\)\)\);\s*\n"
            r"\s*vstore8\(\(\*\(half8\*\)\((?P=cast) \+ 0\)\), 0, (?P<addr>[^;]+)\);\s*)$"
        )
        blocks = list(cast_re.finditer(source))
        if len(blocks) != 8:
            return source
        for m in blocks:
            off0 = int(m.group("off0"))
            off4 = int(m.group("off4"))
            cast = m.group("cast")
            if off4 != off0 + 4 or off0 not in acc_by_off or off0 in used_offsets:
                return source
            # The cast array must appear only inside this recognized block.
            if len(re.findall(r"\b" + re.escape(cast) + r"\b", source)) != 4:
                return source
            used_offsets.add(off0)
            replacements.append((m.group("block"), f"  vstore8(convert_half8({acc_by_off[off0]}), 0, {m.group('addr')});"))

    if used_offsets != set(range(0, 64, 8)):
        return source

    out = source
    for _, _, line in by_idx.values():
        out = out.replace(line + "\n", "", 1)
    for old, new in replacements:
        out = out.replace(old, new + "\n", 1)

    scrub = out
    scrub = re.sub(r"(?m)^\s*(?:float|half)\s+" + re.escape(frag) + r"\[64\];\s*$", "", scrub)
    if re.search(r"\b" + re.escape(frag) + r"\b", scrub):
        return source
    if frag_ty == "float" and re.search(r"\bC_local_cast\w*\b", scrub):
        return source
    return out


def _patch_opencl_grgemm_b_texture(source: str) -> str:
    """Route GR-GEMM's direct-global B vload8s through an RGBA half image.

    This is deliberately limited to the direct-global GR GEMM scaffold used by
    examples/opencl/gemm/gemm_nt.py with B in KN layout.  The shape we accept is
    the post-base-pointer form:

        __global half* tl_Bp = B + (tl_loop_aff1 - tl_wi_aff2);
        bv_1[i] = convert_float8(vload8(0, tl_Bp + i*N));  # i = 0..3

    inside ``for (int pos4 = ... )``.  For KN layout, the host exposes B as an
    image of width N/4 and height K, so each vload8 becomes two read_imageh()
    calls at x = n0/4 and y = pos4*4+i.  If any of the exact scaffold markers are
    absent, return the original source unchanged.
    """

    if os.environ.get("TL_PATCH_B_TEX") != "1":
        return source
    if "tl_wi_aff" not in source or "tl_loop_aff" not in source or "A_frag" in source or "B_frag" in source:
        return source
    if "__global half* tl_Bp = B + (tl_loop_aff1 - tl_wi_aff2);" not in source:
        return source
    if "read_only image2d_t" in source or "read_imageh" in source:
        return source

    loads = re.findall(r"convert_float8\(vload8\(0, tl_Bp(?: \+ (\d+))?\)\)", source)
    if len(loads) != 4:
        return source
    offs = sorted(int(x or "0") for x in loads)
    if offs[0] != 0 or len(set(offs)) != 4:
        return source
    stride = offs[1]
    if stride <= 0 or offs != [0, stride, stride * 2, stride * 3] or stride % 4:
        return source

    out = source
    out = out.replace("__global half* restrict B", "read_only image2d_t Bi")
    out = out.replace(
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n",
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
        "__constant sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
        "static inline float8 tl_read_b8_tex(read_only image2d_t Bi, int x4, int k) {\n"
        "  const half4 b_lo = read_imageh(Bi, smp, (int2)(x4, k));\n"
        "  const half4 b_hi = read_imageh(Bi, smp, (int2)(x4 + 1, k));\n"
        "  return convert_float8((half8)(b_lo, b_hi));\n"
        "}\n",
        1,
    )
    out = out.replace(
        "__global half* tl_Ap = A + tl_loop_aff0; __global half* tl_Bp = B + (tl_loop_aff1 - tl_wi_aff2);",
        f"__global half* tl_Ap = A + tl_loop_aff0; const int tl_B_tex_x = ((tl_loop_aff1 - tl_wi_aff2) - (pos4 * {stride * 4})) >> 2;",
        1,
    )
    for idx, off in enumerate(offs):
        old = "convert_float8(vload8(0, tl_Bp))" if off == 0 else f"convert_float8(vload8(0, tl_Bp + {off}))"
        out = out.replace(old, f"tl_read_b8_tex(Bi, tl_B_tex_x, (pos4 * 4) + {idx})")

    if "tl_Bp" in out or "__global half* restrict B" in out:
        return source
    return out


def _patch_opencl_unused_local_decls(source: str) -> str:
    """Drop simple function-local declarations whose name is never referenced.

    TVM/OpenCL lowering can leave behind dead private declarations after earlier
    peepholes scalar-replace fragments (e.g. ``float C_frag[64];`` or
    ``float8 acc_0;``).  Keep this deliberately textual and conservative:

    - only one declarator per line, with an indented function-body style;
    - only plain private scalar/vector types and constant-size arrays;
    - no initializer (so deleting the line cannot drop side effects);
    - delete only when the declared identifier has zero references elsewhere in
      the whole source.
    """

    # Limit the experimental pass to the GR-GEMM lowering shape that emits the
    # tl_wi_aff/tl_loop_aff address bases.  This keeps unrelated RR/FA kernels
    # stable unless they later grow the same GR scaffold.
    if "tl_wi_aff" not in source or "tl_loop_aff" not in source or "A_frag" in source or "B_frag" in source:
        return source

    decl_re = re.compile(
        r"(?m)^(?P<indent>[ \t]+)"
        r"(?P<type>(?:half|float|double|char|uchar|short|ushort|int|uint|long|ulong)(?:[2348]|16)?)"
        r"\s+(?P<name>[A-Za-z_]\w*)"
        r"(?:\s*\[\s*\d+\s*\])?\s*;\s*$"
    )

    out_parts: list[str] = []
    pos = 0
    for m in decl_re.finditer(source):
        name = m.group("name")
        # Exclude the declaration itself, then require no whole-token use.
        rest = source[: m.start()] + source[m.end():]
        if re.search(r"\b" + re.escape(name) + r"\b", rest):
            continue
        out_parts.append(source[pos:m.start()])
        # Preserve the line break by skipping only the declaration text; the
        # matched line includes no trailing newline under (?m)^...$, so consume
        # one if present to avoid leaving blank lines.
        pos = m.end()
        if pos < len(source) and source[pos] == "\n":
            pos += 1
    if not out_parts:
        return source
    out_parts.append(source[pos:])
    return "".join(out_parts)


def build_opencl(mod, target):
    built = _raw_build_opencl(mod, target)
    source = built.inspect_source()
    patched = _patch_opencl_workitem_id_hoist(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_workitem_affine_hoist(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_loop_affine_hoist(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_grgemm_base_pointers(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_grgemm_epilogue_direct_store(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_grgemm_b_texture(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_unused_local_decls(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_private_accumulators(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_half_staging(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_float8_split(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_typed_shared(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_gdn_dot8(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_gdn_seq_state_stream(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_gdn_seq_out_private_arrays(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_gdn_seq_pass1_float8(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_gdn_seq_output_vstore8(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_gdn_native_exp(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_gdn_scalar1_arrays(source)
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
    patched = _patch_opencl_splat_mul(source)
    if patched != source:
        source = patched
    patched = _patch_opencl_unused_local_decls(source)
    if patched != source:
        source = patched
    if source != built.inspect_source():
        return _SourceModule(built, source)
    return built
