"""Explicit external packed-storage contracts derived from public Layout maps.

Reference packing is CPU-only and bit-preserving. It is never inserted into a
device kernel or silently applied to an ordinary external pointer.
"""
import hashlib
import json
import math

import numpy as np
from tvm import arith, tirx


def compact_contract(layout, dtype="float16", alignment=128):
    """Prove and serialize a static layout without a per-element index table."""
    shape = [int(x) for x in layout.get_input_shape()]
    physical = [int(x) for x in layout.get_output_shape()]
    if not shape or min(shape+physical) <= 0:
        raise ValueError("positive static extents required")
    # The layout library's inverse construction rejects non-injective maps.
    layout.inverse()
    analyzer = arith.Analyzer()
    variables = layout.get_forward_vars()
    from tvm.ir import Range
    for v, n in zip(variables, shape):
        analyzer.bind(v, Range(0, n))
    expr = layout.get_linearized_forward_index()
    if not analyzer.can_prove(expr >= 0) or not analyzer.can_prove(expr < math.prod(physical)):
        raise ValueError("physical layout capacity unproved")
    names = {v: f"i{k}" for k, v in enumerate(variables)}
    def emit(e):
        if isinstance(e, tirx.IntImm):
            if int(e) < 0:
                raise ValueError("negative layout constant unsupported")
            return str(int(e))+"ULL"
        if isinstance(e, tirx.Var):
            return names[e]
        operators = {tirx.Add:"+", tirx.Mul:"*", tirx.FloorDiv:"/", tirx.FloorMod:"%"}
        for cls, op in operators.items():
            if isinstance(e, cls):
                if not analyzer.can_prove(e.a >= 0) or not analyzer.can_prove(e.b >= 0):
                    raise ValueError("nonnegative C index arithmetic unproved")
                if op in ("/", "%") and not analyzer.can_prove(e.b > 0):
                    raise ValueError("positive divisor required")
                bound=analyzer.const_int_bound(e)
                if int(bound.max_value) >= 2**63:
                    raise ValueError("layout index overflow")
                return f"({emit(e.a)} {op} {emit(e.b)})"
        raise ValueError(f"unsupported compact layout expression: {type(e).__name__}")
    result=dict(version=2,logical_dtype=dtype,logical_shape=shape,physical_extent=physical,
                physical_strides=[math.prod(physical[k+1:]) for k in range(len(physical))],
                alignment=alignment,padding="zero",bytes=math.prod(physical)*np.dtype(dtype).itemsize,
                logical_bytes=math.prod(shape)*np.dtype(dtype).itemsize,
                external_pointer_contract="packed-bytes-required",index_c=emit(expr),
                cache_policy="none; caller must repack changed input and version any weight cache")
    result["sha256"]=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return result


def c_packer(spec, name):
    if not name.isidentifier():
        raise ValueError("invalid C packer name")
    size=np.dtype(spec["logical_dtype"]).itemsize
    code=["#include <stdint.h>","#include <stddef.h>","#include <string.h>",
          f"/* layout sha256: {spec['sha256']} */",
          f"static inline int {name}(const void *logical, void *packed, size_t logical_cap, size_t packed_cap) {{",
          f"if (!logical || !packed || logical_cap < {spec['logical_bytes']}ULL || packed_cap < {spec['bytes']}ULL) return -1;",
          f"if ((uintptr_t)packed % {spec['alignment']}u) return -2;",
          f"if ((uintptr_t)logical > UINTPTR_MAX-{spec['logical_bytes']}ULL || (uintptr_t)packed > UINTPTR_MAX-{spec['bytes']}ULL) return -3;",
          f"if ((uintptr_t)logical < (uintptr_t)packed+{spec['bytes']}ULL && (uintptr_t)packed < (uintptr_t)logical+{spec['logical_bytes']}ULL) return -4;",
          f"memset(packed,0,{spec['bytes']}ULL);", "size_t ordinal=0;"]
    for k,n in enumerate(spec['logical_shape']):
        code.append(f"for (size_t i{k}=0;i{k}<{n}ULL;++i{k}) {{")
    code.append(f"memcpy((unsigned char*)packed+({spec['index_c']})*{size}, (const unsigned char*)logical+(ordinal++)*{size},{size});")
    code.extend("}" for _ in spec['logical_shape'])
    code.extend(["return 0;", "}"])
    return "\n".join(code)+"\n"


def contract(layout, dtype="float16", alignment=128, max_elements=16_777_216):
    shape = tuple(int(x) for x in layout.get_input_shape())
    physical = tuple(int(x) for x in layout.get_output_shape())
    if not shape or min(shape + physical) <= 0:
        raise ValueError("positive static layout extents required")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a power of two")
    if max(math.prod(shape), math.prod(physical)) > max_elements:
        raise ValueError("layout exceeds explicit reference validation capacity")
    itemsize = np.dtype(dtype).itemsize
    if dtype not in ("float16", "float32", "int8", "uint8", "int16", "uint16", "int32", "uint32"):
        raise ValueError("unsupported logical dtype")
    expr = layout.get_linearized_forward_index()
    variables = layout.get_forward_vars()
    analyzer = arith.Analyzer()
    offsets = []
    for index in np.ndindex(shape):
        value = analyzer.simplify(tirx.stmt_functor.substitute(expr, {
            v: tirx.IntImm(v.dtype, i) for v, i in zip(variables, index)
        }))
        if not isinstance(value, tirx.IntImm):
            raise ValueError("unresolved layout expression")
        offsets.append(int(value))
    if min(offsets) < 0 or max(offsets) >= math.prod(physical):
        raise ValueError("layout out of physical bounds")
    if len(set(offsets)) != len(offsets):
        raise ValueError("non-injective layout aliases logical elements")
    strides = [math.prod(physical[i+1:]) for i in range(len(physical))]
    result = dict(version=1, logical_dtype=dtype, logical_shape=list(shape),
                  physical_extent=list(physical), physical_strides=strides,
                  alignment=alignment, padding="zero", bytes=math.prod(physical)*itemsize,
                  external_pointer_contract="packed-bytes-required", offsets=offsets)
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def pack(array, spec):
    array = np.asarray(array)
    if list(array.shape) != spec["logical_shape"] or str(array.dtype) != spec["logical_dtype"]:
        raise ValueError("logical input shape/dtype mismatch")
    size = array.dtype.itemsize
    physical = np.zeros((spec["bytes"]//size, size), dtype=np.uint8)
    physical[spec["offsets"]] = np.ascontiguousarray(array).view(np.uint8).reshape(-1, size)
    return physical.reshape(-1)


def unpack(packed, spec):
    raw = np.asarray(packed)
    if raw.dtype != np.uint8 or raw.size != spec["bytes"]:
        raise ValueError("packed byte capacity mismatch")
    dtype = np.dtype(spec["logical_dtype"])
    return raw.reshape(-1, dtype.itemsize)[spec["offsets"]].copy().view(dtype).reshape(spec["logical_shape"])


def copy_spans(source, destination, source_min, destination_min, extent):
    """Exact static mapping proof; coalesce contiguous runs then regular 2D rows.

    Descriptors are byte offsets, not pointer addresses. No assumption about
    layout names, global tile order, equal objects, or panel contiguity.
    """
    if source["logical_dtype"] != destination["logical_dtype"]:
        raise ValueError("physical copy cannot convert dtype")
    size = np.dtype(source["logical_dtype"]).itemsize
    pairs = []
    for index in np.ndindex(tuple(extent)):
        positions = []
        for spec, minimum in ((source, source_min), (destination, destination_min)):
            coord = tuple(a+b for a, b in zip(index, minimum))
            shape = spec["logical_shape"]
            if len(coord) != len(shape) or any(x < 0 or x >= n for x, n in zip(coord, shape)):
                raise ValueError("copy region out of bounds")
            positions.append(spec["offsets"][np.ravel_multi_index(coord, shape)] * size)
        pairs.append(positions)
    pairs.sort(key=lambda p: p[1])
    spans = []
    for src, dst in pairs:
        if spans and src == spans[-1][0]+spans[-1][2] and dst == spans[-1][1]+spans[-1][2]:
            spans[-1][2] += size
        else:
            spans.append([src, dst, size])
    result = []
    i = 0
    while i < len(spans):
        src, dst, width = spans[i]
        rows, ss, ds = 1, width, width
        if i+1 < len(spans) and spans[i+1][2] == width:
            ss, ds = spans[i+1][0]-src, spans[i+1][1]-dst
            if ss >= width and ds >= width:
                while i+rows < len(spans) and spans[i+rows] == [src+rows*ss, dst+rows*ds, width]:
                    rows += 1
        result.append(dict(source_offset=src, destination_offset=dst, width_bytes=width,
                           rows=rows, source_stride=ss, destination_stride=ds))
        i += rows
    return result
