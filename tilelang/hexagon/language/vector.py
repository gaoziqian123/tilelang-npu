"""Explicit 128-byte register IR; unsupported semantics are errors, not hints.

No implicit contraction, approximation, tail read, or logical-layout inference.
The caller must use normal TIR memory operations until memory proofs are wired.
"""
from tvm import tirx as tir
from tvm.ir import Op


def leaf(name, *args, mode="bits_v1", immediate=0):
    """Create a checked physical HVX call (see vector_leaf_status.md)."""
    unary = {"rotate", "interleave16", "deal16", "abs16", "neg16", "abs32", "neg32",
             "sum32_asc", "max32_asc", "widen_linear", "widen_evenodd", "rcp32_nr2",
             "broadcast16", "broadcast32", "exp2_16_bounded", "exp2_16_nonpositive", "exp32_v1", "pred_not"}
    binary = {"align", "narrow_evenodd", "narrow_linear", "div32_nr2", "compare16", "compare32"} | {op + bits for op in ("add", "sub", "mul", "min", "max") for bits in ("16", "32")}
    binary |= {"pred_and", "pred_or"}
    count = 1 if name in unary else 2 if name in binary else 3 if name in ("select", "fma16") else None
    if count is None:
        raise ValueError(f"unsupported HVX leaf: {name}")
    if len(args) != count:
        raise ValueError(f"{name} requires {count} operands")
    native = name in {op + bits for op in ("add", "sub", "mul", "min", "max") for bits in ("16", "32")} | {"sum32_asc", "max32_asc"}
    expected = ("native_target_v1" if native or name == "fma16" else "ieee_bits_v1" if name in ("compare16", "compare32") else "strict_exact_v1" if name in ("widen_linear", "widen_evenodd")
                else "rne_gradual_quiet_payload_v1" if name in ("narrow_evenodd", "narrow_linear")
                else "nr2_positive_bounded_v1" if name in ("rcp32_nr2", "div32_nr2")
                else "qf_poly6_neg14_0_v1" if name == "exp2_16_bounded"
                else "ggml_qf16_clamp24_v1" if name == "exp2_16_nonpositive"
                else "hvx_math32_v1" if name == "exp32_v1" else "bits_v1")
    if mode != expected:
        raise ValueError(f"{name} requires explicit mode={expected}")
    if type(immediate) is not int:
        raise ValueError("HVX immediate must be a static integer")
    if name in ("rotate", "align"):
        if not 0 <= immediate < 128:
            raise ValueError("byte shift outside [0,128)")
    elif name in ("widen_linear", "widen_evenodd"):
        if immediate not in (0, 1):
            raise ValueError("widen part must be 0 or 1")
    elif name in ("broadcast16", "broadcast32"):
        if not 0 <= immediate < (64 if name == "broadcast16" else 32):
            raise ValueError("broadcast lane out of range")
    elif name in ("compare16", "compare32"):
        if not 0 <= immediate < 12:
            raise ValueError("invalid relation/policy")
    elif immediate != 0:
        raise ValueError("unexpected immediate")
    # Raw physical values use u8x128. Type/rounding are in the operation name
    # and version, never inferred from an enclosing kernel or loop.
    if any(str(a.dtype) != "uint8x128" for a in args):
        raise ValueError("physical HVX operands must be uint8x128 bit views")
    return tir.call_intrin("uint8x128", Op.get("tl.hexagon.vector_leaf"), name, mode, immediate, *args)


def _io(dtype, name, *args):
    return tir.call_intrin(dtype, Op.get("tl.hexagon.vector_io"), name, *args)


def bitcast(value, dtype):
    from tvm import DataType
    a, b = DataType(value.dtype), DataType(dtype)
    if a.bits*a.lanes != 1024 or b.bits*b.lanes != 1024:
        raise ValueError("bitcast requires two 128B views")
    return _io(dtype, "bitcast", value)


def splat_bits(bits, dtype):
    from tvm import DataType
    d = DataType(dtype)
    if d.bits not in (16, 32) or d.bits*d.lanes != 1024:
        raise ValueError("splat requires a 128B 16/32-bit view")
    return _io(dtype, "splat"+str(d.bits), tir.Cast("uint32", bits))


def extract_bits(value, lane):
    from tvm import DataType
    d = DataType(value.dtype)
    if d.bits not in (16, 32) or d.bits*d.lanes != 1024 or type(lane) is not int or not 0 <= lane < d.lanes:
        raise ValueError("invalid physical extract")
    return _io("uint32", "extract"+str(d.bits), value, lane)


def _region(buffer, offset):
    # Deliberately static and compact. Base alignment is guarded at runtime by
    # the leaf; allocation size/offset are proven, not caller-supplied booleans.
    from tvm import DataType
    d = DataType(buffer.dtype)
    if len(buffer.shape) != 1 or buffer.strides or d.bits not in (16, 32):
        raise ValueError("HVX memory requires compact rank-one 16/32-bit buffer")
    if not isinstance(buffer.shape[0], tir.IntImm) or type(offset) is not int:
        raise ValueError("HVX memory requires static bounds and offset")
    width = 1024 // d.bits
    if offset < 0 or offset % width or offset + width > int(buffer.shape[0]):
        raise ValueError("full aligned 128B region not proven (unpadded tail unsupported)")
    if not isinstance(buffer.elem_offset, tir.IntImm) or int(buffer.elem_offset) != 0:
        raise ValueError("nonzero/dynamic buffer base offset unsupported")
    return width, str(buffer.dtype)+"x"+str(width)


def load(buffer, offset=0, *, valid_lanes=None, fill_bits=0):
    width, dtype = _region(buffer, offset)
    valid = width if valid_lanes is None else valid_lanes
    if type(valid) is not int or not 0 <= valid <= width:
        raise ValueError("invalid logical tail")
    return _io(dtype, "load", tir.call_intrin("handle", "tirx.address_of", buffer[offset]),
               valid*(128//width), splat_bits(fill_bits, dtype))


def store(buffer, value, offset=0, *, valid_lanes=None):
    width, dtype = _region(buffer, offset)
    valid = width if valid_lanes is None else valid_lanes
    if str(value.dtype) != dtype or type(valid) is not int or not 0 <= valid <= width:
        raise ValueError("store type or tail mismatch")
    return _io("int32", "store", tir.call_intrin("handle", "tirx.address_of", buffer[offset]),
               valid*(128//width), value)


def map_chunks(name, *values, mode, immediate=0):
    """Logical static vectors represented as tuples of typed 128B chunks.

    No oversized vector temporaries or scalar gathers. Every chunk is lowered
    through the same closed physical registry. Cast changes chunk count.
    """
    if not values or not values[0] or any(len(v) != len(values[0]) for v in values):
        raise ValueError("logical chunk counts must match and be nonempty")
    out = []
    for operands in zip(*values):
        dtype = str(operands[0].dtype)
        if dtype not in ("float16x64", "float32x32"):
            raise ValueError("logical arithmetic requires typed FP16/FP32 chunks")
        bits = "16" if dtype == "float16x64" else "32"
        if name in {op+b for op in ("add", "sub", "mul", "min", "max", "abs", "neg", "fma", "compare", "broadcast") for b in ("16", "32")} and not name.endswith(bits):
            raise ValueError("logical operation dtype mismatch")
        if any(str(x.dtype) != dtype for x in operands):
            raise ValueError("logical operand dtype mismatch")
        raw = [bitcast(x, "uint8x128") for x in operands]
        if name == "widen":
            if dtype != "float16x64" or len(raw) != 1:
                raise ValueError("widen requires half chunks")
            for part in (0, 1):
                out.append(bitcast(leaf("widen_linear", raw[0], mode=mode, immediate=part), "float32x32"))
        else:
            out.append(bitcast(leaf(name, *raw, mode=mode, immediate=immediate), dtype))
    return tuple(out)


def narrow_chunks(values, *, mode="rne_gradual_quiet_payload_v1", order="linear"):
    if not values or len(values)%2 or any(str(v.dtype)!="float32x32" for v in values):
        raise ValueError("narrow needs pairs of FP32 chunks")
    if order not in ("linear", "evenodd"):
        raise ValueError("unsupported narrowing order")
    return tuple(bitcast(leaf("narrow_"+order, bitcast(values[i],"uint8x128"),
                            bitcast(values[i+1],"uint8x128"), mode=mode),"float16x64")
                 for i in range(0,len(values),2))


def load_chunks(buffer, lanes, offset=0, *, fill_bits=0):
    """Split a static logical region, proving padded storage for its last chunk."""
    width, _ = _region(buffer, offset)
    if type(lanes) is not int or lanes <= 0:
        raise ValueError("logical length must be positive and static")
    return tuple(load(buffer, offset+i, valid_lanes=min(width,lanes-i), fill_bits=fill_bits)
                 for i in range(0,lanes,width))


def store_chunks(buffer, values, lanes, offset=0):
    width, _ = _region(buffer, offset)
    if type(lanes) is not int or lanes<=0 or len(values)!=(lanes+width-1)//width:
        raise ValueError("logical length/chunk mismatch")
    stmts=[tir.Evaluate(store(buffer,x,offset+i*width,
                          valid_lanes=min(width,lanes-i*width))) for i,x in enumerate(values)]
    return stmts[0] if len(stmts)==1 else tir.SeqStmt(stmts)


def compare_chunks(a, b, *, relation, nan_policy="ordered"):
    relations=("eq","ne","lt","le","gt","ge")
    if relation not in relations or nan_policy not in ("ordered","unordered"):
        raise ValueError("unsupported comparison policy")
    bits="16" if a and str(a[0].dtype)=="float16x64" else "32"
    return tuple(bitcast(x,"uint8x128") for x in map_chunks("compare"+bits,a,b,
                 mode="ieee_bits_v1",immediate=relations.index(relation)+(6 if nan_policy=="unordered" else 0)))


def permute(value, indices, *, lane_bytes):
    """Only cyclic maps and the documented halfword shuffle/deal are supported."""
    if lane_bytes not in (1, 2, 4):
        raise ValueError("unsupported lane size")
    n = 128 // lane_bytes
    indices = tuple(indices)
    if len(indices) != n or any(type(i) is not int or not 0 <= i < n for i in indices):
        raise ValueError("invalid static lane map")
    if indices == tuple((i + indices[0]) % n for i in range(n)):
        return leaf("rotate", value, immediate=indices[0] * lane_bytes)
    if lane_bytes == 2:
        if indices == tuple(i // 2 + (i % 2) * 32 for i in range(64)):
            return leaf("interleave16", value)
        if indices == tuple(2 * (i % 32) + i // 32 for i in range(64)):
            return leaf("deal16", value)
    raise ValueError("unsupported HVX permutation network")


def permute2(a, b, indices, *, lane_bytes):
    """A contiguous 128B window into [a,b], including either identity."""
    if lane_bytes not in (1, 2, 4):
        raise ValueError("unsupported lane size")
    n = 128 // lane_bytes
    indices = tuple(indices)
    if len(indices) != n or any(type(i) is not int for i in indices):
        raise ValueError("invalid static lane map")
    start = indices[0]
    if not 0 <= start <= n or indices != tuple(range(start, start+n)):
        raise ValueError("unsupported HVX two-vector permutation network")
    if start == n:
        return leaf("rotate", b)
    return leaf("align", b, a, immediate=start * lane_bytes)
