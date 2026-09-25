"""Hexagon-v2 low-level TIR intrinsic wrappers."""

from __future__ import annotations

from tvm import tirx
from tvm.tirx.expr import IntImm

from tilelang.language.tir.op import call_intrin as _call_intrin


def _i32(value):
    return IntImm("int32", value) if isinstance(value, int) else value


def _u32(value):
    return IntImm("uint32", value) if isinstance(value, int) else value


def hmx_acc_clear(acc=0):
    return _call_intrin("handle", "tl.hexagon_v2.hmx_acc_clear", _i32(acc))


def hmx_set_bias(scale_bias_ptr):
    return _call_intrin("handle", "tl.hexagon_v2.hmx_set_bias", scale_bias_ptr)


def hmx_mma_deep(ah_ptr, wh_ptr, chain_len, ah_stride=2048, wh_stride=2048):
    return _call_intrin(
        "handle",
        "tl.hexagon_v2.hmx_mma_deep",
        ah_ptr,
        wh_ptr,
        _i32(chain_len),
        _u32(ah_stride),
        _u32(wh_stride),
    )


def hmx_store_after(out_ptr):
    return _call_intrin("handle", "tl.hexagon_v2.hmx_store_after", out_ptr)


def hvx_load(ptr, dtype="u8x128", align=128):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_load", ptr, dtype, _i32(align))


def hvx_store(ptr, value, dtype="u8x128", align=128):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_store", ptr, value, dtype, _i32(align))


def hvx_add(a, b, dtype="f16"):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_add", a, b, dtype)


def hvx_mul(a, b, dtype="f16"):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_mul", a, b, dtype)


def hvx_fma(acc, a, b, dtype="f32"):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_fma", acc, a, b, dtype)


def hvx_exp(value, dtype="f16"):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_exp", value, dtype)


def hvx_reduce_max(ptr, dtype="f32", lanes=128):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_reduce_max", ptr, dtype, _i32(lanes))


def hvx_reduce_sum(ptr, dtype="f32", lanes=128):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_reduce_sum", ptr, dtype, _i32(lanes))


def hvx_h2f(value):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_h2f", value)


def hvx_f2h(lo, hi):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_f2h", lo, hi)


def hvx_copy(dst, src, nbytes, kind="pooled_dcfetch"):
    return _call_intrin("handle", "tl.hexagon_v2.hvx_copy", dst, src, _u32(nbytes), kind)


__all__ = [
    "hmx_acc_clear",
    "hmx_set_bias",
    "hmx_mma_deep",
    "hmx_store_after",
    "hvx_load",
    "hvx_store",
    "hvx_add",
    "hvx_mul",
    "hvx_fma",
    "hvx_exp",
    "hvx_reduce_max",
    "hvx_reduce_sum",
    "hvx_h2f",
    "hvx_f2h",
    "hvx_copy",
]
