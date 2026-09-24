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


__all__ = ["hmx_acc_clear", "hmx_set_bias", "hmx_mma_deep", "hmx_store_after"]
