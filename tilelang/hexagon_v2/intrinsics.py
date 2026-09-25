"""Canonical Hexagon-v2 intrinsic surface.

The objects in this module are the IR-level contract for Hexagon-v2 target
lowering.  Code generators may still emit C source in this phase, but HMX/HVX
operations must be expressed as these first-class intrinsics instead of source
text rewrites.  Runtime services (FastRPC ABI, worker-pool lifetime, VTCM
allocation) intentionally stay outside this surface.
"""

from __future__ import annotations

from dataclasses import dataclass



HMX_ACC_CLEAR = "hexagon.hmx_acc_clear"
HMX_SET_BIAS = "hexagon.hmx_set_bias"
HMX_MMA_DEEP = "hexagon.hmx_mma_deep"
HMX_STORE_AFTER = "hexagon.hmx_store_after"

HVX_LOAD = "hexagon.hvx.load"
HVX_STORE = "hexagon.hvx.store"
HVX_ADD = "hexagon.hvx.add"
HVX_MUL = "hexagon.hvx.mul"
HVX_FMA = "hexagon.hvx.fma"
HVX_EXP = "hexagon.hvx.exp"
HVX_REDUCE_MAX = "hexagon.hvx.reduce_max"
HVX_REDUCE_SUM = "hexagon.hvx.reduce_sum"
HVX_H2F = "hexagon.hvx.h2f"
HVX_F2H = "hexagon.hvx.f2h"
HVX_COPY = "hexagon.hvx.copy"


@dataclass(frozen=True)
class HVXLoad:
    ptr: str
    dtype: str = "u8x128"
    align: int = 128


@dataclass(frozen=True)
class HVXStore:
    ptr: str
    value: str
    dtype: str = "u8x128"
    align: int = 128


@dataclass(frozen=True)
class HVXBinary:
    a: str
    b: str
    dtype: str


@dataclass(frozen=True)
class HVXFMA:
    acc: str
    a: str
    b: str
    dtype: str


@dataclass(frozen=True)
class HVXExp:
    value: str
    dtype: str = "f16"


@dataclass(frozen=True)
class HVXReduce:
    ptr: str
    dtype: str
    lanes: int = 128


@dataclass(frozen=True)
class HVXConvert:
    value: str
    src_dtype: str
    dst_dtype: str


@dataclass(frozen=True)
class HVXCopy:
    dst: str
    src: str
    bytes: str
    kind: str = "pooled_dcfetch"
    align: int = 128


def hvx_load(ptr: str, dtype: str = "u8x128", align: int = 128) -> HVXLoad:
    return HVXLoad(ptr=ptr, dtype=dtype, align=align)


def hvx_store(ptr: str, value: str, dtype: str = "u8x128", align: int = 128) -> HVXStore:
    return HVXStore(ptr=ptr, value=value, dtype=dtype, align=align)


def hvx_add(a: str, b: str, dtype: str) -> HVXBinary:
    return HVXBinary(a=a, b=b, dtype=dtype)


def hvx_mul(a: str, b: str, dtype: str) -> HVXBinary:
    return HVXBinary(a=a, b=b, dtype=dtype)


def hvx_fma(acc: str, a: str, b: str, dtype: str) -> HVXFMA:
    return HVXFMA(acc=acc, a=a, b=b, dtype=dtype)


def hvx_exp(value: str, dtype: str = "f16") -> HVXExp:
    return HVXExp(value=value, dtype=dtype)


def hvx_reduce_max(ptr: str, dtype: str, lanes: int = 128) -> HVXReduce:
    return HVXReduce(ptr=ptr, dtype=dtype, lanes=lanes)


def hvx_reduce_sum(ptr: str, dtype: str, lanes: int = 128) -> HVXReduce:
    return HVXReduce(ptr=ptr, dtype=dtype, lanes=lanes)


def hvx_h2f(value: str) -> HVXConvert:
    return HVXConvert(value=value, src_dtype="f16", dst_dtype="f32x2")


def hvx_f2h(value: str) -> HVXConvert:
    return HVXConvert(value=value, src_dtype="f32x2", dst_dtype="f16")


def hvx_copy(dst: str, src: str, bytes: str, kind: str = "pooled_dcfetch") -> HVXCopy:
    return HVXCopy(dst=dst, src=src, bytes=bytes, kind=kind)


@dataclass(frozen=True)
class HMXAccClear:
    acc: str = "acc"


@dataclass(frozen=True)
class HMXSetBias:
    scale_bias_ptr: str


@dataclass(frozen=True)
class HMXMMADeep:
    """Schema for one deep-chain HMX MMA intrinsic.

    Parameters are byte offsets into VTCM; ``chain_len`` is the number of
    32x32x32 fp16 tiles in this chain and must be ``1..32``.
    """

    ah_off: str
    wh_off: str
    chain_len: str
    ah_stride: str = "HRT_TILE_BYTES"
    wh_stride: str = "HRT_TILE_BYTES"


def hmx_mma_deep(ah_off: str, wh_off: str, chain_len: str) -> HMXMMADeep:
    return HMXMMADeep(ah_off=ah_off, wh_off=wh_off, chain_len=chain_len)


@dataclass(frozen=True)
class HMXStoreAfter:
    out_ptr: str


def hmx_acc_clear(acc: str = "acc") -> HMXAccClear:
    return HMXAccClear(acc=acc)


def hmx_set_bias(scale_bias_ptr: str) -> HMXSetBias:
    return HMXSetBias(scale_bias_ptr=scale_bias_ptr)


def hmx_store_after(out_ptr: str) -> HMXStoreAfter:
    return HMXStoreAfter(out_ptr=out_ptr)


__all__ = [
    "HMX_ACC_CLEAR",
    "HMX_SET_BIAS",
    "HMX_MMA_DEEP",
    "HMX_STORE_AFTER",
    "HVX_LOAD",
    "HVX_STORE",
    "HVX_ADD",
    "HVX_MUL",
    "HVX_FMA",
    "HVX_EXP",
    "HVX_REDUCE_MAX",
    "HVX_REDUCE_SUM",
    "HVX_H2F",
    "HVX_F2H",
    "HVX_COPY",
    "HMXAccClear",
    "HMXSetBias",
    "HMXMMADeep",
    "HMXStoreAfter",
    "HVXLoad",
    "HVXStore",
    "HVXBinary",
    "HVXFMA",
    "HVXExp",
    "HVXReduce",
    "HVXConvert",
    "HVXCopy",
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
