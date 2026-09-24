"""Canonical Hexagon-v2 intrinsic surface.

Phase 1 needs one real intrinsic: ``hexagon.hmx_mma_deep``.  The Python object
records the schema that the eventual TIR lowering/codegen will use; the C source
emitter in :mod:`codegen` lowers it into the self-authored inline assembly
sequence validated by ``attnops_hmx_deep_probe.c``.
"""

from __future__ import annotations

from dataclasses import dataclass



HMX_ACC_CLEAR = "hexagon.hmx_acc_clear"
HMX_SET_BIAS = "hexagon.hmx_set_bias"
HMX_MMA_DEEP = "hexagon.hmx_mma_deep"
HMX_STORE_AFTER = "hexagon.hmx_store_after"


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
    "HMXAccClear",
    "HMXSetBias",
    "HMXMMADeep",
    "HMXStoreAfter",
    "hmx_acc_clear",
    "hmx_set_bias",
    "hmx_mma_deep",
    "hmx_store_after",
]
