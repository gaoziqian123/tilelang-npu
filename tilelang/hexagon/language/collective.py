"""Closed full-active-group FP32 reduction (no exposed partial tickets)."""
from tvm import tirx as tir
from tvm.ir import Op


def group_reduce(value, seed, scratch, *, workers, owner, op="sum"):
    if op not in ("sum", "max"):
        raise ValueError("group_reduce supports sum/max only")
    if workers not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("group_reduce requires power-of-two workers <=64")
    if scratch.dtype != "float32" or len(scratch.shape) != 1:
        raise ValueError("group_reduce scratch must be rank-one FP32")
    return tir.call_intrin("float32", Op.get("tl.hexagon.group_reduce"), op, workers,
                           owner, value, seed, tir.address_of(scratch[0]),
                           scratch.shape[0])
