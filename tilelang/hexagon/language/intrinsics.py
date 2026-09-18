"""Hexagon 扩展原语。"""

from __future__ import annotations

from typing import Any

from tilelang.language.tir.op import call_pure_extern


def _same_dtype_intrin(name: str, value: Any):
    """生成返回 dtype 与输入一致的占位 intrinsic call。"""

    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise TypeError(f"T.hexagon.{name} expects a TIR expression with dtype")
    # Keep these as ordinary TIR extern calls until the Hexagon emitter supports
    # lowering them.  Unregistered ``tirx.*`` intrinsic names fail during Python
    # TIR construction, before the Hexagon verifier can issue HexagonEmitError.
    return call_pure_extern(str(dtype), f"hexagon.{name}", value)


def exp_fp16(v):
    """fp16 近似 exp，占位为 TIR intrinsic + hexagon 注解。"""

    return _same_dtype_intrin("exp_fp16", v)


def exp_fp32(v):
    """fp32 域 exp，占位为 TIR intrinsic + hexagon 注解。"""

    return _same_dtype_intrin("exp_fp32", v)


def silu_fp16(v):
    """fp16 SiLU，占位为 TIR intrinsic + hexagon 注解。"""

    return _same_dtype_intrin("silu_fp16", v)


def h2f(v):
    """HVX fp16 向量转 fp32，占位 lowering 后续接 Q6 intrinsic。"""

    return call_pure_extern("float32", "hexagon.h2f", v)


def f2h(v):
    """HVX fp32 向量转 fp16，占位 lowering 后续接 Q6 intrinsic。"""

    return call_pure_extern("float16", "hexagon.f2h", v)


def dcfetch_hint(addr, dist: int = 8192):
    """dcfetch 预取 hint，占位为副作用 intrinsic。"""

    return call_pure_extern("handle", "hexagon.dcfetch_hint", addr, dist)


def _leaf(name: str, *args: Any):
    return call_pure_extern("handle", f"hexagon.{name}", *args)


def load_state128(src, dst, hv):
    """Load one head's 128x128 fp32 state into an explicit fp32 scratch tile."""

    return _leaf("load_state128", src, dst, hv)


def store_state128(src, dst, hv):
    """Store one head's explicit 128x128 fp32 state tile to output state."""

    return _leaf("store_state128", src, dst, hv)


def load_h2f_rows128(q, k, v, qf, kf, vf, T, hk, hv, t0):
    """Load Q/K/V chunk rows into explicit fp32 tiles using HVX vunpack."""

    return _leaf("load_h2f_rows128", q, k, v, qf, kf, vf, T, hk, hv, t0)


def scan_exp32(g, beta_src, eG, eGinv, beta, eGC, T, hv, t0):
    """Prefix gate recurrence into explicit eG/eGinv/beta/eGC scratch."""

    return _leaf("scan_exp32", g, beta_src, eG, eGinv, beta, eGC, T, hv, t0)


def dot128x2_store(kf, qf, eG, eGinv, beta, A, P, i, j):
    """Build A/P entries from explicit rows via two interleaved fp32 dot128 ops."""

    return _leaf("dot128x2_store", kf, qf, eG, eGinv, beta, A, P, i, j)


def state_x2_matvec128(S, kf, qf, w, o, i):
    """Compute S^T*k and S^T*q for one token using fp32 HVX accumulators."""

    return _leaf("state_x2_matvec128", S, kf, qf, w, o, i)


def affine_rows128(vf, w, beta, eG, i):
    """Apply beta/eG affine transform to one 128-wide fp32 row."""

    return _leaf("affine_rows128", vf, w, beta, eG, i)


def forward_solve32(A, w, i):
    """32-token lower-triangular forward substitution over 128-wide rows."""

    return _leaf("forward_solve32", A, w, i)


def output_rows128(o_acc, w, P, eG, out, T, hv, t0, i):
    """Apply lower-triangular P*w and fp32->fp16 store for one token row."""

    return _leaf("output_rows128", o_acc, w, P, eG, out, T, hv, t0, i)


def state_decay_rows128(kf, eGC, eGinv):
    """Fold chunk-final decay factors into the 32 K rows."""

    return _leaf("state_decay_rows128", kf, eGC, eGinv)


def state_update32(S, kf, w, eGC):
    """Chunk state update S=eGC*S+sum k*w^T with 128-wide fp32 rows."""

    return _leaf("state_update32", S, kf, w, eGC)


def reduce_sum32(src, dst):
    """Reduce one 32-lane fp16/fp32 row vector to a scalar output buffer.

    ``src`` is the row-start element load (e.g. ``Pbuf[kt, r, 0]``); the
    backend reads 32 lanes starting at that address.  A plain element load
    is used instead of a slice because sliced regions trip a known generic
    LowerTileOp substitution bug.
    """

    return _leaf("reduce_sum32", src, dst)


def reduce_sum128(src, dst):
    """Reduce one 128-lane fp16/fp32 vector to a scalar output buffer."""

    return _leaf("reduce_sum128", src, dst)


def reduce_max32(src, dst):
    """Reduce one 32-lane fp16/fp32 row vector to a scalar output buffer."""

    return _leaf("reduce_max32", src, dst)


def reduce_max128(src, dst):
    """Reduce one 128-lane fp16/fp32 row vector to a scalar output buffer."""

    return _leaf("reduce_max128", src, dst)


__all__ = (
    "exp_fp16", "exp_fp32", "silu_fp16", "h2f", "f2h", "dcfetch_hint",
    "load_state128", "store_state128", "load_h2f_rows128", "scan_exp32", "dot128x2_store",
    "state_x2_matvec128", "affine_rows128", "forward_solve32", "output_rows128",
    "state_decay_rows128", "state_update32", "reduce_sum32", "reduce_sum128", "reduce_max32", "reduce_max128",
)
