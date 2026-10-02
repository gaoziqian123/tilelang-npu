"""Explicit FIFO engine regions. No implicit buffer ownership transfer."""
from tilelang import language as T


def submit(slot):
    if not isinstance(slot, int) or not 0 <= slot < 8:
        raise ValueError("async slot must be static [0,8)")
    return T.attr(0, "hexagon.async_scope", slot)


def wait(slot):
    if not isinstance(slot, int) or not 0 <= slot < 8:
        raise ValueError("async slot must be static [0,8)")
    return T.evaluate(T.call_extern("handle", "tl::engine_wait", slot))
