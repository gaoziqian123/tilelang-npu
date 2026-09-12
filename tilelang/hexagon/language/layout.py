"""Hexagon layout constants and HMX AH/WH layout constructors."""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType, BufferLikeTypeTuple
from tilelang.layout.swizzle import _get_buffer_info
from tilelang.hexagon import _ffi_api

RM = "rm"
AH = "ah"
WH = "wh"
NONE = "none"

rm = RM
ah = AH
wh = WH


def make_ah_layout(shape_or_buffer):
    """Create the Hexagon HMX AH zip16 layout.

    ``shape_or_buffer`` may be a shape sequence or a TileLang/TVM buffer-like
    object.  The last two logical dimensions must be multiples of 32.
    """
    if isinstance(shape_or_buffer, (tirx.Buffer, BufferLikeTypeTuple)):
        buf, _, _ = _get_buffer_info(shape_or_buffer)
        return _ffi_api.make_layout(AH, buf)
    return _ffi_api.make_ah_layout(list(shape_or_buffer))


def make_wh_layout(buffer: BufferLikeType):
    """Create the Hexagon HMX WH layout for a 16-bit buffer.

    For one 32x32 HMX tile, WH is byte-identical to AH: both are the zip16
    row-pair interleave produced by one HVX ``vshuff`` from row-major input.
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_layout(WH, buf)


def make_rm_layout(buffer: BufferLikeType):
    """Create the canonical row-major/identity Hexagon layout for ``buffer``."""
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_layout(RM, buf)


def make_layout(mode: str, buffer: BufferLikeType):
    """Create a registered Hexagon layout for ``buffer``.

    New C++ layouts should be exposed here without adding Python-side
    constructor branches: extend the C++ enum/registry table and pass its mode
    name to this helper.
    """
    buf, _, _ = _get_buffer_info(buffer)
    return _ffi_api.make_layout(mode, buf)


def detect_layout_mode(layout, buffer: BufferLikeType) -> str:
    """Return the registered Hexagon layout mode name, or ``"none"``."""
    buf, _, _ = _get_buffer_info(buffer)
    return str(_ffi_api.detect_layout_mode(layout, buf))


def detect_ah_mode(layout, buffer: BufferLikeType) -> str:
    """Return ``"ah"`` when ``layout`` is the canonical AH layout, else ``"none"``."""
    buf, _, _ = _get_buffer_info(buffer)
    return AH if int(_ffi_api.detect_ah_mode(layout, buf)) == 1 else NONE


def detect_wh_mode(layout, buffer: BufferLikeType) -> str:
    """Return ``"wh"`` when ``layout`` is the canonical WH layout, else ``"none"``."""
    buf, _, _ = _get_buffer_info(buffer)
    return WH if int(_ffi_api.detect_wh_mode(layout, buf)) == 2 else NONE


def detect_rm_mode(layout, buffer: BufferLikeType) -> str:
    """Return ``"rm"`` when ``layout`` is canonical row-major, else ``"none"``."""
    buf, _, _ = _get_buffer_info(buffer)
    return RM if int(_ffi_api.detect_rm_mode(layout, buf)) == 3 else NONE

__all__ = (
    "RM",
    "AH",
    "WH",
    "NONE",
    "rm",
    "ah",
    "wh",
    "make_layout",
    "make_ah_layout",
    "make_wh_layout",
    "make_rm_layout",
    "detect_layout_mode",
    "detect_ah_mode",
    "detect_wh_mode",
    "detect_rm_mode",
)
