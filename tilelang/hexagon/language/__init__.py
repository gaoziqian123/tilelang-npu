"""Hexagon language dialect: common TileLang + Hexagon 扩展。"""

from __future__ import annotations

from typing import Any

from types import SimpleNamespace

from tvm import tirx


from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.allocate import alloc_shared as _alloc_shared
from tilelang.language.allocate import alloc_local as _alloc_local
from tilelang.language.allocate import alloc_fragment as _alloc_fragment
from tilelang.language.copy_op import copy as _copy
from tilelang.language.proxy import Tensor as _Tensor
from tilelang.language.utils import _normalize_annotations

from .intrinsics import *  # noqa: F401,F403
from .intrinsics import __all__ as _INTRINSICS_ALL
from .layout import *  # noqa: F401,F403
from .layout import __all__ as _LAYOUT_ALL
from .layout import make_layout
from .scopes import *  # noqa: F401,F403
from .scopes import __all__ as _SCOPES_ALL


def _validate_layout(layout: str | None) -> str:
    if layout is None:
        return "rm"
    if layout not in ("rm", "ah", "wh"):
        raise ValueError(f"Hexagon layout must be 'rm', 'ah' or 'wh', got {layout!r}")
    return layout


def alloc_shared(shape, dtype, scope: str = "vtcm", *, layout: str | None = None):
    """分配 Hexagon VTCM buffer。

    用户仍写 ``T.alloc_shared``；Hexagon dialect 默认把它映射到 ``vtcm``。
    当前 Python 骨架为了让 layout 在 TIR 中可见，将非默认 layout 暂编码进
    scope 后缀（如 ``vtcm.ah``）。后续正式 layout pass 应改为读取 buffer
    layout map，而不是依赖 scope 字符串。
    """

    layout = _validate_layout(layout)
    physical_scope = scope if layout == "rm" else f"{scope}.{layout}"
    buffer = _alloc_shared(shape, dtype, scope=physical_scope)
    if layout in ("ah", "wh"):
        annotate_layout({buffer: make_layout(layout, buffer)})
    return buffer


def Tensor(shape, dtype="float32", data=None, scope=None, *, layout: str | None = None, **kwargs):
    """Declare a Hexagon global tensor, optionally with AH/WH physical layout.

    The normal TileLang ``T.Tensor`` surface has no Hexagon-specific layout
    metadata.  The Hexagon dialect extends it with ``layout=\"rm|ah|wh\"`` and
    encodes non-row-major global layouts as scope suffixes (``global.wh``),
    mirroring the existing VTCM convention (``vtcm.wh``).  Emitters and lowering
    still treat ``global.*`` as ABI-global buffers; the suffix is only generic
    layout metadata.
    """

    layout = _validate_layout(layout)
    physical_scope = scope
    if layout in ("ah", "wh"):
        base_scope = scope or "global"
        physical_scope = f"{base_scope}.{layout}"
    return _Tensor(shape, dtype=dtype, data=data, scope=physical_scope, **kwargs)


def alloc_wscratch(shape, dtype):
    """分配 per-worker DDR scratch buffer。

    ``wscratch`` 不占 VTCM 预算,在生成的 skel C 中映射到每个 worker
    独立的 runtime slot 字段。它用于 GDN 这类含标量递推/随机访问的
    scratch,避免把标量访问错误地下沉到 VTCM。
    """

    return _alloc_shared(shape, dtype, scope="wscratch")


def alloc_local(shape, dtype, scope: str = "vreg"):
    """分配 HVX vreg 语义的 local buffer。"""

    return _alloc_local(shape, dtype, scope=scope)


def alloc_fragment(shape, dtype, scope: str = "hmx.acc"):
    """分配 HMX accumulator fragment。"""

    return _alloc_fragment(shape, dtype, scope=scope)


def copy(src, dst, *, layout: tuple[str, str] | None = None, annotations: dict | None = None, **kwargs):
    """Hexagon layout-aware copy。

    ``layout=(\"rm\", \"ah\")`` / ``(\"ah\", \"rm\")`` 会以 annotation 形式
    留在 ``tl.tileop.copy`` 上，供后续 copy lowering 选择 zip16/unperm recipe。
    """

    ann = _normalize_annotations(annotations)
    if layout is None:
        def _obj_layout(obj) -> str:
            buf = getattr(obj, "buffer", None) or obj
            try:
                scope = str(buf.scope())
            except Exception:
                return "rm"
            if scope.endswith(".ah"):
                return "ah"
            if scope.endswith(".wh"):
                return "wh"
            return "rm"

        inferred = (_obj_layout(src), _obj_layout(dst))
        if inferred != ("rm", "rm"):
            layout = inferred
    if layout is not None:
        if tuple(layout) not in (("rm", "ah"), ("ah", "rm"), ("wh", "rm"), ("rm", "wh"), ("wh", "wh")):
            raise ValueError(f"Hexagon copy layout must be ('rm','ah'), ('ah','rm'), ('wh','rm'), ('rm','wh') or ('wh','wh'), got {layout!r}")
        ann["hexagon.copy.src_layout"] = tirx.StringImm(layout[0])
        ann["hexagon.copy.dst_layout"] = tirx.StringImm(layout[1])
    return _copy(src, dst, annotations=ann, **kwargs)


def reduce_sum(buffer, out, dim: int = -1, clear: bool = True, batch: int = 1, annotations: dict | None = None) -> None:
    """Hexagon 128-lane vector->scalar sum.

    TileLang's generic reducer currently only accepts CUDA-style
    shared/fragment combinations.  Hexagon lowers the natural T.reduce_sum
    spelling directly to a 128-wide HVX reduce helper.
    """

    if dim not in (-1, 0):
        raise ValueError(f"Hexagon reduce_sum currently supports dim=0/-1, got {dim}")
    shape = getattr(buffer, "shape", None) or getattr(getattr(buffer, "buffer", None), "shape", None)
    try:
        n = 1
        for s in shape or ():
            n *= int(s)
    except Exception:
        n = None
    # A 32-lane row is passed as its row-start element load (buf[.., 0]);
    # sliced regions trip a known generic LowerTileOp substitution bug.
    return reduce_sum128(buffer, out) if n == 128 else reduce_sum32(buffer, out)


def reduce_max(buffer, out, dim: int = -1, clear: bool = True, batch: int = 1, nan_propagate: bool = False, annotations: dict | None = None) -> None:
    """Hexagon row max for 32-lane FA rows or 128-lane row vectors.

    This is intentionally row-major only: callers copy HMX score tiles out of AH
    first, then reduce each row with an HVX rotate-fold helper.
    """

    if dim not in (-1, 0):
        raise ValueError(f"Hexagon reduce_max currently supports dim=0/-1, got {dim}")
    shape = getattr(buffer, "shape", None) or getattr(getattr(buffer, "buffer", None), "shape", None)
    try:
        n = 1
        for s in shape or ():
            n *= int(s)
    except Exception:
        n = None
    return reduce_max128(buffer, out) if n == 128 else reduce_max32(buffer, out)


# 对齐文档中的 ``T.hexagon.exp_fp16`` 写法，同时函数也直接导出。
hexagon = SimpleNamespace(
    exp_fp16=exp_fp16,
    exp_fp32=exp_fp32,
    silu_fp16=silu_fp16,
    h2f=h2f,
    f2h=f2h,
    dcfetch_hint=dcfetch_hint,
    reduce_sum128=reduce_sum128,
    reduce_max32=reduce_max32,
    reduce_max128=reduce_max128,
    gdn_prefill=gdn_prefill,
    load_state128=load_state128,
    store_state128=store_state128,
    load_h2f_rows128=load_h2f_rows128,
    scan_exp32=scan_exp32,
    dot128x2_store=dot128x2_store,
    state_x2_matvec128=state_x2_matvec128,
    affine_rows128=affine_rows128,
    forward_solve32=forward_solve32,
    output_rows128=output_rows128,
    state_decay_rows128=state_decay_rows128,
    state_update32=state_update32,
)

__tilelang_dialect__ = "hexagon"
__all__ = tuple(
    dict.fromkeys(
        (
            *_COMMON_ALL,
            *_INTRINSICS_ALL,
            *_LAYOUT_ALL,
            *_SCOPES_ALL,
            "alloc_shared",
            "alloc_wscratch",
            "alloc_local",
            "alloc_fragment",
            "Tensor",
            "copy",
            "reduce_sum",
            "reduce_max",
            "hexagon",
        )
    )
)

del _COMMON_ALL, _INTRINSICS_ALL, _LAYOUT_ALL, _SCOPES_ALL, Any
