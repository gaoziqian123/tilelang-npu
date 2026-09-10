"""Hexagon scope 常量。"""

from __future__ import annotations

GLOBAL = "global"
VTCM = "vtcm"
WSCRATCH = "wscratch"
VREG = "vreg"
HMX_ACC = "hmx.acc"

# 小写别名对齐用户文档中的写法。
global_scope = GLOBAL
vtcm = VTCM
wscratch = WSCRATCH
vreg = VREG
hmx_acc = HMX_ACC

__all__ = (
    "GLOBAL", "VTCM", "WSCRATCH", "VREG", "HMX_ACC",
    "global_scope", "vtcm", "wscratch", "vreg", "hmx_acc",
)
