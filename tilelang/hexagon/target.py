"""Hexagon target 归一化。"""

from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_normalizer


def _hexagon_target_config() -> dict[str, object]:
    """构造 v1 Hexagon target 属性。

    TVM 已内置 ``hexagon`` kind；这里仅补齐 TileLang 需要稳定携带的 key 与
    Android host 语义。真正的 aarch64-linux-android host codegen/打包由后续
    C++/toolchain 层实现。
    """

    return {
        "kind": "hexagon",
        "keys": ["hexagon", "cpu"],
        "mtriple": "hexagon-unknown-none-elf",
        "host": {"kind": "llvm", "mtriple": "aarch64-linux-android"},
        "vtcm-capacity": 8 * 1024 * 1024,
    }


def normalize_hexagon_target(target: TargetLike) -> Target | None:
    """让 ``target=\"hexagon\"`` / ``{"kind":"hexagon"}`` 走同一条路。"""

    if isinstance(target, Target):
        return target if target.kind.name == "hexagon" else None
    if isinstance(target, dict):
        if target.get("kind") != "hexagon":
            return None
        merged = _hexagon_target_config()
        merged.update(target)
        return Target(merged)
    if target.strip() != "hexagon":
        return None
    return Target(_hexagon_target_config())


def target_is_hexagon(target: Target) -> bool:
    """轻量谓词，供后续 Python pass / codegen 分支复用。"""

    return target.kind.name == "hexagon"


register_target_normalizer("hexagon", normalize_hexagon_target, override=True)
