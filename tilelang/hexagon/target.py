from __future__ import annotations

from tvm.target import Target
from tilelang.backend.target import TargetLike, register_target_normalizer


_REMOVED_TARGETS = {"hexagon_legacy", "hexagon_v2"}


def target_is_hexagon(target: Target) -> bool:
    return target.kind.name == "hexagon" and target.tag not in _REMOVED_TARGETS and not (_REMOVED_TARGETS & set(target.keys))


def normalize_hexagon_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        identifiers = {target.kind.name, target.tag, *target.keys}
    elif isinstance(target, dict):
        keys = target.get("keys", [])
        identifiers = {target.get("kind"), target.get("tag"), *(keys.split(",") if isinstance(keys, str) else keys)}
    else:
        identifiers = set(target.replace("=", " ").replace(",", " ").split())
    if identifiers & _REMOVED_TARGETS:
        raise ValueError("Removed Hexagon legacy/v2 target; use the standard hexagon backend explicitly")
    if isinstance(target, Target):
        if not target_is_hexagon(target):
            return None
        config = dict(target.export())
    elif isinstance(target, dict):
        if target.get("kind") != "hexagon":
            return None
        config = dict(target)
    elif target.strip() and target.strip().split()[0] == "hexagon":
        config = dict(Target(target).export())
    else:
        return None
    config.setdefault("mcpu", "hexagonv79")
    config.setdefault("mtriple", "hexagon-unknown-none-elf")
    config.setdefault("mattr", ["+hvxv79", "+hvx-length128b", "+hmx"])
    config.setdefault("vtcm-capacity", 8 * 1024 * 1024)
    config.setdefault("host", "llvm")
    return Target(config)


# Cross compilation is explicit: do not auto-detect a DSP on the build server.
register_target_normalizer("hexagon", normalize_hexagon_target, override=True)
