"""Target attributes and normalization for the Hexagon-v2 prototype."""

from __future__ import annotations

from dataclasses import dataclass

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_normalizer


@dataclass(frozen=True)
class HexagonV2Target:
    vtcm_capacity: int = 8 * 1024 * 1024
    hmx_chain_max: int = 32
    worker_max: int = 6
    dcfetch_distance: int = 8192


DEFAULT_TARGET = HexagonV2Target()


def _hexagon_v2_target_config() -> dict[str, object]:
    """Return a TVM target carrying the v2 discriminator.

    The vendored TVM build does not expose a Python-side TargetKind registrar, so
    this phase uses the built-in ``hexagon`` kind with tag ``hexagon_v2``.  The
    TileLang backend registry still resolves it to the separate ``hexagon_v2``
    manifest via mutually-exclusive ``supports_target`` predicates; callers use
    ``target=\"hexagon_v2\"`` and never share the v1 backend path.
    """

    return {
        "kind": "hexagon",
        "tag": "hexagon_v2",
        "keys": ["hexagon_v2", "hexagon", "cpu"],
        "mtriple": "hexagon-unknown-none-elf",
        "host": {"kind": "llvm", "mtriple": "aarch64-linux-android"},
        "vtcm-capacity": DEFAULT_TARGET.vtcm_capacity,
    }


def normalize_hexagon_v2_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        if target.kind.name == "hexagon" and target.tag == "hexagon_v2":
            return target
        return None
    if isinstance(target, dict):
        kind = target.get("kind")
        if kind not in ("hexagon_v2", "hexagon"):
            return None
        if kind == "hexagon" and target.get("tag") != "hexagon_v2":
            return None
        merged = _hexagon_v2_target_config()
        merged.update(target)
        merged["kind"] = "hexagon"
        merged["tag"] = "hexagon_v2"
        return Target(merged)
    if target.strip() != "hexagon_v2":
        return None
    return Target(_hexagon_v2_target_config())


def target_is_hexagon_v2(target: Target) -> bool:
    return target.kind.name == "hexagon" and target.tag == "hexagon_v2"


register_target_normalizer("hexagon_v2", normalize_hexagon_v2_target, override=True)
