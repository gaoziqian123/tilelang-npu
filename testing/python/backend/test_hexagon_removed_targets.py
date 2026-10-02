"""Deletion contract: removed backends must never alias the standard backend."""

import pytest

from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.hexagon.target import target_is_hexagon


@pytest.mark.parametrize("name", ["hexagon_legacy", "hexagon_v2"])
@pytest.mark.parametrize("form", ["string", "dict_kind", "dict_tag", "dict_keys", "target_tag", "target_keys", "string_tag"])
def test_removed_hexagon_targets_rejected(name, form):
    targets = {
        "string": name,
        "string_tag": f"hexagon -tag={name}",
        "dict_kind": {"kind": name},
        "dict_tag": {"kind": "hexagon", "tag": name},
        "dict_keys": {"kind": "hexagon", "keys": [name]},
        "target_tag": tvm.target.Target({"kind": "hexagon", "tag": name}),
        "target_keys": tvm.target.Target({"kind": "hexagon", "keys": [name]}),
    }
    target = targets[form]
    if isinstance(target, tvm.target.Target):
        assert not target_is_hexagon(target)
    with pytest.raises(ValueError, match="Removed Hexagon legacy/v2 target"):
        determine_target(target, return_object=True)
