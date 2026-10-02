import importlib.util
from pathlib import Path
import pytest
import tilelang


def source(schedule="round_sync", depth=2):
    path = Path(__file__).resolve().parents[3] / "examples/hexagon/gemm/gemm_worker_pipeline.py"
    spec = importlib.util.spec_from_file_location("worker_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tilelang.engine.lower(module.kernel(schedule=schedule, depth=depth), target="hexagon",
                                 enable_host_codegen=False, enable_device_compile=False).kernel_source


def test_round_and_async_codegen():
    text = source()
    assert text.count("tl_wg_team_barrier(") == 1
    assert "tl_wg_group_wait(" not in text
    assert "tl_wg_group_publish(" not in text
    assert "hmx_store_after" in text
    assert ",6,3,0,3,0,0,0,12544,sync_bytes,initial,4,TL_WG_EFFECT_ROUND_SYNC,8}" in text
    old = source("async")
    assert "tl_wg_team_barrier(" not in old
    assert "tl_wg_group_wait(" in old and "tl_wg_group_publish(" in old
    assert ",6,3,3,3,12,0,0,12544,sync_bytes,initial,4,TL_WG_EFFECT_EXACTLY_ONCE,0}" in old


def test_depth_one_collision_rejected():
    with pytest.raises(ValueError, match="edge lifetime"):
        source(depth=1)
