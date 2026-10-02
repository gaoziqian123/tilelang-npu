"""Structural checks for the prepacked resident GEMM schedule (host only)."""
import importlib.util
from pathlib import Path

import pytest
import tilelang


def example():
    path = Path(__file__).resolve().parents[3] / "examples/hexagon/gemm/gemm_prepacked_dma.py"
    spec = importlib.util.spec_from_file_location("resident_gemm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bm,bn", [(32, 64), (64, 64), (64, 128)])
def test_resident_dma_guard_and_owner(bm, bn):
    result = tilelang.engine.lower(example().kernel(
        M=128, N=256, K=128, BM=bm, BN=bn, BK=64,
        schedule="async", reuse_b=True, load=1, merge_edges=True,
        transform_output=True), target="hexagon",
        enable_host_codegen=False, enable_device_compile=False)
    source = result.kernel_source
    # B DMA must be inside the first-M guard, not merely labelled resident
    # in a manifest. A remains outside that guard for every output tile.
    guard = source.index("if (m == 0) {")
    b_copy = source.index(", 0, B,", guard)
    assert source.index("}", guard) > b_copy
    assert source.index(", 0, A,") < guard
    assert "tl_wg_team_barrier" not in source
    assert "if (initial % 2u) return TL_WG_INVALID;" in source
    # Physical worker zero remains the only HMX owner.
    owner = source.index("->physical_id < 1")
    assert owner < source.index("tl::hmx_clear_f16")
    assert "hexkl" not in source
    assert "tl::hvx_add" in source


@pytest.mark.parametrize("bk", [32, 128])
def test_reuse_rejects_incomplete_resident_coverage(bk):
    with pytest.raises(ValueError, match="two K panels"):
        example().kernel(M=128, N=256, K=128, BM=64, BN=64,
                         BK=bk, reuse_b=True)
