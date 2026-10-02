"""Consumer-only staging must never rewrite address-taking producer operations."""
import importlib.util
from pathlib import Path

import pytest
import tilelang


def emit(stage, **kwargs):
    path = Path(__file__).resolve().parents[3] / "examples/hexagon/gemm/gemm_worker_pipeline.py"
    spec = importlib.util.spec_from_file_location("stage_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tilelang.engine.lower(
        module.kernel(stage_reads=stage, **kwargs), target="hexagon",
        enable_host_codegen=False, enable_device_compile=False).kernel_source


@pytest.mark.parametrize("tile", [(32, 32, 32), (64, 256, 1280)])
def test_consumer_only(tile):
    bm, bn, bk = tile
    shape = dict(M=bm*2, N=bn*2, K=bk*2, BM=bm, BN=bn, BK=bk)
    baseline = emit(False, **shape)
    staged = emit(True, cap_ddr=bm*bn*2*2, **shape)
    for primitive in ("hmx_store_after_f16", "gemm_pack_ah", "gemm_pack_wh", "gemm_scatter_release"):
        lines = lambda text: [s for s in text.splitlines() if primitive in s]
        assert lines(staged) == lines(baseline), primitive
    assert staged.count("for (int wg_copy") == 1
    assert "c[i_1] = (c[i_1] + ((float)wg_read_stage[" in staged
    assert "((unsigned char*)wg_ddr + 0 + " in staged
    assert f",0,{bm*bn*4}," in staged
    assert "if (!ddr || ((uintptr_t)ddr & 127u) || !vtcm)" in staged


def test_requires_backing_budget():
    with pytest.raises(ValueError, match="sufficient DDR budget"):
        emit(True)


@pytest.mark.parametrize("bm,bn", [(32, 32), (64, 256)])
def test_physical_copy_preserves_layout_bits(bm, bn):
    # Physical AH two-row interleave, covering both lanes of each row pair.
    # Staging is a bit copy, including NaN/inf/subnormal encodings, not a cast.
    import numpy as np
    physical = np.arange(bm*bn, dtype=np.uint16) * np.uint16(37)
    stage = np.empty_like(physical)
    for offset in range(0, physical.size, 64):
        stage[offset:offset+64] = physical[offset:offset+64]
    seen = set()
    for i in range(bm):
        for j in range(bn):
            index = ((i//32)*(bn//32)+j//32)*1024 + (i%32//2)*64 + (j%32)*2 + i%2
            seen.add(index)
            assert stage[index] == physical[index]
    assert len(seen) == bm*bn
