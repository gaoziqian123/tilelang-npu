"""Generation gates, not substitutes for HF DSP/oracle validation."""
import importlib.util
from pathlib import Path
import pytest
import tilelang

spec = importlib.util.spec_from_file_location(
    'hf_schedule', Path(__file__).resolve().parents[3] /
    'examples/hexagon/fa/fa_tile_schedule.py')
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


@pytest.mark.parametrize('bm,bn,workers', [(32,32,1),(128,512,3),(256,256,6)])
@pytest.mark.parametrize('round_sum', [False, True])
def test_explicit_variant_generates(bm,bn,workers,round_sum):
    func = schedule.make_fa(bm,bn,workers,explicit_hf=True,
                            round_local_sum=round_sum)
    src = tilelang.engine.lower(func,target='hexagon',
        enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert 'tl::exp2_hf(' in src
    assert 'tl::sub_hf(' in src
    assert 'hf_math.h' in src
    assert 'hmx_mma_deep' in src  # Diagnostic keeps known-good GEMM route.


def test_default_does_not_select_hf():
    src = tilelang.engine.lower(schedule.make_fa(),target='hexagon',
        enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert 'tl::exp2_hf(' not in src
    assert 'tl::sub_hf(' not in src


def test_sum_rounding_requires_variant():
    with pytest.raises(ValueError):
        schedule.make_fa(round_local_sum=True)
