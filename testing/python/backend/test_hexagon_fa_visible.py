"""Causal exp prefix covers every visible lane and clears only masked tails."""
import importlib.util
from pathlib import Path
import numpy as np
import tilelang
import pytest


def test_visible_chunks_all_rows():
    for qb in range(8):
        for kb in range(((qb+1)*128+255)//256):
            for row in range(128):
                visible=min(256,max(0,qb*128+row-kb*256+1))
                chunks=(visible+31)//32
                x=np.linspace(-3,0,256,dtype=np.float32)
                x[visible:]=-np.inf
                ref=np.exp(x)
                got=x.copy()
                got[:chunks*32]=np.exp(got[:chunks*32])
                got[chunks*32:]=0
                np.testing.assert_array_equal(got,ref)


@pytest.mark.parametrize("tile_state,fused_state",[(False,False),(True,False),(False,True)])
def test_persistent_visible_exp_lowering(tile_state,fused_state):
    path=Path(__file__).resolve().parents[3]/"examples/hexagon/fa/fa_persistent.py"
    spec=importlib.util.spec_from_file_location("fa_visible_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    src=tilelang.engine.lower(module.make_fa(tile_state,fused_state),target="hexagon",
        enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert "native_exp32" in src
    assert "hmx_mma_deep_f16" in src
    assert "fa_causal_persistent_rm_b1_h16_g4_s1024_d256_kernel" in src
    if fused_state:
        assert "transpose_packed_tile" not in src
        assert "state_t" not in src
        assert "hmx_mma_deep_f16<1>" in src
