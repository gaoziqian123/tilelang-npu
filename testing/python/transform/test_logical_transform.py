"""Host-only checks for the experimental logical-transform path."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tilelang
from tilelang import language as T
from tvm import tirx


def test_transform_rejects_broadcast():
    a = tirx.decl_buffer((2, 64), "float16")
    b = tirx.decl_buffer((1, 64), "float16")
    with pytest.raises(Exception):
        T.transform(a, b)


def test_transform_access_regions():
    a = tirx.decl_buffer((2, 64), "float16")
    b = tirx.decl_buffer((2, 64), "float16")
    call = T.transform(a, b)
    assert call.op.name == "tl.tileop.transform"
    assert call.args[0].op.name == "tl.region"
    assert call.args[1].op.name == "tl.region"


def test_half_lane_network_reference():
    # Bitwise permutation preserves all half encodings, including payload NaNs,
    # signed zero and subnormals. This checks the lane-network mathematics,
    # not execution of the DSP instructions.
    bits = np.arange(65536, dtype=np.uint16).reshape(-1, 128)
    for parity in (0, 1):
        a, b = bits[:, :64], bits[:, 64:]
        deal_a = np.concatenate((a[:, ::2], a[:, 1::2]), axis=1)
        deal_b = np.concatenate((b[:, ::2], b[:, 1::2]), axis=1)
        actual = np.concatenate((deal_a[:, parity*32:(parity+1)*32],
                                 deal_b[:, parity*32:(parity+1)*32]), axis=1)
        np.testing.assert_array_equal(actual, bits[:, parity::2])


def test_transform_worker_codegen():
    path = Path(__file__).resolve().parents[3] / "examples/hexagon/gemm/gemm_prepacked_dma.py"
    spec = importlib.util.spec_from_file_location("transform_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tilelang.engine.lower(
        module.kernel(M=64, N=64, K=64, BM=64, BN=64, BK=32,
                      transform_output=True),
        target="hexagon", enable_host_codegen=False,
        enable_device_compile=False).kernel_source
    assert "Q6_Vh_vdeal_Vh" in source
    assert "tl::dma_copy_2d_wait(C," in source
    assert "float" not in source
    assert "__fp16* p2" in source
    assert "gemm_unpack" not in source
    assert "if (dma_status) return dma_status;" in source
