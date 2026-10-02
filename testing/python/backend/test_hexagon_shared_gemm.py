"""Structural contracts for native shared-output HMX lowering."""
import tilelang
import pytest
from tilelang import language as T


@pytest.mark.parametrize("n", [32, 128])
def test_shared_output_native_lifecycle(n):
    @T.prim_func
    def kernel(A: T.Tensor((32, 2560), "float16"),
               B: T.Tensor((n, 2560), "float16"),
               C: T.Tensor((32, n), "float16")):
        with T.Kernel(1, threads=4):
            a = T.alloc_shared((32, 2560), "float16")
            b = T.alloc_shared((n, 2560), "float16")
            c = T.alloc_shared((32, n), "float16")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, transpose_B=True, clear_accum=True)
            T.copy(c, C)
    src = tilelang.engine.lower(kernel, target="hexagon", enable_host_codegen=False,
                                enable_device_compile=False).kernel_source
    assert "hmx_init_scale" in src
    assert "hmx_store_after_f16" in src
    assert "hmx_acc_read_f16" not in src
    assert "hmx_mma_deep_f16<32>" in src
    assert "hmx_mma_deep_f16<16>" in src
    unpack = "gemm_unpack_ah_pair_strided<1>" if n == 128 else "gemm_unpack_ah_strided<1>"
    assert unpack in src
    assert "gemm_pack_wh_nt_strip<2560>" in src
    assert "gemm_scatter_release" in src
    assert src.index("gemm_scatter_release") < src.index("tl_hex_barrier();")
    assert src.count("tl_hex_barrier();") == 3
    assert src.index("hmx_init_scale") < src.index("hmx_clear_f16")
