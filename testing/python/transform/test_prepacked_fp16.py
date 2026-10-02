import importlib.util
from pathlib import Path
import tilelang


def test_fp16_accumulator_codegen():
    path=Path(__file__).resolve().parents[3]/"examples/hexagon/gemm/gemm_prepacked_dma.py"
    spec=importlib.util.spec_from_file_location("prepacked_example",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text=tilelang.engine.lower(module.kernel(),target="hexagon",
        enable_host_codegen=False,enable_device_compile=False).kernel_source
    assert "__fp16 c[" in text
    assert "__fp16* p2" in text
    assert "float" not in text
    assert "tl::hvx_vec<half_t>" in text and "tl::hvx_add(x, y)" in text
    assert "hmx_store_after_f16" in text
    assert "dma_status" in text and "gemm_pack_" not in text
