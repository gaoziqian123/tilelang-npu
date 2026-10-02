"""Native reduction registration, plus execute emitted scalar C on the host.

The host ABI shim has no math implementation: the actual generated reduction
loops run unchanged. This is not a DSP/HVX numerical acceptance test.
"""
import ctypes
import importlib.util
from pathlib import Path
import subprocess
import numpy as np
import pytest
import tilelang
from tilelang import language as T
from tilelang.layout import Fragment
from tvm.ir.transform import PassContext


@pytest.mark.parametrize("kind", ["sum", "max", "min", "abssum", "absmax"])
@pytest.mark.parametrize("clear", [True, False])
def test_local_reduce_semantics(tmp_path, kind, clear):
    @T.prim_func
    def kernel(A: T.Tensor((2, 7), "float32"), B: T.Tensor((2,), "float32")):
        with T.Kernel(1, threads=1):
            a = T.alloc_local((2, 7), "float32")
            b = T.alloc_local((2,), "float32")
            for i in T.serial(2):
                for j in T.serial(7):
                    a[i, j] = A[i, j]
            for i in T.serial(2):
                b[i] = B[i]
            T.reduce(a, b, kind, dim=1, clear=clear)
            for i in T.serial(2):
                B[i] = b[i]
    src = tilelang.engine.lower(kernel, target="hexagon", enable_host_codegen=False,
                                enable_device_compile=False).kernel_source
    header = tmp_path / "tl_templates/hexagon/common.h"
    header.parent.mkdir(parents=True)
    header.write_text("#include <cstddef>\nnamespace tl { inline void hex_require(bool x) { if (!x) __builtin_trap(); } }\n"
                      "inline int tl_hex_worker_id(){return 0;}\ninline int tl_hex_job_id(){return 0;}\n"
                      "inline int tl_hex_num_jobs(){return 1;}\ninline int tl_hex_num_workers(){return 1;}\n")
    source = tmp_path / "reduce.cpp"
    source.write_text(src)
    so = tmp_path / "reduce.so"
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-I", str(tmp_path),
                    "-I", str(Path(__file__).resolve().parents[3] / "src"),
                    str(source), "-o", str(so)], check=True, capture_output=True)
    a = np.array([[-3, 1, 4, -2, 8, 0, -5], [7, -9, 2, 4, -1, 3, 6]], dtype="float32")
    b = np.array([2, -4], dtype="float32")
    data = abs(a) if kind.startswith("abs") else a
    fn = np.sum if kind in ("sum", "abssum") else np.max if kind in ("max", "absmax") else np.min
    expected = fn(data, axis=1)
    if not clear:
        expected = expected + b if kind in ("sum", "abssum") else (np.maximum if kind in ("max", "absmax") else np.minimum)(b, expected)
    lib = ctypes.CDLL(str(so))
    call = lib.kernel_kernel
    call.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    call(a.ctypes.data, b.ctypes.data)
    np.testing.assert_array_equal(b, expected)


@pytest.mark.parametrize("rows,cols,workers", [(32, 128, 4), (128, 256, 4), (32, 128, 2)])
@pytest.mark.parametrize("kind", ["max", "sum"])
@pytest.mark.parametrize("clear", [True, False])
def test_row_owned_fragment_reduce(tmp_path, rows, cols, workers, kind, clear):
    """Execute emitted loops, with real input data, once for every worker.

    The ABI shim only selects a worker; it implements no reduction or math.
    Checking the output after EACH invocation detects cross-worker writes as
    well as missing rows. Integer-valued floats make exact sum checks possible.
    """
    @T.prim_func
    def kernel(A: T.Tensor((rows, cols), "float32"), B: T.Tensor((rows,), "float32")):
        with T.Kernel(1, threads=workers):
            a = T.alloc_fragment((rows, cols), "float32")
            b = T.alloc_fragment((rows,), "float32")
            T.annotate_layout({a: Fragment((rows, cols),
                forward_thread_fn=lambda i, j: i % workers,
                forward_index_fn=lambda i, j: (i // workers) * cols + j)})
            for i, j in T.Parallel(rows, cols):
                a[i, j] = A[i, j]
            for i in T.Parallel(rows):
                b[i] = B[i]
            T.reduce(a, b, kind, dim=1, clear=clear)
            for i in T.Parallel(rows):
                B[i] = b[i]
    with PassContext(config={"tirx.disable_vectorize": True}):
        src = tilelang.engine.lower(kernel, target="hexagon", enable_host_codegen=False,
            enable_device_compile=False).kernel_source
    assert "AllReduce" not in src
    assert "tl_hex_barrier" not in src
    header = tmp_path / "tl_templates/hexagon/common.h"
    header.parent.mkdir(parents=True)
    header.write_text("#include <cstddef>\nstatic int worker;\n"
        'extern "C" void select_worker(int w){worker=w;}\n'
        "namespace tl { inline void hex_require(bool x){if(!x)__builtin_trap();} }\n"
        "inline int tl_hex_worker_id(){return worker;}\n"
        "inline int tl_hex_job_id(){return 0;}\ninline int tl_hex_num_jobs(){return 1;}\n"
        f"inline int tl_hex_num_workers(){{return {workers};}}\n")
    source, so = tmp_path / "rows.cpp", tmp_path / "rows.so"
    source.write_text(src)
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-I", str(tmp_path),
                    "-I", str(Path(__file__).resolve().parents[3] / "src"),
                    str(source), "-o", str(so)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(so))
    lib.kernel_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.select_worker.argtypes = [ctypes.c_int]
    rng = np.random.default_rng(984)
    for trial in range(3):
        a = rng.integers(-32, 33, (rows, cols)).astype("float32")
        # Maxima occur at different columns, including both ends of the row;
        # no constant-input or worker-0-only reduction can satisfy this test.
        a[np.arange(rows), (np.arange(rows) * 31 + trial * (cols - 1)) % cols] = np.arange(rows) + 64
        b = rng.integers(-256, 257, rows).astype("float32")
        expected = b.copy()
        reduced = (np.max if kind == "max" else np.sum)(a, axis=1)
        if not clear:
            reduced = np.maximum(reduced, b) if kind == "max" else reduced + b
        for worker in reversed(range(workers)):
            lib.select_worker(worker)
            lib.kernel_kernel(a.ctypes.data, b.ctypes.data)
            expected[worker::workers] = reduced[worker::workers]
            np.testing.assert_array_equal(b, expected)


@pytest.mark.parametrize("kind,op", [("max", "MaxOp"), ("sum", "SumOp")])
def test_column_owned_fragment_keeps_collective(kind, op):
    @T.prim_func
    def kernel(A: T.Tensor((8, 128), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=4):
            a = T.alloc_fragment((8, 128), "float32")
            b = T.alloc_fragment((8,), "float32")
            T.annotate_layout({a: Fragment((8, 128),
                forward_thread_fn=lambda i, j: j % 4,
                forward_index_fn=lambda i, j: i * 32 + j // 4)})
            for i, j in T.Parallel(8, 128):
                a[i, j] = A[i, j]
            T.reduce(a, b, kind, dim=1, clear=True)
            for i in T.Parallel(8):
                B[i] = b[i]
    src = tilelang.engine.lower(kernel, target="hexagon", enable_host_codegen=False,
        enable_device_compile=False).kernel_source
    assert f"tl::AllReduce<tl::{op}" in src


@pytest.mark.parametrize("block_q,block_k", [(32, 128), (128, 256)])
def test_fa_full_lower_reproducible(block_q, block_k):
    path = Path(__file__).resolve().parents[3] / "examples/hexagon/fa/fa.py"
    spec = importlib.util.spec_from_file_location("native_fa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def emit():
        return tilelang.engine.lower(module.make_fa(block_q, block_k), target="hexagon",
            enable_host_codegen=False, enable_device_compile=False).kernel_source
    src = emit()
    assert src == emit()
    # HMX assigns whole rows to workers: both reductions must be local, but
    # must still consume the complete K tile and preserve the running maximum.
    assert "AllReduce" not in src
    assert f"float scores[{block_q * block_k // 4}]" in src
    assert f"float maximum[{block_q // 4}]" in src
    assert f"float tile_sum[{block_q // 4}]" in src
    assert "tl::row_reduce_f32<true>(&scores[" in src
    assert "tl::row_reduce_f32<false>(&scores[" in src
    assert f", {block_k}, maximum_clear[" in src
    assert f", {block_k}, tile_sum[" in src
    assert "maximum_clear[" in src and "maximum[i_3]" in src
    assert "tile_sum[i_5] = tl::row_reduce_f32<false>" in src
    assert ("expf(" in src or "tl::native_exp32(" in src) and "INFINITY" in src
    assert "hmx_mma_deep" in src
    assert "fa_online" not in src and "fa_std" not in src


@pytest.mark.parametrize("maximum", [False, True])
def test_row_leaf_boundaries(tmp_path, maximum):
    """Execute the same tree/tail algorithm on host, including ordered fallback."""
    source, so = tmp_path / "leaf.cpp", tmp_path / "leaf.so"
    source.write_text('#include <tl_templates/hexagon/row_reduce.h>\n'
        'extern "C" float run(const float* p,int n,float seed){return '
        f'tl::row_reduce_f32<{str(maximum).lower()}>(p,n,seed);}}\n')
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-I",
        str(Path(__file__).resolve().parents[3] / "src"), str(source),
        "-o", str(so)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(so))
    lib.run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
    lib.run.restype = ctypes.c_float
    for n in [0, 1, 7, 31, 32, 33, 63, 65, 128, 257]:
        for seed in [0., -17., -np.inf, np.inf, np.nan]:
            for exceptional in [None, np.nan, np.inf, -np.inf]:
                a = (np.arange(n, dtype=np.float32) % 17 - 8).copy()
                if n and exceptional is not None:
                    a[n // 2] = exceptional
                ref = np.float32(seed)
                with np.errstate(invalid="ignore"):
                    for value in a:
                        ref = (ref if ref > value else value) if maximum else np.float32(ref + value)
                got = lib.run(a.ctypes.data, n, seed)
                if np.isnan(ref):
                    assert np.isnan(got)
                else:
                    assert got == ref, (maximum, n, seed, exceptional, got, ref)


def test_persistent_bounded_consumers():
    path = Path(__file__).resolve().parents[3] / "examples/hexagon/fa/fa_persistent.py"
    spec = importlib.util.spec_from_file_location("persistent_fa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    src = tilelang.engine.lower(module.make_fa(), target="hexagon",
        enable_host_codegen=False, enable_device_compile=False).kernel_source
    assert "float scores[" not in src and "float partial[" not in src
    assert "p_rm" not in src
    assert "float rowf[512]" in src and "float out[8192]" in src
    assert src.count("tl::gemm_unpack_ah_rows(") == 2
    # copy_rows now stages Q through a bounded (2,256) local buffer and
    # packs directly into AH. This is outside kb; the original probability
    # pack remains inside rp/kb, with the same two-row work and ownership.
    packs = [line.strip() for line in src.splitlines() if 'tl::gemm_pack_ah_rows(' in line]
    assert packs == [
        'tl::gemm_pack_ah_rows((&(((__fp16*)q)[0])), (&(staging_6[0])), 256, ((vi_6 * 8) + (tx * 2)), 2);',
        'tl::gemm_pack_ah_rows((&(((__fp16*)p)[0])), (&(rowh[0])), 256, ((rp * 8) + (tx * 2)), 2);',
    ]
    assert '__fp16 staging_6[512];' in src and '__fp16 rowh[512];' in src
    assert 'for (int32_t vi_6 = 0; vi_6 < 16; ++vi_6)' in src
    assert 'for (int32_t rp = 0; rp < 16; ++rp)' in src
    assert src.index(packs[0]) < src.index('for (int32_t kb =') < src.index(packs[1])
    assert "tl_hex_num_jobs() == 4" in src
    assert "__builtin_convertvector" in src
