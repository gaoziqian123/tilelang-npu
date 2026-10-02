"""Explicit-plan IR gates plus unchanged host-compiled numerical regressions."""
import importlib.util
import ctypes
from pathlib import Path
import subprocess

import numpy as np

import pytest
import tilelang
from tilelang import language as T
from tvm import instrument
from tvm.ir.transform import PassContext

import test_hexagon_reduce as reference_tests


@instrument.pass_instrument
class Capture:
    def __init__(self):
        self.planned = []

    def run_after_pass(self, mod, info):
        if info.name == "tl.PlanLocalRowReduce":
            self.planned.append(str(mod))


@pytest.mark.parametrize("kind", ["sum", "max", "min", "abssum", "absmax"])
@pytest.mark.parametrize("clear", [False, True])
def test_planned_host_compiled(tmp_path, kind, clear):
    capture = Capture()
    with PassContext(config={"tl.hexagon.plan_local_row_reduce": True}, instruments=[capture]):
        reference_tests.test_local_reduce_semantics(tmp_path, kind, clear)
    assert len(capture.planned) == 1
    assert ("hexagon.local_row_reduce" in capture.planned[0]) == (kind in ("sum", "max"))
    if kind in ("sum", "max"):
        assert "seed_after_tree:ordered_tail:stored_ordered_replay" in capture.planned[0]


@pytest.mark.parametrize("cols", [1, 31, 32, 33, 65, 257])
@pytest.mark.parametrize("stride", [1, 2])
def test_contiguous_only(cols, stride):
    @T.prim_func
    def kernel(A: T.Tensor((cols * stride,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            a = T.alloc_local((cols * stride,), "float32")
            b = T.alloc_local((1,), "float32")
            for j in T.serial(cols * stride):
                a[j] = A[j]
            b[0] = B[0]
            for j in T.serial(cols):
                b[0] = b[0] + a[j * stride]
            B[0] = b[0]
    capture = Capture()
    with PassContext(config={"tl.hexagon.plan_local_row_reduce": True}, instruments=[capture]):
        tilelang.engine.lower(kernel, target="hexagon", enable_host_codegen=False,
                              enable_device_compile=False)
    # A unit-trip loop may simplify away before planning.
    assert ("hexagon.local_row_reduce" in capture.planned[0]) == (stride == 1 and cols > 1)


def test_real_fa_ir():
    path = Path(__file__).resolve().parents[3] / "examples/hexagon/fa/fa.py"
    spec = importlib.util.spec_from_file_location("planned_fa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    capture = Capture()
    with PassContext(config={"tl.hexagon.plan_local_row_reduce": True}, instruments=[capture]):
        src = tilelang.engine.lower(module.make_fa(), target="hexagon",
            enable_host_codegen=False, enable_device_compile=False).kernel_source
    assert capture.planned[0].count("hexagon.local_row_reduce") == 2
    assert "tl::row_reduce_f32<true>" in src
    assert "tl::row_reduce_f32<false>" in src


@pytest.mark.parametrize("maximum", [False, True])
def test_leaf_exceptions_tail_seed(tmp_path, maximum):
    reference_tests.test_row_leaf_boundaries(tmp_path, maximum)


@pytest.mark.parametrize("maximum", [False, True])
def test_exact_tree_signed_zero(tmp_path, maximum):
    source, so = tmp_path / "tree.cpp", tmp_path / "tree.so"
    source.write_text('#include <tl_templates/hexagon/row_reduce.h>\n'
        'extern "C" float run(const float* p,int n,float s){return '
        f'tl::row_reduce_f32<{str(maximum).lower()}>(p,n,s);}}\n')
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-I",
        str(Path(__file__).resolve().parents[3] / "src"), str(source),
        "-o", str(so)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(so))
    lib.run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
    lib.run.restype = ctypes.c_float
    def combine(a, b):
        return (a if a > b else b) if maximum else np.float32(a + b)
    rng = np.random.default_rng(153)
    for n in [0, 1, 31, 32, 33, 63, 65, 257]:
        for seed in [np.float32(-0.), np.float32(0.), np.float32(1e10)]:
            for zeros in [False, True]:
                a = (rng.normal(size=n) * 100).astype("float32")
                if zeros:
                    a[:] = -0.
                    a[::2] = 0.
                lanes = np.full(32, -np.inf if maximum else 0., dtype="float32")
                end = n // 32 * 32
                for i in range(0, end, 32):
                    lanes = np.array([combine(x, y) for x, y in zip(lanes, a[i:i+32])], dtype="float32")
                for shift in [16, 8, 4, 2, 1]:
                    lanes = np.array([combine(lanes[j], lanes[(j+shift)%32]) for j in range(32)], dtype="float32")
                ref = combine(seed, lanes[0])
                for value in a[end:]:
                    ref = combine(ref, value)
                got = np.float32(lib.run(a.ctypes.data, n, seed))
                assert got.view("uint32") == np.float32(ref).view("uint32")


def test_persistent_fa_ir_and_hexagon_object(tmp_path):
    path = Path(__file__).resolve().parents[3] / "examples/hexagon/fa/fa_persistent.py"
    spec = importlib.util.spec_from_file_location("planned_persistent_fa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    capture = Capture()
    with PassContext(config={"tl.hexagon.plan_local_row_reduce": True}, instruments=[capture]):
        src = tilelang.engine.lower(module.make_fa(), target="hexagon",
            enable_host_codegen=False, enable_device_compile=False).kernel_source
    ir = capture.planned[0]
    assert "hexagon.local_row_reduce" in ir
    assert "tl::profile_region" in ir  # opaque boundaries are NOT removed
    assert "stored_ordered_replay" in ir
    cc = Path("/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++")
    if not cc.exists():
        pytest.skip("Hexagon compiler unavailable")
    source = tmp_path / "planned.cpp"
    source.write_text(src)
    subprocess.run([str(cc), "-mv79", "-mhvx", "-mhmx", "-mhvx-length=128B",
        "-O2", "-std=c++17", "-I", str(Path(__file__).resolve().parents[3] / "src"),
        "-c", str(source), "-o", str(tmp_path / "planned.o")],
        check=True, capture_output=True)
