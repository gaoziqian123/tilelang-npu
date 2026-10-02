"""Host-only IR-builder contract tests (no target compilation/device)."""

import subprocess
from pathlib import Path

import pytest
import tilelang.language as T
from tvm import tirx
from tvm.script.ir_builder import IRBuilder


def build(depth=2, roles=(("arbitrary", "hvx", 3), ("dot", "hmx", 1), ("sink", "hvx", 2))):
    with IRBuilder() as builder:
        with T.Pipelined(5, num_stages=depth):
            for name, engine, workers in roles:
                with T.pipeline_stage(name, engine=engine, workers=workers):
                    T.evaluate(0)
    return builder.get()


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_depth_and_arbitrary_names(depth):
    loop = build(depth)
    assert int(loop.annotations["tl.workergroup_depth"]) == depth
    assert "num_stages" not in loop.annotations
    attrs = []
    tirx.stmt_functor.post_order_visit(loop, lambda n: attrs.append(n) if isinstance(n, tirx.AttrStmt) else None)
    assert [a.node["name"] for a in attrs] == ["arbitrary", "dot", "sink"]
    assert [int(a.node["workers"]) for a in attrs] == [3, 1, 2]


@pytest.mark.parametrize("roles", [(("only", "hvx", 6),), (("x", "hvx", 1), ("y", "hvx", 5))])
def test_role_shapes(roles):
    assert "tl.workergroup_depth" in build(roles=roles).annotations


def test_weights_are_ir_not_python_scheduling():
    with IRBuilder() as builder:
        with T.Pipelined(1, num_stages=3):
            with T.pipeline_stage("weighted", engine="hvx", weight=7):
                T.evaluate(0)
    assert int(builder.get().body.node["weight"]) == 7


@pytest.mark.parametrize("kwargs", [
    {"engine": "bogus", "workers": 1},
    {"engine": "hmx", "workers": 2},
    {"engine": "hmx", "weight": 1},
    {"engine": "hvx", "workers": -1},
    {"engine": "hvx", "workers": 0},
    {"engine": "hvx", "workers": True},
    {"engine": "hvx", "workers": 1, "weight": 1},
    {"engine": "hvx"},
])
def test_invalid_role_arguments(kwargs):
    with pytest.raises((ValueError, TypeError)):
        T.pipeline_stage("not_semantic", **kwargs)


@pytest.mark.parametrize("depth", [0, -1])
def test_invalid_depth(depth):
    with pytest.raises(ValueError, match="positive num_stages"):
        build(depth)


def test_existing_software_pipeline_unchanged():
    with IRBuilder() as builder:
        with T.Pipelined(5, num_stages=2):
            T.evaluate(0)
    loop = builder.get()
    assert int(loop.annotations["num_stages"]) == 2
    assert "tl.workergroup_depth" not in loop.annotations


@pytest.mark.parametrize("compiler,standard", [("cc", "c11"), ("c++", "c++17")])
def test_abi_header(compiler, standard, tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = '#include "src/tl_templates/hexagon/workergroup_abi.h"\nint main(void) { return TL_WG_ABI_VERSION != 3 || sizeof(tl_wg_plan_desc) != 80 || offsetof(tl_wg_plan_desc, round_count) != 76; }\n'
    result = subprocess.run([compiler, f"-std={standard}", "-Wall", "-Werror", "-I", str(root),
                             "-x", "c" if compiler == "cc" else "c++", "-", "-o", str(tmp_path / "abi")],
                            input=source, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(tmp_path / "abi")], check=True)
