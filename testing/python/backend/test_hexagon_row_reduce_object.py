"""Compile the actual HVX leaf; host emulation cannot test its load schedule."""
import os
from pathlib import Path
import re
import subprocess

import pytest


def test_row_reduce_single_load_loop(tmp_path):
    tools = Path(os.environ.get("HEXAGON_TOOLS", "/root/hexagon-deps/HEXAGON_TOOLS/Tools")) / "bin"
    cc = tools / "hexagon-clang++"
    if not cc.exists():
        pytest.skip("Hexagon compiler unavailable")
    src = tmp_path / "reduce.cpp"
    src.write_text('#include <tl_templates/hexagon/row_reduce.h>\n'
                   'extern "C" float reduce_sum(const float* p,int n,float s){'
                   'return tl::row_reduce_f32<false>(p,n,s);}\n')
    obj = tmp_path / "reduce.o"
    subprocess.run([str(cc), "-mv79", "-mhvx", "-mhvx-length=128B", "-O2",
                    "-std=c++17", "-I", str(Path(__file__).resolve().parents[3] / "src"),
                    "-c", str(src), "-o", str(obj)], check=True, capture_output=True)
    asm = subprocess.check_output([str(tools / "hexagon-llvm-objdump"), "-dr", str(obj)], text=True)
    # Dynamic extent prevents constant-trip full unrolling. Classification and
    # arithmetic must share one vector input load, not two independent passes.
    # LLVM software-pipelines the first load into a peeled prologue. Locate
    # the hardware loop body, rather than confusing static loads with passes.
    loops = re.findall(r"loop0\(0x([0-9a-f]+),", asm)
    vector_loops = []
    for address in loops:
        start = re.search(rf"^\s*{address}:.*$", asm, re.M)
        assert start, asm
        body = asm[start.start():].split(":endloop0", 1)[0]
        if re.search(r"v\d+\s*=\s*vmem[u]?\(", body):
            vector_loops.append(body)
    assert len(vector_loops) == 1, asm
    body = vector_loops[0]
    assert len(re.findall(r"v\d+\s*=\s*vmem[u]?\(", body)) == 1, asm
    assert "vcmp.gt" in body and "vadd(" in body, asm
