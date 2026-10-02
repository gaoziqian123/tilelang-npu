"""Physical leaf contracts and actual target compilation (no device claims)."""
from pathlib import Path
import subprocess
import pytest
import tilelang
from tvm import tirx as tir
from tilelang.hexagon.language.vector import leaf, permute, permute2


def test_maps():
    x = tir.Var("x", "uint8x128")
    y = tir.Var("y", "uint8x128")
    for size in (1, 2, 4):
        n = 128 // size
        for shift in range(n):
            call = permute(x, [(i+shift) % n for i in range(n)], lane_bytes=size)
            assert call.args[0].value == "rotate"
            assert call.args[2].value == shift * size
        for shift in range(n+1):
            call = permute2(x, y, range(shift, shift+n), lane_bytes=size)
            assert call.args[0].value == ("rotate" if shift == n else "align")
            assert call.args[2].value == (0 if shift == n else shift * size)
    assert permute(x, [i//2+(i%2)*32 for i in range(64)], lane_bytes=2).args[0].value == "interleave16"
    assert permute(x, [2*(i%32)+i//32 for i in range(64)], lane_bytes=2).args[0].value == "deal16"


def test_rejections():
    x = tir.Var("x", "uint8x128")
    for name in ("masked_load", "div", "fma", "exp_f32", "unknown"):
        with pytest.raises(ValueError, match="unsupported"):
            leaf(name, x)
    with pytest.raises(ValueError, match="explicit mode"):
        leaf("add32", x, x)
    with pytest.raises(ValueError, match="byte shift"):
        leaf("rotate", x, immediate=128)
    with pytest.raises(ValueError, match="bit views"):
        leaf("rotate", tir.Var("f", "float32x32"))
    with pytest.raises(ValueError, match="network"):
        permute(x, reversed(range(32)), lane_bytes=4)
    with pytest.raises(ValueError, match="network"):
        permute2(x, x, [0]*32, lane_bytes=4)


def test_target_object_and_asm(tmp_path):
    compiler = Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')
    if not compiler.exists():
        pytest.skip("Hexagon compiler unavailable")
    root = Path(__file__).resolve().parents[3]
    names = {"rotate": "a,4", "align": "a,b,4", "interleave16": "a", "deal16": "a",
             "select": "a,b,c", "widen_linear": "a,0", "narrow_evenodd": "a,b", "rcp32_nr2": "a"}
    names.update({op+bits: "a,b" for op in ("add", "sub", "mul", "min", "max") for bits in ("16", "32")})
    names.update({op: "a" for op in ("abs16", "neg16", "abs32", "neg32", "sum32_asc", "max32_asc")})
    source = '#include <tl_templates/hexagon/vector_leaf.h>\n'
    for name, args in names.items():
        source += f'extern "C" HVX_Vector probe_{name}(HVX_Vector a,HVX_Vector b,HVX_Vector c) {{ return tl::hvx_leaf::{name}({args}); }}\n'
    cpp = tmp_path / 'leaf.cpp'
    cpp.write_text(source)
    flags = [str(compiler), '-mv79', '-mhvx', '-mhvx-length=128B', '-O2', '-std=c++17', '-I'+str(root/'src')]
    for kind, suffix in (('-c', 'o'), ('-S', 's')):
        cmd = flags + [kind, str(cpp), '-o', str(tmp_path / ('leaf.'+suffix))]
        print('COMMAND:', ' '.join(cmd))
        result = subprocess.run(cmd, text=True, capture_output=True)
        print(result.stdout, result.stderr, sep='')
        assert result.returncode == 0
    asm = (tmp_path/'leaf.s').read_text()
    assert 'vror' in asm
    assert 'vadd' in asm
    assert 'expf' not in asm and 'exp2' not in asm
    assert 'divsf' not in asm
