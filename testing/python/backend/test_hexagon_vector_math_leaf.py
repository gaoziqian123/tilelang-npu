"""Compile bounded math including rejection paths, without scalar libm."""
from pathlib import Path
import subprocess
import pytest


def test_bounded_math_object(tmp_path):
    clang = Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')
    if not clang.exists():
        pytest.skip('Hexagon compiler unavailable')
    root = Path(__file__).resolve().parents[3]
    source = tmp_path/'math.cpp'
    source.write_text('''#include <tl_templates/hexagon/vector_leaf.h>
extern "C" HVX_Vector exp_leaf(HVX_Vector x) { return tl::hvx_leaf::exp2_16_bounded(x); }
extern "C" HVX_Vector div_leaf(HVX_Vector x,HVX_Vector y) { return tl::hvx_leaf::div32_nr2(x,y); }
extern "C" HVX_Vector cast_leaf(HVX_Vector x) {
  auto a=tl::hvx_leaf::widen_evenodd(x,0);
  auto b=tl::hvx_leaf::widen_evenodd(x,1);
  return tl::hvx_leaf::narrow_linear(a,b);
}
extern "C" HVX_Vector broad_leaf(HVX_Vector x) {
  return tl::hvx_leaf::broadcast16(tl::hvx_leaf::broadcast32(x,31),63);
}
''')
    flags=[str(clang),'-mv79','-mhvx','-mhvx-length=128B','-O2','-std=c++17','-I'+str(root/'src')]
    for option,suffix in [('-c','o'),('-S','s')]:
        cmd=flags+[option,str(source),'-o',str(tmp_path/('math.'+suffix))]
        print('COMMAND:', ' '.join(cmd))
        result=subprocess.run(cmd,capture_output=True,text=True)
        print(result.stdout+result.stderr)
        assert result.returncode==0
    asm=(tmp_path/'math.s').read_text()
    for forbidden in ('exp2_hf_repair','exp2@','exp2(', '__divsf3', '__divdf3'):
        assert forbidden not in asm
    assert 'vmpy' in asm
