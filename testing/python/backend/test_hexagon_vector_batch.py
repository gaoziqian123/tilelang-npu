from pathlib import Path
import subprocess
import pytest
import tilelang
from tvm import tirx as tir
from tilelang.hexagon.language.vector import load, store, splat_bits, map_chunks, narrow_chunks, leaf
from tilelang.hexagon.vector_verify import verify_vector_required


def test_memory_and_chunks():
    a=tir.decl_buffer((128,),"float16",name="a")
    x=load(a,64,valid_lanes=3)
    wide=map_chunks("widen",(x,),mode="strict_exact_v1")
    assert len(wide)==2 and str(wide[0].dtype)=="float32x32"
    h=narrow_chunks(wide)
    stmt=tir.Evaluate(store(a,h[0],64,valid_lanes=3))
    f=tir.PrimFunc([a.data],stmt).with_attr("tl.vector_required",1)
    verify_vector_required(f)
    for offset in (-1,1,128):
        with pytest.raises(ValueError,match="128B"):
            load(a,offset)
    with pytest.raises(ValueError,match="128B"):
        load(tir.decl_buffer((3,),"float16"),valid_lanes=3)
    with pytest.raises(ValueError,match="unsupported"):
        leaf("fma32", *([tir.Var("x","uint8x128")]*3),mode="native_target_v1")


def test_scalar_rejection():
    a=tir.decl_buffer((32,),"float32")
    f=tir.PrimFunc([a.data],tir.Evaluate(a[0]+1)).with_attr("tl.vector_required",1)
    with pytest.raises(ValueError,match="vector_required"):
        verify_vector_required(f)


def test_new_leaves_target(tmp_path):
    cc=Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')
    if not cc.exists(): pytest.skip('Hexagon compiler unavailable')
    root=Path(__file__).resolve().parents[3]
    src=tmp_path/'batch.cpp'
    src.write_text('''#include <tl_templates/hexagon/vector_leaf.h>
extern "C" void batch(void*d,const void*s,HVX_Vector a,HVX_Vector b) {
  auto x=tl::hvx_leaf::load(s,6,a);
  auto q=tl::hvx_leaf::compare<16>(x,b,3,0);
  auto p=tl::hvx_leaf::compare<32>(a,b,1,1);
  auto y=tl::hvx_leaf::fma16(x,q,p);
  tl::hvx_leaf::store(d,y,6);
}
''')
    for flag,suffix in [('-c','o'),('-S','s')]:
        cmd=[str(cc),'-mv79','-mhvx','-mhvx-length=128B','-O2','-std=c++17','-I'+str(root/'src'),flag,str(src),'-o',str(tmp_path/('batch.'+suffix))]
        print('COMMAND:', ' '.join(cmd))
        r=subprocess.run(cmd,capture_output=True,text=True)
        print(r.stdout+r.stderr)
        assert r.returncode==0
