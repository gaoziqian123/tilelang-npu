from pathlib import Path
import subprocess
import pytest
import tilelang
from tvm import IRModule, tirx as tir, get_global_func
from tvm.target import Target
from tilelang.hexagon.vector_lowering import VectorLowering
from tilelang.hexagon.vector_verify import VerifyVectorRequired


def module(bits, lanes, operation):
    dtype=f"float{bits}"
    a=tir.decl_buffer((lanes,),dtype,name="a")
    b=tir.decl_buffer((lanes,),dtype,name="b")
    ramp=tir.Ramp(0,1,lanes)
    x=tir.BufferLoad(a,[ramp])
    y=tir.Broadcast(tir.const(0.5,dtype),lanes)
    if operation=="cast":
        value=tir.Cast(f"float16x{lanes}",tir.Cast(f"float32x{lanes}",x))
    elif operation=="layout":
        width=1024//bits
        value=tir.Shuffle([x],[i//width*width+(i+1)%width for i in range(lanes)])
    else:
        value=tir.Select(x>y,x*y+x,y)
    body=tir.BufferStore(b,value,[ramp])
    f=tir.PrimFunc([a.data,b.data],body).with_attr("global_symbol","logical")
    f=f.with_attr("calling_conv",2).with_attr("tl.vector_required",1).with_attr("tl.hvx_arithmetic_mode","native_target_v1")
    return IRModule({"logical":f})


@pytest.mark.parametrize("bits,lanes,operation",[(16,128,"cast"),(16,128,"map"),(32,96,"map"),(16,128,"layout"),(32,96,"layout")])
def test_tir_to_object(tmp_path,bits,lanes,operation):
    mod=VerifyVectorRequired()(VectorLowering()(module(bits,lanes,operation)))
    src=get_global_func("target.build.tilelang_hexagon_without_compile")(mod,Target("hexagon")).inspect_source()
    assert "tl::hvx_leaf::load" in src and "tl::hvx_leaf::store" in src
    cc=Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')
    if not cc.exists(): pytest.skip("Hexagon compiler unavailable")
    root=Path(__file__).resolve().parents[3]
    cpp=tmp_path/'logical.cpp'; cpp.write_text(src)
    cmd=[str(cc),'-mv79','-mhvx','-mhvx-length=128B','-O2','-std=c++17','-I'+str(root/'src'),'-c',str(cpp),'-o',str(tmp_path/'logical.o')]
    print('COMMAND:', ' '.join(cmd))
    r=subprocess.run(cmd,capture_output=True,text=True); print(r.stdout+r.stderr)
    assert r.returncode==0


def test_strict_not_substituted():
    mod=module(32,64,"map")
    mod.update_func(mod.get_global_var("logical"),mod["logical"].with_attr("tl.hvx_arithmetic_mode","strict"))
    with pytest.raises(ValueError,match="strict unavailable"):
        VectorLowering()(mod)
