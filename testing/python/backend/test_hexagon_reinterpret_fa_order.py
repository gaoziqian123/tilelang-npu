"""Original store/reload order, no manually introduced SSA workaround."""
import ctypes
from pathlib import Path
import subprocess
import tilelang
from tvm import IRModule, tirx as tir, get_global_func
from tvm.target import Target


def test_maximum_then_alpha(tmp_path):
    mx=tir.decl_buffer((1,),'float32',name='mx')
    old=tir.decl_buffer((1,),'float32',name='old')
    out=tir.decl_buffer((1,),'float32',name='out')
    def roundtrip(x):
        return tir.call_intrin('float32','tirx.reinterpret',tir.call_intrin('uint32','tirx.reinterpret',x))
    body=tir.SeqStmt([tir.BufferStore(mx,tir.Max(roundtrip(mx[0]),roundtrip(old[0])),[0]),
                      tir.BufferStore(out,roundtrip(old[0])-roundtrip(mx[0]),[0])])
    f=tir.PrimFunc([mx.data,old.data,out.data],body).with_attr('global_symbol','ordered').with_attr('calling_conv',2)
    src=get_global_func('target.build.tilelang_hexagon_without_compile')(IRModule({'ordered':f}),Target('hexagon')).inspect_source()
    cpp=tmp_path/'ordered.cpp';cpp.write_text(src)
    so=tmp_path/'ordered.so';root=Path(__file__).resolve().parents[3]
    cmd=['/root/autodl-tmp/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++','--target=x86_64-linux-gnu','-O2','-shared','-fPIC','-std=c++17','-I'+str(root/'src'),str(cpp),'-o',str(so)]
    print('COMMAND:', ' '.join(cmd))
    r=subprocess.run(cmd,capture_output=True,text=True);print(r.stdout+r.stderr);assert r.returncode==0
    run=ctypes.CDLL(str(so)).ordered;run.argtypes=[ctypes.c_void_p]*3
    for panel,previous in [(1.,3.),(3.,1.),(2.,2.),(-4.,-1.)]:
        m,o,y=ctypes.c_float(panel),ctypes.c_float(previous),ctypes.c_float()
        run(ctypes.byref(m),ctypes.byref(o),ctypes.byref(y))
        assert m.value==max(panel,previous)
        assert y.value==previous-max(panel,previous)
