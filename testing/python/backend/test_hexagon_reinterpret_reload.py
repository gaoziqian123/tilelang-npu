"""Execute production generated C: stores invalidate loads even through alias."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import tilelang
from tvm import IRModule, tirx as tir, get_global_func
from tvm.target import Target


@pytest.mark.parametrize('alias',[False,True])
def test_reinterpret_store_reload(tmp_path,alias):
    a=tir.decl_buffer((1,),'float32',name='a')
    b=tir.decl_buffer((1,),'float32',name='b')
    out=tir.decl_buffer((2,),'uint32',name='out')
    def bits(x): return tir.call_intrin('uint32','tirx.reinterpret',x)
    body=tir.SeqStmt([
        tir.BufferStore(out,bits(a[0]),[0]),
        tir.BufferStore(b,tir.const(2.0,'float32'),[0]),
        tir.BufferStore(out,bits(a[0]),[1]),
    ])
    f=tir.PrimFunc([a.data,b.data,out.data],body).with_attr('global_symbol','reload').with_attr('calling_conv',2)
    src=get_global_func('target.build.tilelang_hexagon_without_compile')(IRModule({'reload':f}),Target('hexagon')).inspect_source()
    cpp=tmp_path/'reload.cpp';cpp.write_text(src)
    root=Path(__file__).resolve().parents[3]
    so=tmp_path/'reload.so'
    cmd=['/root/autodl-tmp/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++',
         '--target=x86_64-linux-gnu','-O2','-shared','-fPIC','-std=c++17','-I'+str(root/'src'),str(cpp),'-o',str(so)]
    print('COMMAND:', ' '.join(cmd))
    r=subprocess.run(cmd,capture_output=True,text=True);print(r.stdout+r.stderr)
    assert r.returncode==0
    run=ctypes.CDLL(str(so)).reload
    run.argtypes=[ctypes.c_void_p]*3
    av=ctypes.c_float(1.0);bv=ctypes.c_float(9.0);result=(ctypes.c_uint32*2)()
    run(ctypes.byref(av),ctypes.byref(av if alias else bv),result)
    assert list(result)==[0x3f800000,0x40000000 if alias else 0x3f800000]
