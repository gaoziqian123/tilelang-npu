import ctypes
import subprocess
import numpy as np
from tilelang import language as T
from tilelang.hexagon.physical_contract import compact_contract, c_packer, contract, pack


def test_generated_c_all_bits(tmp_path):
    layout=T.Layout((256,256),lambda r,c:(r//32,c//32,r%32//2,c%32*2+r%2))
    spec=compact_contract(layout)
    source=tmp_path/"pack.c"
    source.write_text(c_packer(spec,"pack_internal") +
                     "int run(const void*a,void*b,size_t x,size_t y){return pack_internal(a,b,x,y);}\n")
    lib=tmp_path/"pack.so"
    subprocess.run(["cc","-O2","-shared","-fPIC",str(source),"-o",str(lib)],check=True)
    run=ctypes.CDLL(str(lib)).run
    run.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]
    bits=np.arange(65536,dtype=np.uint16).reshape(256,256)
    raw=np.empty(spec["bytes"]+128,dtype=np.uint8)
    out=raw[(-raw.ctypes.data)%128:][:spec["bytes"]]
    assert run(bits.ctypes.data,out.ctypes.data,bits.nbytes,out.nbytes)==0
    assert np.array_equal(out,pack(bits.view(np.float16),contract(layout)))
    assert run(bits.ctypes.data,out.ctypes.data,bits.nbytes,out.nbytes-1)==-1
    assert run(out.ctypes.data,out.ctypes.data,out.nbytes,out.nbytes)==-4


def test_standard_contract_compact():
    import json
    layout=T.Layout((12288,2560),lambda r,c:(r//32,c//32,c%32//2,r%32*2+c%2))
    spec=compact_contract(layout)
    assert spec["bytes"]==12288*2560*2
    assert len(json.dumps(spec))<2048
