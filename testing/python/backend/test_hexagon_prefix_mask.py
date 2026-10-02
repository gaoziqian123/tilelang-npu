"""Prefix-select lowering must handle infinite fill without proving inf == inf."""
import pytest
import tilelang
import subprocess
from pathlib import Path
from tilelang import language as T


@pytest.mark.parametrize("fill", [float("-inf"), 0.0, -13.0])
def test_prefix_select(fill):
    @T.prim_func
    def kernel(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1, threads=1):
            x = T.alloc_local((256,), "float32")
            for j in T.serial(256):
                x[j] = A[j]
            for j in T.serial(256):
                x[j] = T.Select(j <= 97, x[j], T.float32(fill))
            for j in T.serial(256):
                B[j] = x[j]
    src = tilelang.engine.lower(kernel, target="hexagon", enable_host_codegen=False,
                                enable_device_compile=False).kernel_source
    assert "tl::mask_prefix_f32(" in src


def test_dynamic_prefix_boundaries(tmp_path):
    source = tmp_path / "mask.cc"
    source.write_text(r'''
#include <tl_templates/hexagon/mask.h>
#include <cstdint>
#include <cstring>
#include <climits>
int main() {
  const int sizes[]={0,1,7,31,32,33,63,64,65,255,256,257};
  const uint32_t fills[]={0xff800000u,0x7fc01234u,0x80000000u,0xc1500000u};
  for(int n:sizes) for(int last: {INT_MIN,-1,0,30,31,32,63,64,255,256,INT_MAX})
    for(uint32_t bits:fills) for(int offset=0;offset<8;offset++) {
      float a[280]; uint32_t expected[280]; float fill;
      std::memcpy(&fill,&bits,4);
      for(int i=0;i<280;i++) { expected[i]=0x3f000000u+i; std::memcpy(a+i,expected+i,4); }
      for(int i=0;i<n;i++) if(i>last) expected[offset+i]=bits;
      tl::mask_prefix_f32(a+offset,n,last,fill);
      if(std::memcmp(a,expected,sizeof(a))) return 1;
    }
}
'''.replace('#include <climits>', '#include <climits>\n#include <initializer_list>'))
    root = Path(__file__).resolve().parents[3]
    exe = tmp_path / "mask"
    subprocess.run(["c++", "-std=c++17", "-O2", "-I"+str(root/"src"), str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
