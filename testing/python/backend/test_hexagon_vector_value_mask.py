"""Non-FA eager value masks: actual Hexagon object compile, not syntax-only."""
from pathlib import Path
import subprocess
import pytest
import tilelang
from tilelang import language as T


@pytest.mark.parametrize("bits", [16, 32, 64])
def test_mask_carrier_host_bits(tmp_path, bits):
    """Execute the exact emitted select idiom with exceptional payload bits.

    This is a host compiler check of the carrier conversion, not an HVX device
    numerical claim. Unsigned payloads make NaN/-0 preservation bit-observable.
    """
    import shutil
    cc = shutil.which('clang++') or '/root/autodl-tmp/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++'
    src = tmp_path / 'carrier.cpp'
    src.write_text('''
#include <stdint.h>
#include <string.h>
typedef int32_t pred __attribute__((ext_vector_type(32)));
typedef intBITS_t mask __attribute__((ext_vector_type(32)));
typedef uintBITS_t value __attribute__((ext_vector_type(32)));
int main() {
  for(int boundary=-1;boundary<=33;boundary++) {
    pred p; value a,b;
    for(int i=0;i<32;i++) {
      p[i]=i<boundary ? -1 : 0;
      a[i]=(uintBITS_t)(UINT64_C(0x7ff800007fc07e00)+i);
      b[i]=(uintBITS_t)(UINT64_C(0x8000000080008000)+i);
    }
    value got=__builtin_convertvector((p)!=0,mask) ? a : b;
    for(int i=0;i<32;i++) if(got[i]!=(p[i]?a[i]:b[i])) return 1;
  }
}
'''.replace('BITS', str(bits)))
    exe = tmp_path / 'carrier'
    subprocess.run([cc, '-O2', '-std=c++17', str(src), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


@pytest.mark.parametrize("dtype", ["float16", "float32", "int16", "int32"])
@pytest.mark.parametrize("n,offset", [(64, 0), (64, 32), (33, 1)])
@pytest.mark.parametrize("mask", ["plain", "and", "wide"])
def test_value_mask_object(tmp_path, dtype, n, offset, mask):
    @T.prim_func
    def kernel(A: T.Tensor((n + offset,), dtype), B: T.Tensor((n + offset,), dtype)):
        with T.Kernel(1, threads=1):
            for row in T.serial(3):
                for j in T.vectorized(n):
                    if mask == "plain":
                        B[j + offset] = T.Select(j + 7 <= row * 3 + 11,
                                                A[j + offset], T.cast(0, dtype))
                    elif mask == "and":
                        B[j + offset] = T.Select(T.And(j < row * 3 + 11, j >= row),
                                                A[j + offset], T.cast(0, dtype))
                    else:
                        B[j + offset] = T.Select(T.cast(j, "int64") < T.cast(row, "int64") + 17,
                                                A[j + offset], T.cast(0, dtype))
    src = tilelang.engine.lower(kernel, target="hexagon", enable_host_codegen=False,
                                enable_device_compile=False).kernel_source
    cc = Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')
    if not cc.exists():
        pytest.skip('Hexagon SDK unavailable')
    root = Path(tilelang.__file__).resolve().parents[1]
    source = tmp_path / 'mask.cpp'
    source.write_text(src)
    subprocess.run([str(cc), '-mv79', '-mhvx', '-mhvx-length=128B', '-mhmx',
                    '-O2', '-std=c++17', '-I', str(root / 'src'), '-c', str(source),
                    '-o', str(tmp_path / 'mask.o')], check=True)
