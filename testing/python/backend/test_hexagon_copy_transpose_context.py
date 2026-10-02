"""Exercise the shared guard with absent lowering thread context."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_missing_thread_context(tmp_path):
    cpp = tmp_path/'context.cpp'
    cpp.write_text('''
#include "hexagon/op/affine_transpose.h"
#include <exception>
#include <string>
int main() {
  tvm::tl::LowerArgs args;
  tvm::arith::Analyzer analyzer;
  try {
    tvm::tl::hexagon::LegalAffineTranspose({}, {}, {}, {}, args, &analyzer);
  } catch (const std::exception &e) {
    return std::string(e.what()).find("requires defined thread_bounds and thread_index")
           == std::string::npos;
  }
  return 2;
}
''')
    includes = ['src', '3rdparty/tvm/include', '3rdparty/tvm/src',
                '3rdparty/tvm/3rdparty/tvm-ffi/include',
                '3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include']
    exe = tmp_path/'context'
    subprocess.run(['g++', '-std=c++17', '-DTVM_LOG_CUSTOMIZE=1',
                    *['-I'+str(ROOT/p) for p in includes], str(cpp),
                    '-L'+str(ROOT/'build/lib'), '-Wl,-rpath,'+str(ROOT/'build/lib'),
                    '-ltilelang', '-ltvm_compiler', '-ltvm_runtime', '-ltvm_ffi', '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
