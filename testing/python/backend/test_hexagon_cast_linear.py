"""Exact generic casts: independent bit oracle and actual Hexagon object gate."""
import ctypes
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path('/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin')


@pytest.mark.parametrize('lanes', [32, 64, 128])
def test_linear(tmp_path, lanes):
    source = tmp_path/'cast.cpp'
    source.write_text('#include <tl_templates/hexagon/cast_layout.h>\n'
                      'extern "C" void run(void*d,const float*s){'
                      f'tl::cast_linear_f32_f16<{lanes}>(d,s);}}\n')
    lib = tmp_path/'cast.so'
    subprocess.run(['g++', '-O2', '-shared', '-fPIC', '-std=c++17',
                    '-I'+str(ROOT/'src'), str(source), '-o', str(lib)], check=True)
    run = ctypes.CDLL(str(lib)).run
    run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    rng = np.random.default_rng(8731)
    special = np.array([0, 0x80000000, 0x477fefff, 0x477ff000, 0x477ff001,
                        0xc77ff000, 0x33000000, 0x33800000, 0x7f800001,
                        0xffbfffff, 0x7f800000, 0xff800000], dtype=np.uint32)
    for offset in [0, 1, 7, 31]:
        for batch in range(64):
            x = rng.integers(0, 2**32, lanes+32, dtype=np.uint32)
            x[offset:offset+len(special)] = special
            raw = np.full(2*lanes+132, 0xa5, np.uint8)
            run(raw.ctypes.data+2, x.ctypes.data+4*offset)
            u = x[offset:offset+lanes]
            with np.errstate(over='ignore', invalid='ignore'):
                ref = u.view(np.float32).astype(np.float16).view(np.uint16)
            nan = (u & 0x7fffffff) > 0x7f800000
            # Contract: preserve the representable high payload and quiet bit.
            # NumPy additionally folds discarded signalling payload into bit 0.
            ref[nan] = ((u[nan] >> 16) & 0x8000) | 0x7e00 | ((u[nan] >> 13) & 0x3ff)
            np.testing.assert_array_equal(raw[2:2+lanes*2].view(np.uint16), ref)
            assert np.all(raw[:2] == 0xa5) and np.all(raw[2+lanes*2:] == 0xa5)
    obj = tmp_path/'cast.o'
    subprocess.run([str(TOOLS/'hexagon-clang++'), '-mv79', '-mhvx',
                    '-mhvx-length=128B', '-O2', '-std=c++17',
                    '-I'+str(ROOT/'src'), '-c', str(source), '-o', str(obj)], check=True)
    asm = subprocess.check_output([str(TOOLS/'hexagon-llvm-objdump'), '-dr', str(obj)], text=True)
    assert '__truncsfhf2' not in asm
    assert 'vinsert' not in asm
    assert re.search(r'v\d+\.hf = v\d+:\d+\.qf32', asm)
