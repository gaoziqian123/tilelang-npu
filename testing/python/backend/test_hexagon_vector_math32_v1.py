"""Standalone tests: no TileLang/TVM import, no device, no simulated-HVX claim."""
import decimal
import math
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "src/tl_templates/hexagon/vector_math32.h"
CC = os.environ.get("HEXAGON_CXX", "/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++")


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def value(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def rnd(x):
    return value(bits(x))


def model(b):
    """IEEE RNE operation model, NOT a Hexagon sf/qf hardware emulator."""
    mag = b & 0x7fffffff
    if mag > 0x7f800000:
        return b | 0x00400000
    if b >> 31 and mag >= 0x42d00000:
        return 0
    if not b >> 31 and mag > 0x42b17217:
        return 0x7f800000
    x = value(b)
    t = rnd(rnd(x * value(0x3fb8aa3b)) + value(0x4b400000))
    k = bits(t) - 0x4b400000
    f = rnd(t - value(0x4b400000))
    r = rnd(x - rnd(f * value(0x3f317200)))
    r = rnd(r - rnd(f * value(0x35bfbe8e)))
    y = value(0x37d00d01)
    coeff = re.search(r"const int coeff\[8\] = \{([^}]+)", HEADER.read_text()).group(1)
    for c in re.findall(r"0x[0-9a-f]+", coeff):
        y = rnd(rnd(y * r) + value(int(c, 16)))
    yb = bits(y)
    e = (yb >> 23) + k
    if e > 0:
        return yb + (k << 23)
    if e < -23:
        return 0
    sig = (yb & 0x7fffff) | 0x800000
    shift = 1 - e
    q, rem = divmod(sig, 1 << shift)
    half = 1 << (shift - 1)
    return q + int(rem > half or (rem == half and q & 1))


class Math32V1(unittest.TestCase):
    def test_high_precision_model(self):
        # Decimal exp is independent of the polynomial, with 70 decimal digits.
        decimal.getcontext().prec = 70
        inputs = {0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff}
        for center in [-104., -150 * math.log(2), -149 * math.log(2),
                       -126 * math.log(2), 0., math.log(value(0x7f7fffff))]:
            b = bits(center)
            inputs.update(range(max(0, b - 32), min(0xffffffff, b + 32) + 1))
        # Range reduction boundaries, exponent boundaries, uniform + raw bits.
        for k in range(-150, 129):
            for offset in [0., .5]:
                b = bits((k + offset) * math.log(2))
                inputs.update(range(max(0, b - 2), b + 3))
        rng = random.Random(3201)
        inputs.update(bits(rng.uniform(-104, 88.72283)) for _ in range(12000))
        inputs.update(rng.getrandbits(32) for _ in range(12000))
        worst = 0.
        count = 0
        for b in sorted(inputs):
            x = value(b)
            if not math.isfinite(x) or not -104 <= x <= value(0x42b17217):
                continue
            got = value(model(b))
            ref = decimal.Decimal.from_float(x).exp()
            err = abs(decimal.Decimal.from_float(got) - ref)
            # Fixed v1 approximate contract: relative + subnormal absolute term.
            bound = decimal.Decimal("0.000002") * ref + decimal.Decimal(2) ** -149
            self.assertLessEqual(err, bound, (hex(b), got, str(ref)))
            if ref >= decimal.Decimal(2) ** -126:
                worst = max(worst, float(err / ref))
            count += 1
        print(f"IEEE-model Decimal70 samples={count} max_normal_relative={worst:.9g}", flush=True)

    def test_special_model(self):
        for b in [0, 0x80000000]:
            self.assertEqual(model(b), 0x3f800000)
        for b in [0x7f800001, 0x7fc12345, 0xff812345, 0xffffffff]:
            self.assertEqual(model(b), b | 0x00400000)
        for b in [0xff800000, bits(-104.), bits(-1000.)]:
            self.assertEqual(model(b), 0)
        for b in [0x7f800000, 0x42b17218, bits(1000.)]:
            self.assertEqual(model(b), 0x7f800000)
        # Single rounding is observably different from mul+add.
        a = decimal.Decimal.from_float(value(0x3f800001))
        b = decimal.Decimal.from_float(value(0x3f7ffffe))
        exact = a * b - 1
        self.assertEqual(rnd(rnd(float(a) * float(b)) - 1.), 0.)
        self.assertEqual(float(exact), -2. ** -46)

    def test_hexagon_object_and_assembly(self):
        if not Path(CC).is_file():
            self.skipTest("Hexagon compiler unavailable; target UNVERIFIED")
        with tempfile.TemporaryDirectory(prefix="math32-v1-") as tmp:
            p = Path(tmp)
            src = p / "probe.cc"
            src.write_text('#include "vector_math32.h"\nextern "C" HVX_Vector probe(HVX_Vector x) { return tl::hvx_math32_v1::exp_f32(x); }\n')
            base = [CC, "-mv79", "-mhvx", "-mhvx-length=128b", "-O2", "-std=c++17",
                    "-ffp-contract=off", "-I" + str(HEADER.parent)]
            for mode, suffix in [("-c", ".o"), ("-S", ".s")]:
                cmd = base + [mode, str(src), "-o", str(p / ("probe" + suffix))]
                result = subprocess.run(cmd, capture_output=True, text=True)
                print("COMMAND:", " ".join(cmd), "\n" + result.stdout + result.stderr + f"rc={result.returncode}", flush=True)
                self.assertEqual(result.returncode, 0)
            asm = (p / "probe.s").read_text()
            self.assertNotRegex(asm, r"\b(?:call|callr|vextract)\b")
            for instruction in ["vmpy", "vadd", "vmux", "vasr"]:
                self.assertIn(instruction, asm)
            print("ASM: vmpy/vadd/vmux/vasr present; no call/callr/vextract", flush=True)
            src.write_text('#include "vector_math32.h"\nHVX_Vector rejected(HVX_Vector x) { return tl::hvx_math32_v1::fma_f32(x,x,x); }\n')
            cmd = base + ["-c", str(src), "-o", str(p / "reject.o")]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print("COMMAND:", " ".join(cmd), "\n" + result.stdout + result.stderr + f"rc={result.returncode}", flush=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("single-rounding FP32 FMA mode is unsupported", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
