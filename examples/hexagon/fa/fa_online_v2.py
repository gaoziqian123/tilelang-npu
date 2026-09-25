#!/usr/bin/env python3
"""Emit FA prefill Hexagon-v2 explicit-ABI kernel.

This is the production FA v2 entry: full Q/K/V/O pointers plus explicit
KV-head interval.  The internal structure is fa256-style online softmax with
deep-chain HMX and HVX helpers emitted by ``tilelang.hexagon_v2.codegen``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "fa_online_v2.c"
HEXAGON_CLANG = Path("/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang")
if not HEXAGON_CLANG.exists():
    HEXAGON_CLANG = Path("/root/hexagon-deps/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang")
HEXAGON_CLANG_FLAGS = [
    "-mv79", "-mhvx", "-mhvx-length=128B", "-mhmx", "-fsyntax-only",
    "-I/root/hexagon-deps/HEXKL_DIR/hexkl_addon/include",
    "-I/root/hexagon-deps/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/incs",
    "-I/root/hexagon-deps/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/incs/stddef",
    "-I/root/project/backend/npu/attn/skel/hexagon_Release_toolv19_v79",
    "-I/root/project/backend/npu/attn/skel/src",
]

os.environ["PYTHONPATH"] = str(ROOT)
sys.path.insert(0, str(ROOT))

import importlib.util
import tilelang  # noqa: E402,F401
import tilelang.hexagon_v2  # noqa: F401,E402
from tilelang.engine import lower as engine_lower  # noqa: E402

_STD_V2 = Path(__file__).resolve().parent / "fa_std_v2.py"
_spec = importlib.util.spec_from_file_location("tl_hexagon_fa_std_v2", _STD_V2)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {_STD_V2}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_tir = _mod.get_tir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit Hexagon-v2 FA online explicit-ABI kernel.")
    p.add_argument("--name", default="attnops_tl_fa_online_v2")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--skip-clang", action="store_true")
    p.add_argument("--self-check", action="store_true")
    return p.parse_args()


def emit(args: argparse.Namespace) -> str:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    old = {k: os.environ.get(k) for k in ("TILELANG_HEXAGON_V2_KIND", "TILELANG_HEXAGON_V2_SYMBOL", "TILELANG_HEXAGON_V2_EMIT_C")}
    os.environ["TILELANG_HEXAGON_V2_KIND"] = "fa_online_v2"
    os.environ["TILELANG_HEXAGON_V2_SYMBOL"] = args.name
    os.environ["TILELANG_HEXAGON_V2_EMIT_C"] = str(args.out)
    try:
        src = engine_lower(get_tir(), target="hexagon_v2").kernel_source
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    args.out.write_text(src, encoding="utf-8")
    return src


def structural_check(src: str) -> int:
    checks = [
        ("explicit ABI", "unsigned char *q_buf" in src and "head_lo, int head_hi" in src),
        ("no ABL slice", "g_lo = (abl >> 8)" not in src and "g_hi = (abl >> 12)" not in src),
        ("layout comments", "Q/O row-major" in src and "V WH-packed-T" in src),
        ("deep-chain HMX", "activation.hf = mxmem(%1, %0):deep" in src),
        ("Q prescale", "fa2_stage_q_scaled" in src and "0x2C00" in src),
        ("AH softmax", "fa2_fold_max" in src and "fa2_exp_neg" in src),
        ("fused writeback", re.search(r"oah = .*fa2_OAHROW", src) is not None and "1.0f / l" in src),
    ]
    rc = 0
    for name, ok in checks:
        print(f"CHECK_{'PASS' if ok else 'FAIL'} {name}")
        rc = rc or (0 if ok else 1)
    return rc


def run_syntax_check(out: Path, skip: bool) -> int:
    if skip:
        print("HEXAGON_CLANG_SKIP requested")
        return 0
    res = subprocess.run([str(HEXAGON_CLANG), *HEXAGON_CLANG_FLAGS, str(out)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(res.stdout, end="")
    if res.returncode:
        print(f"HEXAGON_CLANG_FAIL rc={res.returncode}")
        return res.returncode
    print("HEXAGON_CLANG_PASS")
    return 0


def main() -> int:
    args = parse_args()
    src = emit(args)
    print(f"EMIT_OK path={args.out} lines={len(src.splitlines())}")
    rc = structural_check(src) if args.self_check else 0
    return run_syntax_check(args.out, args.skip_clang) or rc


if __name__ == "__main__":
    raise SystemExit(main())
