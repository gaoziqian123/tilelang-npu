#!/usr/bin/env python3
"""Emit the Hexagon-v2 FA std kernel.

The default output is now the structured v2 implementation: TileLang FA std ABI
with fa256-style online softmax, Q pre-scale, AH row-pair softmax, and fused
PV readout/normalize/writeback.  Set ``TILELANG_HEXAGON_V2_FA_BRIDGE=1`` to
regenerate the older legacy-shell bridge for A/B.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "examples" / "hexagon-legacy" / "fa" / "fa_std.py"
DEFAULT_OUT = Path(__file__).resolve().parent / "out" / "fa_std_v2.c"
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

import tilelang  # noqa: E402,F401
import tilelang.hexagon_v2  # noqa: F401,E402
from tilelang.engine import lower as engine_lower  # noqa: E402


def _load_legacy():
    spec = importlib.util.spec_from_file_location("tl_legacy_fa_std", LEGACY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {LEGACY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_tir(S: int = 1024, HQ: int = 16, HKV: int = 4, D: int = 256, tile: int = 32):
    legacy = _load_legacy()
    return (legacy.fa_std.get_tir(S=S, HQ=HQ, HKV=HKV, D=D, tile=tile)
            .with_attr("global_symbol", "attnops_tl_fa_std_v2")
            .with_attr("hexagon.fa_kv_slice_from_abl", True)
            .with_attr("hexagon.loop_bounds", "g:g_lo:g_hi"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit Hexagon-v2 FA std bridge kernel.")
    parser.add_argument("--s", type=int, default=1024)
    parser.add_argument("--hq", type=int, default=16)
    parser.add_argument("--hkv", type=int, default=4)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--name", default="attnops_tl_fa_std_v2")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-clang", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def emit(args: argparse.Namespace) -> str:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    old = {k: os.environ.get(k) for k in ("TILELANG_HEXAGON_V2_KIND", "TILELANG_HEXAGON_V2_SYMBOL", "TILELANG_HEXAGON_V2_EMIT_C")}
    os.environ["TILELANG_HEXAGON_V2_KIND"] = "fa_std"
    os.environ["TILELANG_HEXAGON_V2_SYMBOL"] = args.name
    os.environ["TILELANG_HEXAGON_V2_EMIT_C"] = str(args.out)
    try:
        src = engine_lower(get_tir(args.s, args.hq, args.hkv, args.d, args.tile), target="hexagon_v2").kernel_source
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    args.out.write_text(src, encoding="utf-8")
    return src


def structural_check(src: str) -> int:
    checks: list[tuple[str, bool]] = []
    checks.append(("deep asm present", "mxmem(%1, %0):deep" in src and "fa2_hmx_mma_deep_split" in src))
    checks.append(("legacy hmx recipe removed", "hrt_hmx_mm_f16" not in src))
    checks.append(("QK kt=8", re.search(r"fa2_hmx_mma_deep_split\(V, qah[\s\S]*?,\s*FA2_ND\);", src) is not None))
    checks.append(("PV online kt chain", "fa2_hmx_mma_deep_split(V, fa2_SROW" in src and "qt + 1" in src))
    checks.append(("bridge pool phases removed", src.count("attnops_pool_run_ctx(") == 0))
    checks.append(("KV slice attrs lowered", "g_lo" in src and "g_hi" in src))
    checks.append(("Q prescale present", "fa2_stage_q_scaled" in src and "0x2C00" in src))
    checks.append(("fp16 AH softmax", "fa2_exp_neg" in src and "fa2_fold_max" in src))
    checks.append(("no ScoreFull RM staging", "hrt_tlgdn_acc_tile_to_vtcm_rm" not in src and "hrt_exp_fp32_vec" not in src))
    rc = 0
    for name, ok in checks:
        print(f"CHECK_{'PASS' if ok else 'FAIL'} {name}")
        rc = rc or (0 if ok else 1)
    return rc


def run_syntax_check(out: Path, skip: bool) -> int:
    if skip:
        print("HEXAGON_CLANG_SKIP requested")
        return 0
    cmd = [str(HEXAGON_CLANG), *HEXAGON_CLANG_FLAGS, str(out)]
    res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(res.stdout, end="")
    if res.returncode:
        print(f"HEXAGON_CLANG_FAIL rc={res.returncode}")
        return res.returncode
    print("HEXAGON_CLANG_PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.s != 1024 or args.tile != 32 or args.d != 256 or args.hq != 16 or args.hkv != 4:
        print("FA_STD_V2_SHAPE_FAIL expected S=1024,tile=32,HQ=16,HKV=4,D=256")
        return 1
    src = emit(args)
    print(f"EMIT_OK path={args.out} lines={len(src.splitlines())}")
    rc = structural_check(src) if args.self_check else 0
    rc = run_syntax_check(args.out, args.skip_clang) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
