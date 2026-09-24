#!/usr/bin/env python3
"""Emit small-panel TileLang Hexagon GEMM_NT variants.

These variants intentionally reuse the existing attnops_tl_gemm_nt ABI and IDL
entry shape (slab,w,M,N,K,abl).  They exercise user-selected HMX tile widths that
are smaller than the old {1024,512,256} panel set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gemm_nt import emit as emit_gemm_nt, run_syntax_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit block_N=128 and block_N=32 Hexagon GEMM variants.")
    parser.add_argument("--m", type=int, default=960)
    parser.add_argument("--n", type=int, default=None, help="Override N for both variants; defaults are 256 for block_N=128 and 128 for block_N=32.")
    parser.add_argument("--k", type=int, default=2560)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--skip-clang", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for bn, default_n in ((128, 256), (32, 128)):
        out = args.out_dir / f"gemm_small_bn{bn}.c"
        ns = argparse.Namespace(m=args.m, n=args.n or default_n, k=args.k, block_m=32, block_n=bn, out=out,
                                skip_clang=args.skip_clang, name=f"attnops_tl_gemm_bn{bn}")
        src = emit_gemm_nt(ns)
        print(f"EMIT_SMALL_OK block_N={bn} path={out} lines={len(src.splitlines())}")
        rc = run_syntax_check(out, args.skip_clang)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
