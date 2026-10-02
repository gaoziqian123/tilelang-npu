"""Emit standard TileLang GEMM_NT and compile a DSP object (no deployment).

Native HMX readout stays fp16 in a shared AH tile buffer.
Each persistent job stages one B panel and computes M x NP outputs with one HMX owner.
The runtime must serialize jobs on that owner and provide a cooperative pool.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import tilelang
from tilelang import language as T
from tilelang.hexagon.cost_model import estimate_launch
from tilelang.hexagon.target import normalize_hexagon_target


def make_gemm(m, n, k, clear_accum=True, np=384, workers=1, block_m=1024):
    if any(d <= 0 or d % 32 for d in (m, n, k, np)) or n % np:
        raise ValueError("M/N/K/NP must be positive multiples of 32; NP must divide N")
    if k > 4096:
        raise ValueError("K > 4096 requires K-panel tiling (not supported)")
    if workers not in (1, 2, 4, 6):
        raise ValueError("workers must be 1, 2, or 4 (fragment layout requires a divisor of 32)")
    if not clear_accum:
        raise ValueError("shared fp16 output requires clear_accum=True")
    if block_m % 32 or m % block_m:
        raise ValueError("block_m must divide M and be a multiple of 32")
    if (block_m + np) * k * 2 + block_m * np * 2 + 256 > 8 * 1024 * 1024:
        raise ValueError("operand staging plus readout exceeds the 8 MiB VTCM budget")
    @T.prim_func
    def gemm_nt(A: T.Tensor((m, k), "float16"), B: T.Tensor((n, k), "float16"), C: T.Tensor((m, n), "float16")):
        with T.Kernel(m // block_m, threads=workers) as bm:
            a = T.alloc_shared((block_m, k), "float16")
            b = T.alloc_shared((np, k), "float16")
            c = T.alloc_shared((block_m, np), "float16")
            T.copy(A[bm * block_m:(bm + 1) * block_m, :], a)
            for bn in T.serial(n // np):
                T.copy(B[bn * np:(bn + 1) * np, :], b)
                if not clear_accum:
                    T.clear(c)
                T.gemm(a, b, c, transpose_B=True, clear_accum=clear_accum)
                T.copy(c, C[bm * block_m:(bm + 1) * block_m, bn * np:(bn + 1) * np], coalesced_width=32)
    return gemm_nt.with_attr("global_symbol", f"gemm_nt_m{m}_n{n}_k{k}_np{np}_w{workers}_clear{int(clear_accum)}_npersistent")


def find_clang():
    sdk = os.environ.get("HEXAGON_SDK")
    if sdk:
        root = Path(sdk).expanduser().resolve()
        candidates = sorted(root.glob("tools/HEXAGON_Tools/*/Tools/bin/hexagon-clang"))
        # Also support the SDK/tools sibling layout used by the build server.
        candidates.append(root.parent.parent.parent / "HEXAGON_TOOLS" / "Tools" / "bin" / "hexagon-clang")
        for candidate in reversed(candidates):
            if candidate.is_file():
                return str(candidate)
    clang = shutil.which("hexagon-clang")
    if clang:
        return clang
    raise RuntimeError("Set HEXAGON_SDK or put hexagon-clang on PATH (or use --skip-clang)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for dim, default in (("m", 1024), ("n", 12288), ("k", 2560), ("np", 384)):
        parser.add_argument("--" + dim, type=int, default=default)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--block-m", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-clang", action="store_true")
    parser.add_argument("--clear-accum", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if any(d <= 0 or d % 32 for d in (args.m, args.n, args.k)):
        parser.error("M/N/K must be positive multiples of 32")
    jobs = args.m // args.block_m if args.block_m > 0 else 0
    if jobs <= 0:
        parser.error("N must contain at least one persistent NP panel job")
    launch = estimate_launch("gemm", {"jobs": jobs, "work_per_job": args.m * args.np * args.k}, normalize_hexagon_target("hexagon"))
    workers = args.workers if args.workers is not None else launch.num_workers
    try:
        kernel = make_gemm(args.m, args.n, args.k, args.clear_accum, args.np, workers, args.block_m)
    except ValueError as error:
        parser.error(str(error))
    if args.out is None:
        args.out = Path(__file__).resolve().parent / "out" / (str(kernel.attrs["global_symbol"]) + ".c")
    print(f"LAUNCH jobs={jobs} workers={workers} estimate={launch}")
    print(f"VTCM block_m={args.block_m} total={(args.block_m + args.np) * args.k * 2 + 2 * args.block_m * args.np + 256} budget={8 * 1024 * 1024}")
    artifact = tilelang.engine.lower(kernel,
                                    target="hexagon", enable_host_codegen=False, enable_device_compile=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(artifact.kernel_source, encoding="utf-8")
    print(f"EMIT_OK path={args.out} lines={len(artifact.kernel_source.splitlines())}", flush=True)
    if args.skip_clang:
        print("HEXAGON_CLANG_SKIP requested")
        return
    cmd = [find_clang(), "-x", "c++", "-mv79", "-mhvx", "-mhvx-length=128B", "-mhmx", "-O2", "-fPIC", "-std=c++17", "-fstack-usage",
           "-I", str(ROOT / "src"), "-c", str(args.out), "-o", str(args.out.with_suffix(".o"))]
    subprocess.run(cmd, check=True)
    print(f"COMPILE_OK path={args.out.with_suffix('.o')}")


if __name__ == "__main__":
    main()
