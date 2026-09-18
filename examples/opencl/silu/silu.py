from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import tilelang
import tilelang.opencl  # noqa: F401 - registers OpenCL TileOp implementations
import tilelang.language as T


def make_kernel(nelem: int, threads: int):
    @T.prim_func
    def silu_kernel(
        A: T.Tensor((nelem,), "float32"),
        O: T.Tensor((nelem,), "float32"),
    ):
        with T.Kernel(T.ceildiv(nelem, threads), threads=threads) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads + tx
            # No bounds guard in current OpenCL backend: choose nelem divisible by threads.
            x = A[idx]
            O[idx] = x / (T.float32(1.0) + T.exp(-x))

    return silu_kernel


def syntax_check(path: Path) -> int | None:
    clang = subprocess.run(
        ["bash", "-lc", "command -v clang"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if clang.returncode != 0:
        print("CLANG_SYNTAX_SKIP no clang in PATH")
        return None
    headers = Path("/root/project/backend/gpu/OpenCL-Headers")
    cmd = [clang.stdout.strip(), "-x", "cl", "-cl-std=CL3.0", "-fsyntax-only"]
    if headers.exists():
        cmd += ["-I", str(headers)]
    cmd.append(str(path))
    res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(res.stdout, end="")
    print(f"CLANG_SYNTAX_RC {res.returncode}")
    return res.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit TileLang OpenCL SiLU kernel")
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("out") / "silu.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    nelem = args.rows * args.cols
    if nelem % args.threads != 0:
        raise SystemExit(f"nelem={nelem} must be divisible by threads={args.threads}")

    artifact = tilelang.lower(make_kernel(nelem, args.threads), target="opencl", enable_device_compile=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(artifact.kernel_source, encoding="utf-8")
    print(f"EMIT_OK silu rows={args.rows} cols={args.cols} threads={args.threads}")
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")
    if not args.skip_clang:
        syntax_check(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
