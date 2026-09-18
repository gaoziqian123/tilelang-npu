from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import tilelang
import tilelang.opencl  # noqa: F401 - registers OpenCL TileOp implementations
import tilelang.language as T


def make_kernel(rows: int, cols: int, threads: int):
    @T.prim_func
    def rmsnorm_kernel(
        A: T.Tensor((rows, cols), "float32"),
        O: T.Tensor((rows, cols), "float32"),
        eps: T.float32,
        ncols: T.int32,
    ):
        # One work-group per row.  Local memory holds the partial x^2 reduction.
        with T.Kernel(rows, threads=threads) as row:
            tx = T.get_thread_binding(0)
            smem = T.alloc_shared((threads,), "float32")

            smem[tx] = T.float32(0.0)
            for i in T.serial(cols // threads):
                col = i * threads + tx
                x = A[row, col]
                smem[tx] = smem[tx] + x * x
            T.sync_threads()

            for step in T.serial(8):
                stride = 128 >> step
                if tx < stride:
                    smem[tx] = smem[tx] + smem[tx + stride]
                T.sync_threads()

            scale = T.rsqrt(smem[0] / T.Cast("float32", ncols) + eps)
            for i in T.serial(cols // threads):
                col = i * threads + tx
                O[row, col] = A[row, col] * scale

    return rmsnorm_kernel


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
    ap = argparse.ArgumentParser(description="Emit TileLang OpenCL RMSNorm kernel")
    ap.add_argument("--rows", type=int, default=960)
    ap.add_argument("--cols", type=int, default=2560)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out" / "rmsnorm.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    if args.cols % args.threads != 0:
        raise SystemExit(f"cols={args.cols} must be divisible by threads={args.threads}")
    if args.threads & (args.threads - 1):
        raise SystemExit("threads must be a power of two for tree reduction")

    artifact = tilelang.lower(make_kernel(args.rows, args.cols, args.threads), target="opencl", enable_device_compile=False)
    # Legacy TVM OpenCL codegen currently prints scalar args in float-before-int
    # order here; keep the formal example ABI stable for tl_probe/runtime users.
    src = artifact.kernel_source.replace(
        "__global float* restrict O, float eps, int ncols",
        "__global float* restrict O, int ncols, float eps",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(src, encoding="utf-8")
    print(f"EMIT_OK rmsnorm rows={args.rows} cols={args.cols} threads={args.threads}")
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(src.splitlines())}")
    if not args.skip_clang:
        syntax_check(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
