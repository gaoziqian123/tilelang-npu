from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import tilelang
import tilelang.opencl  # noqa: F401 - registers OpenCL TileOp implementations
from tilelang import tvm
import tilelang.language as T


def make_kernel(height: int, width: int, threads: int):
    @T.prim_func
    def texture_staging_kernel(
        X: T.Tensor((height, width, 4), "float16", scope="global.texture"),
        O: T.Tensor((height,), "float32"),
    ):
        # One work-group per row.  Stage the whole RGBA row from texture into
        # __local memory, then reduce from the staged copy after a barrier.
        with T.Kernel(height, threads=threads) as row:
            tx = T.get_thread_binding(0)
            staged = T.alloc_shared((width, 4), "float16")
            partial = T.alloc_shared((threads,), "float32")

            T.copy(X[row, 0, 0], staged)
            T.sync_threads()

            partial[tx] = T.float32(0.0)
            for i in T.serial((width * 4) // threads):
                idx = i * threads + tx
                tex = idx // 4
                ch = idx - tex * 4
                partial[tx] = partial[tx] + T.Cast("float32", staged[tex, ch])
            T.sync_threads()

            for step in T.serial(8):
                stride = 128 >> step
                if tx < stride:
                    partial[tx] = partial[tx] + partial[tx + stride]
                T.sync_threads()

            if tx == 0:
                O[row] = partial[0]

    return texture_staging_kernel


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
    ap = argparse.ArgumentParser(description="Emit TileLang OpenCL texture staging kernel")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out" / "texture_staging.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    if (args.width * 4) % args.threads != 0:
        raise SystemExit(f"width*4={args.width * 4} must be divisible by threads={args.threads}")
    if args.threads != 256:
        raise SystemExit("threads must be 256 for the fixed 8-step tree reduction")

    with tvm.target.Target("opencl"), tvm.transform.PassContext():
        artifact = tilelang.lower(make_kernel(args.height, args.width, args.threads), target="opencl", enable_device_compile=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(artifact.kernel_source, encoding="utf-8")
    print(f"EMIT_OK texture_staging height={args.height} width={args.width} threads={args.threads}")
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")
    if not args.skip_clang:
        syntax_check(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
