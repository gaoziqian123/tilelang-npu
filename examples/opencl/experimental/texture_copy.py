from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import tilelang
import tilelang.opencl  # noqa: F401 - registers OpenCL TileOp implementations
import tilelang.language as T


def make_kernel(height: int, width: int):
    @T.prim_func
    def texture_copy_kernel(
        B: T.Tensor((height, width, 4), "float16", scope="global.texture"),
        O: T.Tensor((height, width * 4), "float16"),
    ):
        # One work-group per (h, w); four lanes copy the RGBA/half4 channel.
        with T.Kernel(width, height, threads=4) as (w, h):
            c = T.get_thread_binding(0)
            O[h, w * 4 + c] = B[h, w, c] * T.float16(2.0)

    return texture_copy_kernel


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
    ap = argparse.ArgumentParser(description="Emit TileLang OpenCL texture copy kernel")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out" / "texture_copy.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    artifact = tilelang.lower(make_kernel(args.height, args.width), target="opencl", enable_device_compile=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(artifact.kernel_source, encoding="utf-8")
    print(f"EMIT_OK texture_copy height={args.height} width={args.width}")
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(artifact.kernel_source.splitlines())}")
    if not args.skip_clang:
        syntax_check(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
