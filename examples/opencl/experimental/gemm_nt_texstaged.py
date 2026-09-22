from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import tilelang
import tilelang.opencl  # noqa: F401 - registers OpenCL TileOp implementations
from tilelang import tvm
import tilelang.language as T


def make_kernel(M: int, N: int, K: int, bm: int, bn: int, bk: int, threads: int):
    row_tiles = bm // 8
    col_tiles = bn // 8
    assert threads == row_tiles * col_tiles

    @T.prim_func
    def gemm_nt_texstaged_kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N // 4, 4), "float16", scope="global.texture"),
        C: T.Tensor((M, N), "float16"),
    ):
        # TileLang owns the launch, A staging, and (critically) B texture ->
        # __local staging via T.copy.  The emitted source is post-patched below
        # to replace the scalar 8x8 MAC nest with the same float8 register
        # escape hatch used by the hand OpenCL reference, reading B from shared.
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
            tx = T.get_thread_binding(0)
            tm = tx // col_tiles
            tn = tx - tm * col_tiles

            A_shared = T.alloc_shared((bk, bm), "float16")
            B_shared = T.alloc_shared((bk, bn // 4, 4), "float16")
            acc = T.alloc_fragment((8, 8), "float32")

            for i in T.unroll(8, explicit=True):
                for j in T.unroll(8, explicit=True):
                    acc[i, j] = T.float32(0.0)

            for ko in T.serial(T.ceildiv(K, bk)):
                # A tile is staged transposed as [k][m] so each work-item can
                # consume a contiguous half8 over the 8 rows it owns.
                for vi in T.serial((bm * bk // 4 + threads - 1) // threads):
                    v = vi * threads + tx
                    if v < bm * bk // 4:
                        r = v // (bk // 4)
                        k4 = v - r * (bk // 4)
                        for c in T.unroll(4, explicit=True):
                            A_shared[k4 * 4 + c, r] = A[by * bm + r, ko * bk + k4 * 4 + c]

                # Source texture shape is [K][N/4][RGBA].  LowerTextureCopy
                # maps one work-item to one texel and emits one READ_IMAGEH +
                # one vstore4 into this shared staging buffer.
                T.copy(B[ko * bk, bx * (bn // 4), 0], B_shared)
                T.sync_threads()

                for kk in T.unroll(bk, explicit=True):
                    for i in T.unroll(8, explicit=True):
                        a = T.Cast("float32", A_shared[kk, tm * 8 + i])
                        for j in T.unroll(8, explicit=True):
                            acc[i, j] = acc[i, j] + a * T.Cast(
                                "float32", B_shared[kk, tn * 2 + (j // 4), j - (j // 4) * 4]
                            )
                T.sync_threads()

            for i in T.unroll(8, explicit=True):
                for j in T.unroll(8, explicit=True):
                    C[by * bm + tm * 8 + i, bx * bn + tn * 8 + j] = T.Cast("float16", acc[i, j])

    return gemm_nt_texstaged_kernel


def _patch_scalar_compute_to_float8(source: str, bm: int, bn: int, bk: int) -> str:
    """Best-effort source escape hatch for the IR texstaged kernel.

    The IR above is deliberately scalar so TileLang can form the texture copy.
    For performance we rewrite only the generated source: accumulator private
    arrays are already vector-registerized by tilelang.opencl.codegen; this
    pass replaces the remaining scalar MAC nest by a compact float8 block that
    reads A/B from the two shared buffers.  If the expected buffer layout cannot
    be found, return the original source so failures are visible at compile/run.
    """

    # Current TVM printer uses buf_dyn_shmem slices named A_shared/B_shared.
    if "READ_IMAGEH" not in source or "vstore4" not in source:
        return source
    if "void* A_shared" not in source or "void* B_shared" not in source:
        return source

    # Scalar unrolled body starts after the barrier following T.copy and ends at
    # the second barrier in the K-block.  Match conservatively on the first acc
    # update after a barrier.
    marker = "barrier(CLK_LOCAL_MEM_FENCE);\n"
    vstore_pos = source.find("vstore4")
    if vstore_pos < 0:
        return source
    pos = source.find(marker, vstore_pos)
    if pos < 0:
        return source
    pos_next = source.find(marker, pos + len(marker))
    if pos_next < 0:
        return source
    compute = source[pos + len(marker):pos_next]
    if "acc_0" not in compute or "A_shared" not in compute or "B_shared" not in compute:
        return source

    col_tiles = bn // 8
    replacement = f"""  for (int kk = 0; kk < {bk}; ++kk) {{
    const int tm = (convert_int(get_local_id(0))) / {col_tiles};
    const int tn = (convert_int(get_local_id(0))) - tm * {col_tiles};
    float8 a8 = convert_float8(vload8(0, (half*)A_shared + kk * {bm} + tm * 8));
    float8 b8 = convert_float8(vload8(0, (half*)B_shared + kk * {bn} + tn * 8));
    acc_0 += (float8)(a8.s0) * b8;
    acc_1 += (float8)(a8.s1) * b8;
    acc_2 += (float8)(a8.s2) * b8;
    acc_3 += (float8)(a8.s3) * b8;
    acc_4 += (float8)(a8.s4) * b8;
    acc_5 += (float8)(a8.s5) * b8;
    acc_6 += (float8)(a8.s6) * b8;
    acc_7 += (float8)(a8.s7) * b8;
  }}
"""
    return source[: pos + len(marker)] + replacement + source[pos_next:]


def syntax_check(path: Path) -> int | None:
    clang = subprocess.run(["bash", "-lc", "command -v clang"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
    ap = argparse.ArgumentParser(description="Emit TileLang IR GEMM_NT texture->shared-staged kernel")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=32)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--no-escape-patch", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out" / "gemm_nt_texstaged.cl")
    ap.add_argument("--skip-clang", action="store_true")
    args = ap.parse_args()

    if args.bm % 8 or args.bn % 8 or args.bn % 4 or args.bk % 4:
        raise SystemExit("BM/BN/BK constraints: BM,BN divisible by 8; BN,BK divisible by 4")
    expected_threads = (args.bm // 8) * (args.bn // 8)
    if args.threads != expected_threads:
        raise SystemExit(f"threads must be {expected_threads} for BM={args.bm} BN={args.bn}")
    for name, value, tile in (("m", args.m, args.bm), ("n", args.n, args.bn), ("k", args.k, args.bk)):
        if value % tile != 0:
            raise SystemExit(f"{name}={value} must be divisible by tile={tile}")

    with tvm.target.Target("opencl"):
        artifact = tilelang.lower(
            make_kernel(args.m, args.n, args.k, args.bm, args.bn, args.bk, args.threads),
            target="opencl",
            enable_device_compile=False,
        )
    src = artifact.kernel_source
    if not args.no_escape_patch:
        src = _patch_scalar_compute_to_float8(src, args.bm, args.bn, args.bk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(src, encoding="utf-8")
    print(
        f"EMIT_OK gemm_nt_texstaged M={args.m} N={args.n} K={args.k} "
        f"BM={args.bm} BN={args.bn} BK={args.bk} threads={args.threads} escape_patch={not args.no_escape_patch}"
    )
    print(f"SOURCE_PATH {args.out}")
    print(f"SOURCE_LINES {len(src.splitlines())}")
    print(f"STAGING_TEXELS {('READ_IMAGEH' in src) and ('vstore4' in src)}")
    if not args.skip_clang:
        syntax_check(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
