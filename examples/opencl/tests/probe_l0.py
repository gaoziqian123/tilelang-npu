from __future__ import annotations

import re

import tilelang
import tilelang.opencl  # noqa: F401 - register OpenCL backend
import tilelang.language as T


M = 128
N = 128
NELEM = M * N
THREADS = 256


@T.prim_func
def probe_l0_kernel(A: T.Tensor((NELEM,), "float32"), B: T.Tensor((NELEM,), "float32")):
    with T.Kernel(T.ceildiv(NELEM, THREADS), threads=THREADS) as bx:
        tx = T.get_thread_binding(0)
        i = bx * THREADS + tx
        B[i] = A[i] + T.float32(1.0)


def main() -> int:
    artifact = tilelang.lower(probe_l0_kernel, target="opencl", enable_device_compile=False)
    src = artifact.kernel_source
    if "__kernel void probe_l0_kernel_kernel" not in src:
        raise AssertionError("missing OpenCL kernel entry")
    if src.count("get_global_id") != 0:
        # TileLang's OpenCL path lowers via get_group_id/get_local_id, not global_id.
        raise AssertionError("unexpected generic global-id lowering")
    if "get_group_id(0)" not in src or "get_local_id(0)" not in src:
        raise AssertionError("missing work-group/local-id indexing")
    if not re.search(r"B\[[^\]]+\]\s*=\s*\(A\[[^\]]+\] \+ 1\.000000e\+00f\)", src):
        raise AssertionError("missing B[i] = A[i] + 1 elementwise store")
    if "for (" in src:
        raise AssertionError("L0 probe should lower to one element per work-item without loops")
    print("PROBE_L0_OK")
    print(f"SOURCE_LINES {len(src.splitlines())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
