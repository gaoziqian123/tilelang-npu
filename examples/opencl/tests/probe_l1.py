from __future__ import annotations

import tilelang
import tilelang.opencl  # noqa: F401 - register OpenCL backend
import tilelang.language as T


ROWS = 128
COLS = 256
THREADS = 256


@T.prim_func
def probe_l1_kernel(A: T.Tensor((ROWS, COLS), "float32"), O: T.Tensor((ROWS,), "float32")):
    with T.Kernel(ROWS, threads=THREADS) as row:
        tx = T.get_thread_binding(0)
        smem = T.alloc_shared((THREADS,), "float32")
        smem[tx] = A[row, tx]
        T.sync_threads()

        for step in T.serial(8):
            stride = 128 >> step
            if tx < stride:
                smem[tx] = smem[tx] + smem[tx + stride]
            T.sync_threads()

        if tx == 0:
            O[row] = smem[0]


def main() -> int:
    artifact = tilelang.lower(probe_l1_kernel, target="opencl", enable_device_compile=False)
    src = artifact.kernel_source
    if "__kernel void probe_l1_kernel_kernel" not in src:
        raise AssertionError("missing OpenCL kernel entry")
    if "__local float" not in src:
        raise AssertionError("missing local/shared reduction buffer")
    if src.count("barrier(CLK_LOCAL_MEM_FENCE)") < 2:
        raise AssertionError("missing shared-memory barriers")
    if "for (int step = 0; step < 8; ++step)" not in src or "128 >> step" not in src:
        raise AssertionError("tree-reduction loop was not emitted")
    if "O[(convert_int(get_group_id(0)))]" not in src:
        raise AssertionError("missing row output store")
    print("PROBE_L1_OK")
    print(f"SOURCE_LINES {len(src.splitlines())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
