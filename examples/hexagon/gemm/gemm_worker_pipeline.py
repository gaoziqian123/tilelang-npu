"""Fixed-group GEMM through the production Hexagon pipeline, no device launch."""
import argparse
from pathlib import Path
import tilelang
from tilelang import language as T


def kernel(M=32, N=64, K=64, BM=32, BN=32, BK=32, depth=2, load=3, acc=2,
           cap_workers=6, cap_events=256, cap_vtcm=8 * 1024 * 1024, schedule="round_sync",
           stage_reads=False, cap_ddr=0):
    if any(x <= 0 or x % 32 for x in (M, N, K, BM, BN, BK)):
        raise ValueError("positive tile-aligned static shapes required")
    if M % BM or N % BN or K % BK:
        raise ValueError("tails are not supported")
    @T.prim_func
    def gemm_worker(A: T.Tensor((M, K), "float16"),
                    B: T.Tensor((N, K), "float16"),
                    C: T.Tensor((M, N), "float32")):
        T.func_attr({"tl.workergroup_max_workers": cap_workers,
                     "tl.workergroup_max_events": cap_events,
                     "tl.workergroup_max_vtcm_bytes": cap_vtcm,
                     "tl.workergroup_stage_reads": int(stage_reads),
                     "tl.workergroup_max_ddr_bytes": cap_ddr})
        with T.Kernel(1, threads=load + 1 + acc):
            a = T.alloc_shared((BM, BK), "float16")
            b = T.alloc_shared((BN, BK), "float16")
            partial = T.alloc_shared((BM, BN), "float16")
            c = T.alloc_fragment((BM, BN), "float32")
            for bm in T.serial(M // BM):
                for bn in T.serial(N // BN):
                    T.clear(c)
                    for bk in T.Pipelined(K // BK, num_stages=depth, annotations={"tl.workergroup_schedule": schedule}):
                        with T.pipeline_stage("fetch", engine="hvx", workers=load):
                            T.copy(A[bm*BM:(bm+1)*BM, bk*BK:(bk+1)*BK], a)
                            T.copy(B[bn*BN:(bn+1)*BN, bk*BK:(bk+1)*BK], b)
                        with T.pipeline_stage("product", engine="hmx", workers=1):
                            T.gemm(a, b, partial, transpose_B=True, clear_accum=True)
                        with T.pipeline_stage("update", engine="hvx", workers=acc):
                            for i, j in T.Parallel(BM, BN):
                                c[i, j] += T.cast(partial[i, j], "float32")
                    T.copy(c, C[bm*BM:(bm+1)*BM, bn*BN:(bn+1)*BN])
    return gemm_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("M",32),("N",64),("K",64),("BM",32),("BN",32),("BK",32),
                          ("depth",2),("load",3),("acc",2),("cap-workers",6),
                          ("cap-events",256),("cap-vtcm",8388608)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schedule", choices=["round_sync", "async"], default="round_sync")
    parser.add_argument("--stage-reads", action="store_true")
    parser.add_argument("--cap-ddr", type=int, default=0)
    args = vars(parser.parse_args())
    output = args.pop("output")
    result = tilelang.engine.lower(kernel(**args), target="hexagon",
                                  enable_host_codegen=False, enable_device_compile=False)
    output.write_text(result.kernel_source)


if __name__ == "__main__":
    main()
