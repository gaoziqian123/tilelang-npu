"""Native online causal GQA, expressed only with standard TileLang operations.

RM external ABI. This entry is experimental until the full numerical/RPC gate
in backend/npu/attn/TL_FA_NATIVE.md passes; emission alone is not acceptance.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import tilelang
from tilelang import language as T


def make_fa(block_q=32, block_k=128, workers=4):
    if block_q % 32 or block_k % 32 or 1024 % block_q or 1024 % block_k:
        raise ValueError("tile sizes must divide 1024 and be multiples of 32")

    @T.prim_func
    def fa(Q: T.Tensor((16384, 256), "float16"),
           K: T.Tensor((4096, 256), "float16"),
           V: T.Tensor((4096, 256), "float16"),
           O: T.Tensor((16384, 256), "float16")):
        with T.Kernel(16, 1024 // block_q, threads=workers) as (h, qb):
            q = T.alloc_shared((block_q, 256), "float16")
            k = T.alloc_shared((block_k, 256), "float16")
            v = T.alloc_shared((block_k, 256), "float16")
            vt = T.alloc_shared((256, block_k), "float16")
            p = T.alloc_shared((block_q, block_k), "float16")
            p_rm = T.alloc_shared((block_q, block_k), "float16")
            scores = T.alloc_fragment((block_q, block_k), "float32")
            partial = T.alloc_fragment((block_q, 256), "float32")
            out = T.alloc_fragment((block_q, 256), "float32")
            maximum = T.alloc_fragment((block_q,), "float32")
            old_maximum = T.alloc_fragment((block_q,), "float32")
            denominator = T.alloc_fragment((block_q,), "float32")
            tile_sum = T.alloc_fragment((block_q,), "float32")
            correction = T.alloc_fragment((block_q,), "float32")
            T.copy(Q[h * 1024 + qb * block_q:h * 1024 + (qb + 1) * block_q, :], q)
            T.fill(maximum, -T.infinity("float32"))
            T.clear(denominator)
            T.clear(out)
            for kb in T.serial(T.ceildiv((qb + 1) * block_q, block_k)):
                T.copy(K[(h // 4) * 1024 + kb * block_k:(h // 4) * 1024 + (kb + 1) * block_k, :], k)
                T.copy(V[(h // 4) * 1024 + kb * block_k:(h // 4) * 1024 + (kb + 1) * block_k, :], v)
                T.transpose(v, vt)
                T.gemm(q, k, scores, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_q, block_k):
                    scores[i, j] = T.Select(kb * block_k + j <= qb * block_q + i,
                                                  scores[i, j] * 0.0625, -T.infinity("float32"))
                T.copy(maximum, old_maximum)
                T.reduce_max(scores, maximum, dim=1, clear=False)
                for i in T.Parallel(block_q):
                    correction[i] = T.exp(old_maximum[i] - maximum[i])
                for i, j in T.Parallel(block_q, block_k):
                    scores[i, j] = T.exp(scores[i, j] - maximum[i])
                T.reduce_sum(scores, tile_sum, dim=1)
                for i in T.Parallel(block_q):
                    denominator[i] = denominator[i] * correction[i] + tile_sum[i]
                T.copy(scores, p_rm)
                T.copy(p_rm, p)
                T.gemm(p, vt, partial, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(block_q, 256):
                    out[i, j] = out[i, j] * correction[i] + partial[i, j]
            for i in T.Parallel(block_q):
                denominator[i] = 1.0 / denominator[i]
            for i, j in T.Parallel(block_q, 256):
                out[i, j] = out[i, j] * denominator[i]
            T.copy(out, O[h * 1024 + qb * block_q:h * 1024 + (qb + 1) * block_q, :])
    return fa.with_attr("global_symbol", f"fa_causal_b1_h16_g4_s1024_d256_q{block_q}_k{block_k}_w{workers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--block-q", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=128)
    args = parser.parse_args()
    artifact = tilelang.engine.lower(make_fa(args.block_q, args.block_k), target="hexagon",
                                    enable_host_codegen=False, enable_device_compile=False)
    args.out.write_text(artifact.kernel_source)
    print(f"EMIT_OK {args.out}")
