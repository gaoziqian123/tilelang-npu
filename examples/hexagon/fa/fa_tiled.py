"""Tile-resident online attention, independent RM ABI experiment.

The block product [diag(alpha), 0, P] @ [Oprev; 0; V] expresses
correction and PV in ONE ordinary GEMM accumulator lifecycle. The 32 zero
columns align the probability region to 64 elements; they are not omitted
work or a change to the causal FLOP accounting. No FA-specific helper.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import tilelang
from tilelang import language as T


def make_fa(bq=32):
    prefix = max(64, bq)
    width = prefix + 256
    blocks = 1024 // bq
    @T.prim_func
    def fa(Q: T.Tensor((16384,256), "float16"),
           K: T.Tensor((4096,256), "float16"),
           V: T.Tensor((4096,256), "float16"),
           O: T.Tensor((16384,256), "float16")):
        with T.Kernel(16384//bq, threads=4) as job:
            tx = T.get_thread_binding()
            q = T.alloc_shared((bq,256), "float16")
            k = T.alloc_shared((256,256), "float16")
            vr = T.alloc_shared((width,256), "float16")
            vt = T.alloc_shared((256,width), "float16")
            p = T.alloc_shared((bq,width), "float16")
            scores = T.alloc_shared((bq,256), "float16")
            out = T.alloc_shared((bq,256), "float16")
            maximum = T.alloc_local((bq//4,), "float32")
            denominator = T.alloc_local((bq//4,), "float32")
            alpha = T.alloc_local((2,), "float32")
            rowh = T.alloc_local((2,256), "float16")
            prow = T.alloc_local((2,width), "float16")
            rowf = T.alloc_local((2,256), "float32")
            rowmax = T.alloc_local((2,), "float32")
            rowsum = T.alloc_local((2,), "float32")
            T.copy(Q[job*bq:(job+1)*bq,:],q)
            T.clear(vr)
            for i in T.serial(bq//4):
                maximum[i]=-T.infinity("float32")
                denominator[i]=0.0
            for kb in T.serial(T.ceildiv((job%blocks+1)*bq,256)):
                T.copy(K[(job//(blocks*4))*1024+kb*256:(job//(blocks*4))*1024+(kb+1)*256,:],k)
                T.copy(V[(job//(blocks*4))*1024+kb*256:(job//(blocks*4))*1024+(kb+1)*256,:],vr[prefix:width,:])
                if kb > 0:
                    for rp in T.serial(bq//8):
                        T.copy(out[rp*8+tx*2:rp*8+tx*2+2,:],rowh)
                        for r in T.serial(2):
                            for j in T.vectorized(256):
                                vr[rp*8+tx*2+r,j]=rowh[r,j]
                T.sync_threads()
                T.transpose(vr,vt)
                T.gemm(q,k,scores,transpose_B=True,clear_accum=True)
                for rp in T.serial(bq//8):
                    T.copy(scores[rp*8+tx*2:rp*8+tx*2+2,:],rowh)
                    for r in T.serial(2):
                        for j in T.vectorized(256):
                            rowf[r,j]=T.cast(rowh[r,j],"float32")*0.0625
                        for j in T.vectorized(256):
                            rowf[r,j]=T.Select(kb*256+j<=(job%blocks)*bq+rp*8+tx*2+r,
                                              rowf[r,j],-T.infinity("float32"))
                    T.reduce_max(rowf,rowmax,dim=1)
                    for r in T.serial(2):
                        rowmax[r]=T.max(maximum[rp*2+r],rowmax[r])
                        alpha[r]=T.exp(maximum[rp*2+r]-rowmax[r])
                        maximum[rp*2+r]=rowmax[r]
                        for j in T.vectorized(256):
                            rowf[r,j]=T.exp(rowf[r,j]-rowmax[r])
                    T.reduce_sum(rowf,rowsum,dim=1)
                    for r in T.serial(2):
                        denominator[rp*2+r]=denominator[rp*2+r]*alpha[r]+rowsum[r]
                        for j in T.vectorized(prefix):
                            prow[r,j]=T.float16(0)
                        prow[r,rp*8+tx*2+r]=T.cast(alpha[r],"float16")
                        for j in T.vectorized(256):
                            prow[r,j+prefix]=T.cast(rowf[r,j],"float16")
                    T.copy(prow,p[rp*8+tx*2:rp*8+tx*2+2,:])
                T.gemm(p,vt,out,transpose_B=True,clear_accum=True)
            for rp in T.serial(bq//8):
                T.copy(out[rp*8+tx*2:rp*8+tx*2+2,:],rowh)
                for r in T.serial(2):
                    denominator[rp*2+r]=1.0/denominator[rp*2+r]
                    for j in T.vectorized(256):
                        O[job*bq+rp*8+tx*2+r,j]=T.cast(
                            T.cast(rowh[r,j],"float32")*denominator[rp*2+r],"float16")
            T.sync_threads()
    return fa.with_attr("global_symbol", "fa_tiled_rm_b1_h16_g4_s1024_d256")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--block-q",type=int,choices=(32,64,128),default=128)
    args=parser.parse_args()
    artifact=tilelang.engine.lower(make_fa(args.block_q),target="hexagon",enable_host_codegen=False,
                                   enable_device_compile=False)
    args.out.write_text(artifact.kernel_source)
    print("EMIT_OK",args.out)
