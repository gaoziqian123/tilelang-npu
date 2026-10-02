"""Persistent KV-head schedule; standard TileLang operations, RM external ABI.

First stage keeps RM K/V resident. Layout conversion remains per consumer;
this isolates DDR reuse from the later packed-layout producer/consumer work.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import tilelang
from tilelang import language as T
from tilelang.hexagon.language.accumulator import block_diagonal_product_sum_f16
from tilelang.hexagon import _ffi_api
from tvm.ir import Op
from tilelang.hexagon.language.async_scope import submit, wait


def make_fa(tile_state=False, fused_state=False, mixed_exp=False, grouped_rows=False, pipeline=False, instrumentation=True, workers=4):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    pairs_per_worker = (64 + workers - 1) // workers
    fused=block_diagonal_product_sum_f16(128,256,256)
    @T.macro
    def profile(region):
        if instrumentation:
            T.evaluate(T.call_extern("handle", "tl::profile_region", region))
    @T.macro
    def copy_rows(src,dst,base,rows,tx):
        # Whole 128-byte vectors have one cyclic owner; never distribute
        # individual half lanes across a non-power-of-two worker count.
        staging=T.alloc_local((2,256),"float16")
        for vi in T.serial(T.ceildiv(rows//2,workers)):
            if vi*workers+tx < rows//2:
                for r in T.serial(2):
                    for lane in T.vectorized(256):
                        staging[r,lane]=src[base+(vi*workers+tx)*2+r,lane]
                T.copy(staging,dst[(vi*workers+tx)*2:(vi*workers+tx)*2+2,:])
        T.sync_threads()
    @T.macro
    def qk(Q,A,B,C,D,S,block):
        if block == 0:
            T.gemm(Q,A,S,transpose_B=True,clear_accum=True)
        elif block == 1:
            T.gemm(Q,B,S,transpose_B=True,clear_accum=True)
        elif block == 2:
            T.gemm(Q,C,S,transpose_B=True,clear_accum=True)
        else:
            T.gemm(Q,D,S,transpose_B=True,clear_accum=True)
    @T.prim_func
    def fa(Q: T.Tensor((16384,256), "float16"),
           K: T.Tensor((4096,256), "float16"),
           V: T.Tensor((4096,256), "float16"),
           O: T.Tensor((16384,256), "float16")):
        with T.Kernel(4, threads=workers) as kh:
            tx = T.get_thread_binding()
            kr = T.alloc_shared((1024,256), "float16")
            vr = T.alloc_shared((1024,256), "float16")
            q = T.alloc_shared((128,256), "float16")
            k = T.alloc_shared((256,256), "float16")
            k1 = T.alloc_shared((256,256), "float16")
            k2 = T.alloc_shared((256,256), "float16")
            k3 = T.alloc_shared((256,256), "float16")
            v = T.alloc_shared((256,256), "float16")
            vt = T.alloc_shared((256,256), "float16")
            vt1 = T.alloc_shared((256,256), "float16")
            vt2 = T.alloc_shared((256,256), "float16")
            vt3 = T.alloc_shared((256,256), "float16")
            p = T.alloc_shared((128,256), "float16")
            scores = T.alloc_shared((128,256), "float16")
            scores_next = T.alloc_shared((128,256), "float16")
            partial = T.alloc_shared((128,256), "float16")
            out = T.alloc_local((pairs_per_worker*2,256), "float32")
            state = T.alloc_shared((128,256), "float16")
            oldrow = T.alloc_local((2,256), "float16")
            diag = T.alloc_shared((128,32),"float16")
            diagrow = T.alloc_local((2,32),"float16")
            if fused_state:
                T.annotate_layout({diag:_ffi_api.make_layout("ah",diag),
                    state:_ffi_api.make_layout("ah",state),
                    p:_ffi_api.make_layout("ah",p),vt:_ffi_api.make_layout("wh",vt),
                    vt1:_ffi_api.make_layout("wh",vt1),vt2:_ffi_api.make_layout("wh",vt2),vt3:_ffi_api.make_layout("wh",vt3)})
            maximum = T.alloc_local((pairs_per_worker*2,), "float32")
            denominator = T.alloc_local((pairs_per_worker*2,), "float32")
            correction = T.alloc_local((pairs_per_worker*2,), "float32")
            rowh = T.alloc_local((2,256), "float16")
            rowf = T.alloc_local((2,256), "float32")
            rowmax = T.alloc_local((2,), "float32")
            rowsum = T.alloc_local((2,), "float32")
            profile(1)
            copy_rows(K,kr,kh*1024,1024,tx)
            copy_rows(V,vr,kh*1024,1024,tx)
            T.copy(kr[0:256,:],k)
            T.copy(kr[256:512,:],k1)
            T.copy(kr[512:768,:],k2)
            T.copy(kr[768:1024,:],k3)
            copy_rows(vr,v,0,256,tx)
            T.transpose(v,vt)
            copy_rows(vr,v,256,256,tx)
            T.transpose(v,vt1)
            copy_rows(vr,v,512,256,tx)
            T.transpose(v,vt2)
            copy_rows(vr,v,768,256,tx)
            T.transpose(v,vt3)
            for hlocal in T.serial(4):
                for qb in T.serial(8):
                    profile(1)
                    if grouped_rows:
                        for rp in T.serial(T.ceildiv(64-tx,workers)):
                            for r in T.serial(2):
                                for j in T.vectorized(256):
                                    rowh[r,j]=Q[(kh*4+((rp*workers+tx)*2+r)%4)*1024+(hlocal*8+qb)*32+((rp*workers+tx)*2+r)//4,j]
                            T.copy(rowh,q[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:])
                        T.sync_threads()
                    else:
                        copy_rows(Q,q,(kh*4+hlocal)*1024+qb*128,128,tx)
                    if tile_state:
                        T.clear(state)
                    if fused_state:
                        for r in T.serial(2):
                            for j in T.vectorized(256):
                                oldrow[r,j]=T.float16(0)
                        for rp in T.serial(T.ceildiv(64-tx,workers)):
                            T.copy(oldrow,state[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:])
                    for i in T.serial(pairs_per_worker*2):
                        maximum[i]=-T.infinity("float32")
                        denominator[i]=0.0
                        if not tile_state and not fused_state:
                            for j in T.vectorized(256):
                                out[i,j]=0.0
                    if pipeline:
                        T.sync_threads()
                        with submit(0):
                            qk(q,k,k1,k2,k3,scores,0)
                    for kb in T.serial(T.ceildiv((hlocal*8+qb+1)*32 if grouped_rows else (qb+1)*128,256)):
                        profile(2)
                        if pipeline:
                            if kb==0:
                                wait(0)
                            if kb+1<T.ceildiv((qb+1)*128,256):
                                with submit(0):
                                    qk(q,k,k1,k2,k3,scores_next,kb+1)
                        elif kb == 0:
                            T.gemm(q,k,scores,transpose_B=True,clear_accum=True)
                        elif kb == 1:
                            T.gemm(q,k1,scores,transpose_B=True,clear_accum=True)
                        elif kb == 2:
                            T.gemm(q,k2,scores,transpose_B=True,clear_accum=True)
                        else:
                            T.gemm(q,k3,scores,transpose_B=True,clear_accum=True)
                        profile(3)
                        for rp in T.serial(T.ceildiv(64-tx,workers)):
                            profile(8)
                            T.copy(scores[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:],rowh)
                            for r in T.serial(2):
                                for j in T.vectorized(256):
                                    rowf[r,j]=T.cast(rowh[r,j],"float32")*0.0625
                                for j in T.vectorized(256):
                                    rowf[r,j]=T.Select(kb*256+j<=((hlocal*8+qb)*32+((rp*workers+tx)*2+r)//4 if grouped_rows else qb*128+(rp*workers+tx)*2+r),
                                        rowf[r,j],-T.infinity("float32"))
                            profile(9)
                            T.reduce_max(rowf,rowmax,dim=1)
                            profile(10)
                            for r in T.serial(2):
                                profile(13)
                                rowmax[r]=T.max(maximum[rp*2+r],rowmax[r])
                                correction[rp*2+r]=T.exp(maximum[rp*2+r]-rowmax[r])
                                maximum[rp*2+r]=rowmax[r]
                                profile(14)
                                # Causal tile extent, not a value-dependent fast path.
                                # Keep masked tails exactly zero while avoiding exp
                                # work for wholly invisible 32-lane vectors.
                                for chunk in T.serial(T.ceildiv(T.min(256, T.max(0,
                                        ((hlocal*8+qb)*32+((rp*workers+tx)*2+r)//4 if grouped_rows else qb*128+(rp*workers+tx)*2+r)-kb*256+1)),32)):
                                    for lane in T.vectorized(32):
                                        if mixed_exp:
                                            rowf[r,chunk*32+lane]=T.cast(T.call_intrin("float16", Op.get("tl.hexagon.exp_f16_softmax"), T.cast(rowf[r,chunk*32+lane]-rowmax[r],"float16")),"float32")
                                        else:
                                            rowf[r,chunk*32+lane]=T.exp(rowf[r,chunk*32+lane]-rowmax[r])
                                for chunk in T.serial(T.ceildiv(T.min(256, T.max(0,
                                        ((hlocal*8+qb)*32+((rp*workers+tx)*2+r)//4 if grouped_rows else qb*128+(rp*workers+tx)*2+r)-kb*256+1)),32),8):
                                    for lane in T.vectorized(32):
                                        rowf[r,chunk*32+lane]=0.0
                            profile(11)
                            T.reduce_sum(rowf,rowsum,dim=1)
                            profile(12)
                            for r in T.serial(2):
                                denominator[rp*2+r]=denominator[rp*2+r]*correction[rp*2+r]+rowsum[r]
                                for j in T.vectorized(256):
                                    rowh[r,j]=T.cast(rowf[r,j],"float16")
                            T.copy(rowh,p[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:])
                            if fused_state:
                                for r in T.serial(2):
                                    for j in T.vectorized(32):
                                        diagrow[r,j]=T.float16(0)
                                    diagrow[r,((rp*workers+tx)*2+r)%32]=T.cast(correction[rp*2+r],"float16")
                                T.copy(diagrow,diag[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:])
                        profile(4)
                        if pipeline:
                            T.sync_threads()
                            with submit(1):
                                qk(p,vt,vt1,vt2,vt3,partial,kb)
                            wait(1)
                        elif fused_state:
                            if kb == 0:
                                fused(diag,p,vt,state)
                            elif kb == 1:
                                fused(diag,p,vt1,state)
                            elif kb == 2:
                                fused(diag,p,vt2,state)
                            else:
                                fused(diag,p,vt3,state)
                        elif kb == 0:
                            T.gemm(p,vt,partial,transpose_B=True,clear_accum=True)
                        elif kb == 1:
                            T.gemm(p,vt1,partial,transpose_B=True,clear_accum=True)
                        elif kb == 2:
                            T.gemm(p,vt2,partial,transpose_B=True,clear_accum=True)
                        else:
                            T.gemm(p,vt3,partial,transpose_B=True,clear_accum=True)
                        profile(5)
                        for rp in T.serial(0 if fused_state else T.ceildiv(64-tx,workers)):
                            T.copy(partial[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:],rowh)
                            if tile_state:
                                T.copy(state[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:],oldrow)
                            for r in T.serial(2):
                                for j in T.vectorized(256):
                                    if tile_state:
                                        oldrow[r,j]=T.cast(T.cast(oldrow[r,j],"float32")*correction[rp*2+r]+T.cast(rowh[r,j],"float32"),"float16")
                                    else:
                                        out[rp*2+r,j]=out[rp*2+r,j]*correction[rp*2+r]+T.cast(rowh[r,j],"float32")
                            if tile_state:
                                T.copy(oldrow,state[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:])
                        T.sync_threads()
                        if pipeline and kb+1<T.ceildiv((qb+1)*128,256):
                            wait(0)
                            for rp in T.serial(T.ceildiv(64-tx,workers)):
                                T.copy(scores_next[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:],rowh)
                                T.copy(rowh,scores[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:])
                            T.sync_threads()
                    profile(6)
                    for rp in T.serial(T.ceildiv(64-tx,workers)):
                        if tile_state or fused_state:
                            T.copy(state[(rp*workers+tx)*2:(rp*workers+tx)*2+2,:],oldrow)
                        for r in T.serial(2):
                            denominator[rp*2+r]=1.0/denominator[rp*2+r]
                            for j in T.vectorized(256):
                                if tile_state or fused_state:
                                    O[(kh*4+hlocal)*1024+qb*128+(rp*workers+tx)*2+r,j]=T.cast(T.cast(oldrow[r,j],"float32")*denominator[rp*2+r],"float16")
                                elif grouped_rows:
                                    O[(kh*4+((rp*workers+tx)*2+r)%4)*1024+(hlocal*8+qb)*32+((rp*workers+tx)*2+r)//4,j]=T.cast(out[rp*2+r,j]*denominator[rp*2+r],"float16")
                                else:
                                    O[(kh*4+hlocal)*1024+qb*128+(rp*workers+tx)*2+r,j]=T.cast(out[rp*2+r,j]*denominator[rp*2+r],"float16")
                    T.sync_threads()
            profile(0)
    return fa.with_attr("global_symbol", "fa_causal_persistent_rm_b1_h16_g4_s1024_d256")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--tile-state",action="store_true")
    parser.add_argument("--fused-state",action="store_true")
    parser.add_argument("--mixed-exp",action="store_true",help="explicit fp16 exp; max/sum/correction and default O remain fp32")
    parser.add_argument("--grouped-rows",action="store_true")
    parser.add_argument("--pipeline",action="store_true")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument('--block-q',type=int,default=128)
    parser.add_argument('--block-k',type=int,default=256)
    parser.add_argument('--tile-schedule',action='store_true')
    parser.add_argument('--explicit-hf',action='store_true')
    parser.add_argument('--round-local-sum',action='store_true')
    parser.add_argument("--no-instrumentation", action="store_true",
                        help="omit all region profiling at DSL construction time")
    args=parser.parse_args()
    if args.grouped_rows and (args.tile_state or args.fused_state):
        parser.error("grouped rows currently supports fp32 O only")
    if args.tile_schedule:
        if args.tile_state or args.fused_state or args.mixed_exp or args.grouped_rows or args.pipeline:
            parser.error('tile schedule requires fp32 state/exp and ungrouped synchronous execution')
        from fa_tile_schedule import make_fa as make_tile
        kernel=make_tile(args.block_q,args.block_k,args.workers,not args.no_instrumentation,args.explicit_hf,args.round_local_sum)
    else:
        if (args.block_q,args.block_k)!=(128,256): parser.error('non-default tiles require --tile-schedule')
        kernel=make_fa(args.tile_state,args.fused_state,args.mixed_exp,args.grouped_rows,args.pipeline,not args.no_instrumentation,args.workers)
    artifact=tilelang.engine.lower(kernel,target="hexagon",enable_host_codegen=False,
                                   enable_device_compile=False)
    args.out.write_text(artifact.kernel_source)
    print("EMIT_OK",args.out)
