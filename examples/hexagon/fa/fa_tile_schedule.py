"""Parametric persistent online attention using ordinary DSL operations.

Six is not special: paired rows have cyclic worker ownership. Full packed
K/V remain resident for a KV-head job; only subregions feed each GEMM.
"""
import tilelang
from tilelang import language as T
from tilelang.hexagon.language.hf import exp2_hf, sub_hf


def pair_row(iteration, worker, workers):
    """Shared schedule expression, used by both DSL and ownership tests."""
    return 2 * (iteration * workers + worker)


def key_blocks(query_block, block_q, block_k):
    return ((query_block + 1) * block_q + block_k - 1) // block_k


def make_fa(block_q=128, block_k=256, workers=4, instrumentation=False,
            explicit_hf=False, round_local_sum=False):
    if round_local_sum and not explicit_hf:
        raise ValueError('local HF sum rounding requires the explicit HF variant')
    if workers < 1 or any(x < 32 or x % 32 or 1024 % x for x in (block_q,block_k)):
        raise ValueError('positive workers and 32-multiple tiles dividing 1024 required')
    local_rows=2*((block_q//2+workers-1)//workers)
    @T.macro
    def profile(region):
        if instrumentation:
            T.evaluate(T.call_extern('handle','tl::profile_region',region))
    @T.macro
    def stage(src,dst,base,rows,tx):
        buf=T.alloc_local((2,256),'float16')
        for it in T.serial(T.ceildiv(rows//2,workers)):
            if it*workers+tx < rows//2:
                for r in T.serial(2):
                    for j in T.vectorized(256):
                        buf[r,j]=src[base+pair_row(it,tx,workers)+r,j]
                T.copy(buf,dst[pair_row(it,tx,workers):pair_row(it,tx,workers)+2,:])
        T.sync_threads()
    @T.prim_func
    def fa(Q:T.Tensor((16384,256),'float16'),K:T.Tensor((4096,256),'float16'),
           V:T.Tensor((4096,256),'float16'),O:T.Tensor((16384,256),'float16')):
        with T.Kernel(4,threads=workers) as kh:
            tx=T.get_thread_binding()
            q=T.alloc_shared((block_q,256),'float16')
            kr=T.alloc_shared((1024,256),'float16')
            vr=T.alloc_shared((1024,256),'float16')
            k=T.alloc_shared((1024,256),'float16')
            v=T.alloc_shared((block_k,256),'float16')
            vt=T.alloc_shared((256,1024),'float16')
            score=T.alloc_shared((block_q,block_k),'float16')
            prob=T.alloc_shared((block_q,block_k),'float16')
            partial=T.alloc_shared((block_q,256),'float16')
            rowh=T.alloc_local((2,block_k),'float16')
            rowf=T.alloc_local((2,block_k),'float32')
            rowo=T.alloc_local((2,256),'float16')
            out=T.alloc_local((local_rows,256),'float32')
            maximum=T.alloc_local((local_rows,),'float32')
            denominator=T.alloc_local((local_rows,),'float32')
            correction=T.alloc_local((local_rows,),'float32')
            mx=T.alloc_local((2,),'float32')
            sm=T.alloc_local((2,),'float32')
            profile(2)
            stage(K,kr,kh*1024,1024,tx)
            stage(V,vr,kh*1024,1024,tx)
            T.copy(kr,k)
            T.transpose(vr,vt)
            for h in T.serial(4):
                for qb in T.serial(1024//block_q):
                    profile(1)
                    stage(Q,q,(kh*4+h)*1024+qb*block_q,block_q,tx)
                    for i in T.serial(local_rows):
                        maximum[i]=-T.infinity('float32')
                        denominator[i]=0.0
                        for j in T.vectorized(256):
                            out[i,j]=0.0
                    for kb in T.serial(key_blocks(qb,block_q,block_k)):
                        profile(3)
                        T.gemm(q,k[kb*block_k:(kb+1)*block_k,:],score,transpose_B=True,clear_accum=True)
                        for it in T.serial(T.ceildiv(block_q//2-tx,workers)):
                            profile(9)
                            T.copy(score[pair_row(it,tx,workers):pair_row(it,tx,workers)+2,:],rowh)
                            for r in T.serial(2):
                                for j in T.vectorized(block_k):
                                    if explicit_hf:
                                        # Base-2 scores, with a visible HF rounding boundary.
                                        # Q/O remain F16 here, unlike upstream's F32 ABI.
                                        rowf[r,j]=T.Select(kb*block_k+j<=qb*block_q+pair_row(it,tx,workers)+r,T.cast(T.cast(T.cast(rowh[r,j],'float32')*0.09016844005556021,'float16'),'float32'),-T.infinity('float32'))
                                    else:
                                        rowf[r,j]=T.Select(kb*block_k+j<=qb*block_q+pair_row(it,tx,workers)+r,T.cast(rowh[r,j],'float32')*0.0625,-T.infinity('float32'))
                            profile(10)
                            T.reduce_max(rowf,mx,dim=1)
                            for r in T.serial(2):
                                profile(14)
                                mx[r]=T.max(mx[r],maximum[it*2+r])
                                if explicit_hf:
                                    correction[it*2+r]=T.cast(exp2_hf(T.cast(maximum[it*2+r]-mx[r],'float16')),'float32')
                                else:
                                    correction[it*2+r]=T.exp(maximum[it*2+r]-mx[r])
                                maximum[it*2+r]=mx[r]
                                profile(15)
                                for chunk in T.serial(T.ceildiv(T.min(block_k,T.max(0,qb*block_q+2*(it*workers+tx)+r-kb*block_k+1)),32)):
                                    for lane in T.vectorized(32):
                                        if explicit_hf:
                                            rowf[r,chunk*32+lane]=T.cast(exp2_hf(sub_hf(T.cast(rowf[r,chunk*32+lane],'float16'),T.cast(mx[r],'float16'))),'float32')
                                        else:
                                            rowf[r,chunk*32+lane]=T.exp(rowf[r,chunk*32+lane]-mx[r])
                                for chunk in T.serial(T.ceildiv(T.min(block_k,T.max(0,qb*block_q+2*(it*workers+tx)+r-kb*block_k+1)),32),block_k//32):
                                    for lane in T.vectorized(32):
                                        rowf[r,chunk*32+lane]=0.0
                            profile(12)
                            T.reduce_sum(rowf,sm,dim=1)
                            if round_local_sum:
                                for r in T.serial(2):
                                    sm[r]=T.cast(T.cast(sm[r],'float16'),'float32')
                            profile(13)
                            for r in T.serial(2):
                                denominator[it*2+r]=denominator[it*2+r]*correction[it*2+r]+sm[r]
                                for j in T.vectorized(block_k):
                                    rowh[r,j]=T.cast(rowf[r,j],'float16')
                            T.copy(rowh,prob[pair_row(it,tx,workers):pair_row(it,tx,workers)+2,:])
                        T.sync_threads()
                        profile(5)
                        T.gemm(prob,vt[:,kb*block_k:(kb+1)*block_k],partial,transpose_B=True,clear_accum=True)
                        profile(6)
                        for it in T.serial(T.ceildiv(block_q//2-tx,workers)):
                            T.copy(partial[pair_row(it,tx,workers):pair_row(it,tx,workers)+2,:],rowo)
                            for r in T.serial(2):
                                for j in T.vectorized(256):
                                    out[it*2+r,j]=out[it*2+r,j]*correction[it*2+r]+T.cast(rowo[r,j],'float32')
                        T.sync_threads()
                    profile(7)
                    for it in T.serial(T.ceildiv(block_q//2-tx,workers)):
                        for r in T.serial(2):
                            denominator[it*2+r]=1.0/denominator[it*2+r]
                            for j in T.vectorized(256):
                                O[(kh*4+h)*1024+qb*block_q+pair_row(it,tx,workers)+r,j]=T.cast(out[it*2+r,j]*denominator[it*2+r],'float16')
                    T.sync_threads()
            profile(0)
    return fa.with_attr('global_symbol','fa_causal_persistent_rm_b1_h16_g4_s1024_d256')
