"""Three-stage GDN DSL. Device delivery and validation are separate gates."""
import argparse
from pathlib import Path
import subprocess
import tilelang
from tilelang import language as T


def gram(nt=1024, hk=16, hv=32, d=128):
    nc = (nt+63)//64
    tp = nc*64
    @T.prim_func
    def k1(K: T.Tensor((hk*tp,d),'float16'), G: T.Tensor((hk*nc*64,64),'float16'),
           KM:T.Tensor((64,d),'float16')):
        with T.Kernel(1,threads=1):
            a = T.alloc_shared((64,d),'float16')
            b = T.alloc_shared((64,d),'float16')
            c = T.alloc_shared((64,64),'float16')
            for h in T.serial(hk):
                for ch in T.serial(nc):
                    for i in T.serial(64):
                        for j in T.vectorized(d):
                            KM[i,j] = T.if_then_else(ch*64+i < nt,K[h*tp+ch*64+i,j],T.float16(0))
                    T.copy(KM,a)
                    T.copy(KM,b)
                    T.gemm(a,b,c,transpose_B=True,clear_accum=True)
                    T.copy(c,G[(h*nc+ch)*64:(h*nc+ch+1)*64,:])
    return k1


def wy(nt=1024,hk=16,hv=32,d=128):
    nc=(nt+63)//64
    tp=nc*64
    nw=6
    @T.prim_func
    def k2(K:T.Tensor((hk,tp,d),'float16'), V:T.Tensor((hv,tp,d),'float16'),
           g:T.Tensor((hv,tp),'float32'), beta:T.Tensor((hv,tp),'float32'),
           G:T.Tensor((hk*nc*64,64),'float16'), U:T.Tensor((hv,nc,64,d),'float16'),
           W:T.Tensor((hv,nc,64,d),'float16'), P:T.Tensor((hv,nc,64),'float32'),
           L:T.Tensor((nw,64,64),'float32'), UF:T.Tensor((nw,64,d),'float32'),
           WF:T.Tensor((nw,64,d),'float32'), E:T.Tensor((hv,nc,64),'float32'),
           X:T.Tensor((nw,64),'float32')):
        with T.Kernel(1,threads=nw):
            lane=T.get_thread_binding()
            for wave in T.serial((hv*nc+nw-1)//nw):
                job=wave*nw+lane
                if job<hv*nc:
                    h=job//nc
                    ch=job%nc
                    kh=h%hk
                    for i in T.serial(64):
                        P[h,ch,i]=T.if_then_else(i>0,P[h,ch,i-1],T.float32(0))+T.if_then_else(ch*64+i<nt,g[h,ch*64+i],T.float32(0))
                    for i in T.vectorized(64):
                        E[h,ch,i]=T.exp(P[h,ch,i])
                    for i in T.serial(64):
                        for j in T.vectorized(64):
                            X[lane,j]=P[h,ch,i]-P[h,ch,j]
                        for j in T.vectorized(64):
                            L[lane,i,j]=T.if_then_else(j<i,T.exp(X[lane,j]),T.float32(0))
                        for j in T.vectorized(64):
                            L[lane,i,j]=L[lane,i,j]*T.if_then_else(ch*64+i<nt,beta[h,ch*64+i],T.float32(0))*T.Cast('float32',G[(kh*nc+ch)*64+i,j])
                    for i in T.serial(64):
                        for j in T.vectorized(d):
                            UF[lane,i,j]=T.if_then_else(ch*64+i<nt,beta[h,ch*64+i]*T.Cast('float32',V[h,ch*64+i,j]),T.float32(0))
                            WF[lane,i,j]=T.if_then_else(ch*64+i<nt,beta[h,ch*64+i]*E[h,ch,i]*T.Cast('float32',K[kh,ch*64+i,j]),T.float32(0))
                        for j in T.serial(i):
                            for col in T.vectorized(d):
                                UF[lane,i,col] -= L[lane,i,j]*UF[lane,j,col]
                                WF[lane,i,col] -= L[lane,i,j]*WF[lane,j,col]
                        for col in T.vectorized(d):
                            U[h,ch,i,col]=T.Cast('float16',UF[lane,i,col])
                            W[h,ch,i,col]=T.Cast('float16',WF[lane,i,col])
    return k2


def main_stage(nt=1024,hk=16,hv=32,d=128):
    nc=(nt+63)//64
    tp=nc*64
    @T.prim_func
    def k3(Q:T.Tensor((hk*tp,d),'float16'),K:T.Tensor((hk*tp,d),'float16'),
           U:T.Tensor((hv*nc*64,d),'float16'),W:T.Tensor((hv*nc*64,d),'float16'),
           P:T.Tensor((hv*nc,64),'float32'),E:T.Tensor((hv*nc,64),'float32'),
           S0:T.Tensor((hv*d,d),'float32'),S1:T.Tensor((hv*d,d),'float32'),
           O:T.Tensor((hv*tp,d),'float16'),S:T.Tensor((d,d),'float32'),
           ST:T.Tensor((d,d),'float32'),SH:T.Tensor((d,d),'float16'),
           QR:T.Tensor((64,d),'float16'),KR:T.Tensor((64,d),'float16'),
           WS:T.Tensor((64,d),'float16'),QS:T.Tensor((64,d),'float16'),
           QK:T.Tensor((64,64),'float16'),PR:T.Tensor((64,d),'float16'),
           R:T.Tensor((64,d),'float16'),RT:T.Tensor((d,64),'float16'),
           PT:T.Tensor((64,64),'float16'),EX:T.Tensor((64,64),'float32'),
           KE:T.Tensor((64,),'float32'),KF:T.Tensor((64,d),'float32'),
           RF:T.Tensor((64,d),'float32'),LOG:T.Tensor((34,),'uint64')):
        with T.Kernel(1,threads=1):
            T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),-1))
            a=T.alloc_shared((64,d),'float16')
            b=T.alloc_shared((d,d),'float16')
            c=T.alloc_shared((64,d),'float16')
            k=T.alloc_shared((64,d),'float16')
            p=T.alloc_shared((64,64),'float16')
            r=T.alloc_shared((d,64),'float16')
            for h in T.serial(hv):
                T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),1))
                T.copy(S0[h*d:(h+1)*d,:],S)
                for ch in T.serial(nc):
                    job=h*nc+ch
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),2))
                    T.transpose(S,ST)
                    for row in T.serial(d):
                        for col in T.vectorized(d):
                            SH[row,col]=T.Cast('float16',ST[row,col])
                    T.copy(SH,b)
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),3))
                    T.copy(W[job*64:(job+1)*64,:],a)
                    T.gemm(a,b,c,transpose_B=True,clear_accum=True)
                    T.copy(c,WS)
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),4))
                    for i in T.serial(64):
                        for col in T.vectorized(d):
                            R[i,col]=U[job*64+i,col]-WS[i,col]
                    T.transpose(R,RT)
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),5))
                    for i in T.serial(64):
                        for col in T.vectorized(d):
                            QR[i,col]=T.if_then_else(ch*64+i<nt,Q[(h%hk)*tp+ch*64+i,col],T.float16(0))
                            KR[i,col]=T.if_then_else(ch*64+i<nt,K[(h%hk)*tp+ch*64+i,col],T.float16(0))
                    T.copy(QR,a)
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),6))
                    T.gemm(a,b,c,transpose_B=True,clear_accum=True)
                    T.copy(c,QS)
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),7))
                    T.copy(KR,k)
                    T.gemm(a,k,p,transpose_B=True,clear_accum=True)
                    T.copy(p,QK)
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),8))
                    for i in T.serial(64):
                        for j in T.vectorized(64):
                            EX[i,j]=P[job,i]-P[job,j]
                        for j in T.vectorized(64):
                            EX[i,j]=T.if_then_else(j<=i,T.exp(EX[i,j]),T.float32(0))
                        for j in T.vectorized(64):
                            PT[i,j]=T.Cast('float16',T.Cast('float32',QK[i,j])*EX[i,j])
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),9))
                    T.copy(PT,p)
                    T.copy(RT,r)
                    T.gemm(p,r,c,transpose_B=True,clear_accum=True)
                    T.copy(c,PR)
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),10))
                    for i in T.serial(64):
                        for col in T.vectorized(d):
                            O[h*tp+ch*64+i,col]=T.if_then_else(ch*64+i<nt,T.Cast('float16',(E[job,i]*T.Cast('float32',QS[i,col])+T.Cast('float32',PR[i,col]))*T.float32(d**-.5)),T.float16(0))
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),11))
                    for i in T.vectorized(64):
                        KE[i]=P[job,63]-P[job,i]
                    for i in T.vectorized(64):
                        KE[i]=T.exp(KE[i])
                    for i in T.serial(64):
                        for col in T.vectorized(d):
                            KF[i,col]=T.Cast('float32',KR[i,col])*KE[i]
                            RF[i,col]=T.Cast('float32',R[i,col])
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),12))
                    for row in T.serial(d):
                        for col in T.vectorized(d):
                            S[row,col]=S[row,col]*E[job,63]
                    T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),13))
                    for row in T.serial(d):
                        for i in T.serial(64):
                            for col in T.vectorized(d):
                                S[row,col]+=KF[i,row]*RF[i,col]
                T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),14))
                T.copy(S,S1[h*d:(h+1)*d,:])
            T.evaluate(T.call_extern('void','tl::profile_mark',T.address_of(LOG[0]),0))
    # This entry is built only with the checked nonoverlapping slab registry.
    # Register the profiling operation's audited bounded effect instead of
    # attaching a synchronization promise to arbitrary call_extern names.
    from tvm import tirx
    from tvm.ir import Op
    import importlib.util
    spec = importlib.util.spec_from_file_location('gdn_slab_layout', Path(__file__).with_name('layout.py'))
    slab_layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slab_layout)
    def typed_profile(node):
        if isinstance(node, tirx.Call) and node.op == Op.get('tirx.call_extern') and node.args[0].value == 'tl::profile_mark':
            return tirx.Call(node.dtype, Op.get('tl.hexagon.profile_mark'), node.args[1:])
        return node
    k3 = k3.with_body(tirx.stmt_functor.ir_transform(k3.body, None, typed_profile))
    return slab_layout.checked_noalias(k3, nt, hk, hv, d)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--t',type=int,default=1024)
    p.add_argument('--hk',type=int,default=16)
    p.add_argument('--hv',type=int,default=32)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--ordered-accumulator',action='store_true')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    root=Path(tilelang.__file__).resolve().parents[1]
    cc='/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++'
    for factory in (gram,wy,main_stage):
        f=factory(a.t,a.hk,a.hv)
        from tvm.ir.transform import PassContext
        with PassContext(config={'tl.hexagon.affine_transpose':True,
                                 'tl.enable_ordered_accumulator_promotion':a.ordered_accumulator}):
            source=tilelang.engine.lower(f,target='hexagon',enable_host_codegen=False,enable_device_compile=False).kernel_source
        path=a.out/(factory.__name__+'.cpp')
        path.write_text(source)
        flags=[cc,'-mv79','-mhvx','-mhvx-length=128B','-mhmx','-O2','-fPIC','-std=c++17','-I'+str(root/'src')]
        if a.profile:flags+=['-DTL_HEX_REGION_PROFILE=1']
        subprocess.run(flags+['-c',str(path),'-o',str(path.with_suffix('.o'))],check=True)
        subprocess.run(flags+['-S',str(path),'-o',str(path.with_suffix('.s'))],check=True)
        print('ACTUAL_COMPILE_OK',path,flush=True)


if __name__=='__main__':
    main()
