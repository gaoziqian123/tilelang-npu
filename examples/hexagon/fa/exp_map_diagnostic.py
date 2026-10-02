"""Generic local row exp map; diagnostic only, not an attention replacement."""
import argparse
from pathlib import Path
import tilelang
from tilelang import language as T

@T.prim_func
def exp_map_diagnostic(X:T.Tensor((2,256),'float32'),
                       M:T.Tensor((2,),'float32'),
                       Y:T.Tensor((2,256),'float32')):
    with T.Kernel(1,threads=1):
        for r in T.serial(2):
            for j in T.vectorized(256):
                Y[r,j]=T.exp(X[r,j]-M[r])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=tilelang.engine.lower(exp_map_diagnostic,target='hexagon',
        enable_host_codegen=False,enable_device_compile=False)
    a.out.write_text(result.kernel_source)
    print(result.kernel_source)
