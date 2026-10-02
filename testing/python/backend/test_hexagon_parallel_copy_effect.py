"""TileOp write regions must participate in parallel ownership lowering."""
from pathlib import Path
import subprocess
import tilelang
from tilelang import language as T
from tilelang.hexagon.language import vector as V


def test_parallel_private_vector_store(tmp_path):
    partition = T.Fragment((128,), forward_thread_fn=lambda r: r % 2,
                           forward_index_fn=lambda r: r // 2)

    @T.prim_func
    def repro(A: T.Tensor((128,256), 'float16'), O: T.Tensor((128,256), 'float16')):
        with T.Kernel(1, threads=2):
            row = T.alloc_local((256,), 'float16')
            for r in T.Parallel(128, loop_layout=partition):
                T.copy(A[r:r+1,:], row)
                x = V.load(row, 0)
                T.evaluate(V.store(row, x, 0))
                T.copy(row, O[r:r+1,:])

    src = tilelang.engine.lower(repro, target='hexagon', enable_host_codegen=False,
                                enable_device_compile=False).kernel_source
    cpp = tmp_path/'kernel.cpp'; cpp.write_text(src)
    root = Path(__file__).resolve().parents[3]
    def run(cmd):
        print('COMMAND:', ' '.join(map(str,cmd)))
        r = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
        print(r.stdout+r.stderr)
        assert r.returncode == 0
    run(['/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++', '-mv79',
         '-mhvx', '-mhvx-length=128B', '-O2', '-std=c++17', '-I'+str(root/'src'),
         '-c', str(cpp), '-o', str(tmp_path/'kernel.o')])
    # Portable leaf emulation only: execute the UNEDITED generated loop/index
    # code on two pthreads. Each output chunk store is counted atomically.
    headers = tmp_path/'tl_templates/hexagon'; headers.mkdir(parents=True)
    (headers/'common.h').write_text(r'''
#pragma once
#include <stdint.h>
#include <stddef.h>
using half_t = __fp16;
extern thread_local int worker;
inline int tl_hex_num_jobs(){return 1;}
inline int tl_hex_job_id(){return 0;}
inline int tl_hex_num_workers(){return 2;}
inline int tl_hex_worker_id(){return worker;}
namespace tl {
inline void hex_require(bool c){if(!c)__builtin_trap();}
template<size_t N> bool is_aligned(const void*p){return !((uintptr_t)p%N);}
}
''')
    (headers/'vector_leaf.h').write_text(r'''
#pragma once
#include "common.h"
struct alignas(128) HVX_Vector { unsigned char bytes[128]; };
namespace tl { namespace hvx_leaf {
inline HVX_Vector splat16(uint32_t x){HVX_Vector v;for(int i=0;i<64;i++){uint16_t s=x;__builtin_memcpy(v.bytes+2*i,&s,2);}return v;}
inline HVX_Vector load(const void*p,int,HVX_Vector){HVX_Vector v;__builtin_memcpy(&v,p,128);return v;}
inline void store(void*p,HVX_Vector v,int){__builtin_memcpy(p,&v,128);}
}}
''')
    (headers/'hvx.h').write_text(r'''
#pragma once
#include "vector_leaf.h"
extern void record_store(void*);
namespace tl {
template<class T> struct hvx_vec { HVX_Vector raw; };
template<class T> void hvx_store(T*p,hvx_vec<T> v){record_store(p);__builtin_memcpy(p,&v.raw,128);}
}
''')
    harness = tmp_path/'harness.cpp'
    harness.write_text(r'''
#include <atomic>
#include <thread>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
thread_local int worker;
alignas(128) static uint16_t a[128*256],o[128*256];
static std::atomic<unsigned> writes[512];
extern "C" void repro_kernel(__fp16*,__fp16*);
void record_store(void*p){
  auto offset=(uintptr_t)p-(uintptr_t)o;
  assert(offset<sizeof(o) && offset%128==0);
  auto chunk=offset/128;
  assert((chunk/4)%2==(unsigned)worker);
  writes[chunk].fetch_add(1);
}
int main(){
 for(int repeat=0;repeat<16;repeat++){
  for(unsigned i=0;i<128*256;i++){a[i]=i%1024;o[i]=0xffff;}
  for(auto &x:writes)x=0;
  auto run=[](int id){worker=id;repro_kernel((__fp16*)a,(__fp16*)o);};
  std::thread x(run,0),y(run,1);x.join();y.join();
  assert(!memcmp(a,o,sizeof(a)));
  for(auto &count:writes)assert(count==1);
 }
 puts("PASS generated cyclic rows: 2 pthreads, 16 launches, all 512 output chunks exactly once, owner parity, full output");
}
''')
    exe=tmp_path/'run'
    run(['/root/autodl-tmp/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++',
         '--target=x86_64-linux-gnu', '-O2', '-std=c++17', '-pthread', '-Wno-psabi',
         '-I'+str(tmp_path), str(cpp), str(harness), '-o', str(exe)])
    run([str(exe)])
