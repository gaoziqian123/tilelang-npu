"""Shared row state is the supported alternative to unproved fragment layouts.

Planner tests below use actual DSL. The pthread oracle independently exercises
the row ownership/lifetime contract, not generated target code execution.
"""
from pathlib import Path
import subprocess
import pytest
import tilelang
from tilelang import language as T
from tvm import IRModule, tirx, get_global_func
from tvm.target import Target


def planned(shift=0, owner='state'):
    team=2 if owner=='state' else 4
    @T.prim_func
    def kernel(O:T.Tensor((128,), 'float32')):
        with T.Kernel(1,threads=team):
            state=T.alloc_shared((128,), 'float32')
            T.clear(state)
            for k in T.Pipelined(7,num_stages=2,annotations={'tl.workergroup_schedule':'async'}):
                with T.pipeline_stage('produce',engine='hvx',workers=2,physical_owner='state'):
                    for row in T.Parallel(128):
                        state[row]=state[row]+T.float32(1)
                with T.pipeline_stage('consume',engine='hvx',workers=2,physical_owner=owner):
                    for row in T.Parallel(128-shift):
                        O[row]=state[row+shift]
    mod=tirx.transform.BindTarget(Target('hexagon'))(IRModule({'main':kernel}))
    mod=tilelang.transform.MaterializeKernelLaunch()(mod)
    return get_global_func('tl.hexagon.transform.PlanWorkerPipeline')(4,64,8388608)(mod)


def test_shared_same_owner():
    mod=planned()
    get_global_func('tl.hexagon.transform.InferWorkerOwnership')()(mod)


@pytest.mark.parametrize('shift,owner',[(1,'state'),(0,'different')])
def test_changed_ownership_rejected(shift,owner):
    with pytest.raises(ValueError,match='partition|recurrence'):
        planned(shift,owner)


def test_pthread_row_lifetime(tmp_path):
    source=tmp_path/'rows.c'
    source.write_text(r'''
#define _XOPEN_SOURCE 700
#include <pthread.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
static float state[128], result[97][128];
static pthread_barrier_t boundary;
static void *worker(void *arg) {
  unsigned id=(uintptr_t)arg;
  float staging[8]; /* genuinely private, not shared by the two workers */
  for(unsigned k=0;k<97;k++) {
    /* collective producers may use a different partition */
    for(unsigned r=id*64;r<(id+1)*64;r++) state[r]=(float)(k*128+r);
    pthread_barrier_wait(&boundary); /* collective -> row partition */
    for(unsigned r=id;r<128;r+=2) {
      for(unsigned j=0;j<8;j++) staging[j]=state[r]+j;
      state[r]=staging[7];
    }
    /* same worker revisits its rows in the next stage; no cross-worker read */
    for(unsigned r=id;r<128;r+=2) result[k][r]=state[r]*2;
    pthread_barrier_wait(&boundary); /* consumption before next collective */
  }
  return 0;
}
int main(void) {
  for(unsigned repeat=0;repeat<32;repeat++) {
    for(unsigned k=0;k<97;k++) for(unsigned r=0;r<128;r++) result[k][r]=-999;
    assert(!pthread_barrier_init(&boundary,0,2));
    pthread_t a,b;
    assert(!pthread_create(&a,0,worker,(void*)(uintptr_t)0));
    assert(!pthread_create(&b,0,worker,(void*)(uintptr_t)1));
    assert(!pthread_join(a,0));assert(!pthread_join(b,0));
    for(unsigned k=0;k<97;k++) for(unsigned r=0;r<128;r++)
      assert(result[k][r]==(float)((k*128+r+7)*2));
    assert(!pthread_barrier_destroy(&boundary));
  }
  puts("PASS shared rows: 2 pthreads, 97 iterations, 32 repeats, private staging");
}
''')
    exe=tmp_path/'rows'
    cmd=['cc','-O2','-std=c11','-pthread',str(source),'-o',str(exe)]
    print('COMMAND:', ' '.join(cmd))
    r=subprocess.run(cmd,capture_output=True,text=True);print(r.stdout+r.stderr)
    assert r.returncode==0
    print('COMMAND:',str(exe))
    r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=20)
    print(r.stdout+r.stderr);assert r.returncode==0
