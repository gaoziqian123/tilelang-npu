"""Execute unedited C emitted from role TIR on the real pthread adapter."""
from pathlib import Path
import subprocess
import pytest
import tilelang.language as T
from tvm import IRModule, tirx, get_global_func
from tvm.script.ir_builder import IRBuilder


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("trips", [0, 1, 7])
def test_generated_callback(tmp_path, depth, trips):
    shared = tirx.decl_buffer((32,), "float32", scope="shared", name="payload")
    output = tirx.decl_buffer((3, max(1, trips), 32), "float32", name="output")
    with IRBuilder() as builder:
        with T.thread_binding(0, 6, thread="threadIdx.x"):
            with T.serial(0, 3) as outer:
                with T.Pipelined(trips, num_stages=depth, annotations={"tl.workergroup_schedule": "async"}) as k:
                    with T.pipeline_stage("arbitrary_feed", engine="hvx", workers=4):
                        with T.parallel(0, 32) as j:
                            T.buffer_store(shared, T.Cast("float32", outer * 1000 + k * 32 + j), [j])
                    with T.pipeline_stage("unrelated_label", engine="hvx", workers=2):
                        with T.parallel(0, 32) as j:
                            T.buffer_store(output, shared[j] * 3 + 1, [outer, k, j])
    func = tirx.PrimFunc([output.data], builder.get(), buffer_map={output.data: output})
    mod = get_global_func("tl.hexagon.transform.PlanWorkerPipeline")(6, 64, 8192)(IRModule({"main": func}))
    generated = get_global_func("tl.hexagon.EmitWorkerHost")(mod["main"])
    source = tmp_path / "generated.c"
    source.write_text(str(generated))
    harness = tmp_path / "harness.c"
    harness.write_text(f'''
#include "tl_workergroup_pthread.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
extern int tl_generated_worker_run(const tl_wg_caps*,void**,void*,void*,uint64_t,uint64_t);
static unsigned char scratch[8192] __attribute__((aligned(128)));
static unsigned char sync_mem[8192] __attribute__((aligned(128)));
static float output[3][{max(1, trips)}][32];
int main(void) {{
  tl_wg_pthread_pool *pool=tl_wg_pthread_create(6,scratch,sizeof(scratch),0,0,sync_mem,sizeof(sync_mem));
  assert(pool); tl_wg_bind_platform(tl_wg_pthread_platform(pool));
  tl_wg_caps caps={{TL_WG_ABI_VERSION,6,6,64,8192,0,8192}};
  void *args[]={{output}};
  for(unsigned start=0;start<8;++start) {{
    memset(scratch,0xa5,sizeof(scratch));
    for(unsigned i=0;i<sizeof(output)/sizeof(float);++i) ((float*)output)[i]=-999;
    assert(tl_generated_worker_run(&caps,args,scratch,sync_mem,sizeof(sync_mem),start)==0);
    for(unsigned b=0;b<3;++b) for(unsigned k=0;k<{trips};++k) for(unsigned j=0;j<32;++j)
      assert(output[b][k][j]==(float)((b*1000+k*32+j)*3+1));
    if({trips}==0) for(unsigned i=0;i<sizeof(output)/sizeof(float);++i) assert(((float*)output)[i]==-999);
    for(unsigned i={128 * depth};i<sizeof(scratch);++i) assert(scratch[i]==0xa5);
  }}
  tl_wg_bind_platform(0); tl_wg_pthread_destroy(pool);
  puts("PASS generated callback: 8 starts, oracle, poison, canary"); return 0;
}}
''')
    root = Path(__file__).resolve().parents[3]
    backend = root.parent / "backend/npu/attn"
    exe = tmp_path / "run"
    subprocess.run(["cc", "-std=c11", "-O2", "-pthread", "-I", str(root / "src/tl_templates/hexagon"),
                    "-I", str(backend), "-I", str(backend / "skel/src"), str(source), str(harness),
                    str(backend / "tl_workergroup_pthread.c"), str(backend / "skel/src/tl_workergroup_runtime.c"),
                    "-o", str(exe)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=20)
    assert result.stdout == "PASS generated callback: 8 starts, oracle, poison, canary\n"


@pytest.mark.parametrize("option", ["order", "stage", "sync", "group"])
def test_manual_schedule_roles_rejected(option):
    value = [[0]] if option in ("sync", "group") else [0]
    with pytest.raises(ValueError, match="manual software-pipeline"):
        with IRBuilder():
            with T.Pipelined(2, num_stages=2, **{option: value}):
                with T.pipeline_stage("a", engine="hvx", workers=1):
                    T.evaluate(0)
