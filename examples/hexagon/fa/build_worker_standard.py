"""Emit the full fixed-shape v4 ABI package; no RPC or device execution."""
import argparse
import concurrent.futures
import hashlib
import json
import re
from pathlib import Path
import subprocess
import tilelang
from tilelang import language as T
from tilelang.hexagon.physical_contract import compact_contract, c_packer
from fa_worker_pipeline import make_fa, packed


def build(output, compiler, intrinsic_rows=True, compile_timeout=900, state_workers=1, guard_diagnostics=False):
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be fresh/empty; refusing to overwrite artifacts')
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for qb in range(8):
        try:
            lowered = tilelang.engine.lower(make_fa(qb, intrinsic_rows, state_workers), target='hexagon',
                                           enable_host_codegen=False, enable_device_compile=False)
        except Exception as exc:
            (output/'manifest.json').write_text(json.dumps(dict(
                status='BLOCKED', query_block=qb, intrinsic_rows=intrinsic_rows, state_workers=state_workers,
                error=str(exc), device_validation='UNVERIFIED', cases=cases), indent=2)+'\n')
            raise
        path = output / f'qb{qb}.c'
        path.write_text(lowered.kernel_source)
        funcs = [f for f in lowered.device_mod.functions.values()
                 if f.attrs and 'tl.workergroup_groups' in f.attrs]
        assert len(funcs) == 1
        f = funcs[0]
        args = [{'name': str(p.name), 'type': str(p.type_annotation)} for p in f.params[:-3]]
        assert len(args) == 4
        symbol = str(f.attrs['global_symbol'])
        vtcm = int(f.attrs['tl.workergroup_vtcm_bytes'])
        assert vtcm + 768 <= 4*1024*1024
        cases.append(dict(query_block=qb, symbol=symbol, parameters=args,
                          panels=(qb+2)//2, vtcm_bytes=vtcm,
                          iterations=int(f.attrs['tl.workergroup_iteration_count'])))
        print('EMIT', qb, symbol, args, 'VTCM', vtcm, flush=True)

    # Physical map for a logical V[K,D] view is the composition of logical
    # transpose and the exact WH Layout used by the kernel V[D,K] parameter.
    vl = T.Layout((1024,256), lambda k,d:(d//32,k//32,k%32//2,(d%32)*2+k%2))
    specs = {name: compact_contract(layout) for name,layout in
             [('Q',packed(128,256)), ('K',packed(1024,256,True)), ('V',vl)]}
    (output/'pack.h').write_text('#pragma once\n' + '\n'.join(
        c_packer(spec, 'fa_pack_'+name) for name,spec in specs.items()))
    prototypes=[]
    for case in cases:
        names=[p['name'] for p in case['parameters']]
        assert set(names)=={'Q','K','V','O'}, names
        prototypes.append('int '+case['symbol']+'_owner(const tl_wg_caps*,void*,void*,void*,uint64_t,uint64_t,'+
                          ','.join('__fp16* '+name for name in names)+');')
    header = '#pragma once\n#include <tl_templates/hexagon/workergroup_abi.h>\nextern "C" {\n'+'\n'.join(prototypes)+'\n'
    signature='int fa_standard_v4(const tl_wg_caps *caps,void *vtcm,void *sync,uint64_t sync_bytes,__fp16 *Q,uint64_t Qbytes,__fp16 *K,uint64_t Kbytes,__fp16 *V,uint64_t Vbytes,__fp16 *O,uint64_t Obytes)'
    header+=signature+';\n}\n'
    (output/'dispatcher.h').write_text(header)
    code='#include "dispatcher.h"\n'+signature+' {\n'
    code+='if(!caps || caps->abi_version!=4 || !vtcm || !sync) return TL_WG_INVALID;\n'
    for name,size in [('Q',8388608),('K',2097152),('V',2097152),('O',8388608)]:
        code+=f'if(!{name} || ((uintptr_t){name}&127u) || {name}bytes<{size}ull || (uintptr_t){name}>UINTPTR_MAX-{size}ull) return TL_WG_INVALID;\n'
    for a,b in [('Q','K'),('Q','V'),('Q','O'),('K','V'),('K','O'),('V','O')]:
        sizes={'Q':8388608,'K':2097152,'V':2097152,'O':8388608}
        code+=f'if((uintptr_t){a}<(uintptr_t){b}+{sizes[b]}ull && (uintptr_t){b}<(uintptr_t){a}+{sizes[a]}ull) return TL_WG_INVALID;\n'
    code+='for(unsigned h=0;h<16;++h) {\n'
    for case in cases:
        qb=case['query_block']
        pointers={'Q':f'Q+(h*8+{qb})*32768ull', 'K':'K+(h/4)*262144ull',
                  'V':'V+(h/4)*262144ull', 'O':f'O+(h*8+{qb})*32768ull'}
        call=','.join(pointers[p['name']] for p in case['parameters'])
        code+=f'{{int rc={case["symbol"]}_owner(caps,0,vtcm,sync,sync_bytes,0,{call}); if(rc) return rc;}}\n'
    code+='}\nreturn 0;\n}\n'
    (output/'dispatcher.cpp').write_text(code)
    manifest=dict(shape=dict(B=1,HQ=16,HKV=4,S=1024,D=256,BQ=128,BK=256),
                  intrinsic_rows=intrinsic_rows, status='BUILDING',
                  device_validation='UNVERIFIED',
                  abi_version=4, cases=cases, inputs=specs, output_layout='row-major',
                  input_contract='Q: 128 query blocks; K/V: four independently packed heads',
                  owner_calls_per_rpc=128, initial_ordinal=0, events=24, physical_workers=2+state_workers,
                  state_workers=state_workers,
                  guard_diagnostics=guard_diagnostics,
                  sync_bytes='runtime tl_wg_sync_bytes; caller registered backing',
                  completion='blocking owner calls on original HMX-open thread; no RPC per block',
                  redundant_input='Q is DMA-copied each key panel; K/V not resident across calls')
    manifest['precision_contract'] = dict(state_dtype='float16', score_scale='half after HMX readout',
        reference='independent FP64', gate='finite and cosine >= 0.999',
        max_relative_error='diagnostic only; user-authorized requirement change',
        input_hash_required=True, nan_poison_required=True,
        math='half log2e 0x3dc5; exp2 ggml_qf16_clamp24_v1; NR2 FP32 reciprocal rounded to half then half multiply; no FP32 cross-panel state')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    root=Path(__file__).resolve().parents[3]
    flags=[compiler,'-O2','-mv79','-mhvx','-mhvx-length=128B','-mhmx','-std=c++17','-I',str(root/'src'),'-x','c++']
    if guard_diagnostics:
        flags += ['-DTL_HEX_GUARD_DIAGNOSTICS=1', '-g']
        manifest['diagnostic_sink'] = 'runtime must define extern C void tl_hex_guard_failure(unsigned id); failed guard still traps'
        manifest['guard_ids'] = {'1001': 'exp2 nonpositive/NaN', '1002': 'positive bounded reciprocal'}
    def compile_one(path):
        for mode,suffix in [('-c','.o'),('-S','.s')]:
            command=flags+[mode,'-fstack-usage',str(path),'-o',str(path.with_suffix(suffix))]
            print('COMMAND',' '.join(command),flush=True)
            subprocess.run(command,check=True,timeout=compile_timeout)
        print('COMPILED',path.name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(compile_one, [output/f'qb{i}.c' for i in range(8)]+[output/'dispatcher.cpp']))
    subprocess.run([compiler,'-mv79','-r']+[str(output/f'qb{i}.o') for i in range(8)]+
                   [str(output/'dispatcher.o'),'-o',str(output/'fa_standard.o')],check=True)
    manifest['status'] = 'OBJECTS_BUILT_NOT_DEVICE_VALIDATED'
    manifest['scalar_math_calls'] = {f'qb{i}': sorted(set(re.findall(
        r'\bcall\s+([^\s;{}]+)', (output/f'qb{i}.s').read_text()))) for i in range(8)}
    manifest['scalar_math_calls'] = {qb: [s for s in calls if re.search(
        r'exp|div|recip', s, re.I)] for qb, calls in manifest['scalar_math_calls'].items()}
    manifest['no_scalar_math_calls'] = not any(manifest['scalar_math_calls'].values())
    manifest['build_flags'] = flags
    manifest['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), Path(__file__).with_name('fa_worker_pipeline.py'))}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    (output/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('COMPLETE',output,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compiler',default='/root/hexagon-deps/HEXAGON_TOOLS/Tools/bin/hexagon-clang++')
    parser.add_argument('--intrinsic-rows',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--compile-timeout',type=int,default=900)
    parser.add_argument('--state-workers',type=int,choices=(1,2),default=1)
    parser.add_argument('--guard-diagnostics',action='store_true')
    args=parser.parse_args()
    build(args.output,args.compiler,args.intrinsic_rows,args.compile_timeout,args.state_workers,args.guard_diagnostics)
