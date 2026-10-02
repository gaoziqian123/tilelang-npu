"""One isolated detached attempt; root ps gate, hashes, true done and full check.

Retention: input/gold + first/last raw outputs; SHA256 all iterations before
deleting middle phone dumps. Host runner checks full O/S1 outside every timer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import re
import subprocess
import time
import numpy as np
from reference import metrics

def command(argv):
    print('COMMAND',shlex.join(list(map(str,argv))),flush=True)
    return subprocess.check_output(list(map(str,argv)),text=True)

def busy(ps):
    # Match userspace executable names, not kernel glink-LOOPBACK_CTL_* names.
    result=[]
    for line in ps.splitlines():
        cols=line.split(None,4)
        if len(cols)<4 or cols[3].startswith('['):continue
        if re.search(r'(^|/)(runner|llama[^ /]*|sched_rt|gdn_[^ /]*|tl_[^ /]*|qwen35_flow)( |$)',cols[3]):result.append(line)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--iters',type=int,default=1);a=p.parse_args();out=a.out
    cfg=json.loads((out/'layout.json').read_text())
    tag='gdn3_'+str(time.time_ns());parent='/data/data/com.termux/files/home';remote=parent+'/'+tag
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','oneplus13-reverse']
    ps=command(ssh+['su -c "ps -A -o USER,PID,PPID,NAME,ARGS"'])
    (out/(tag+'.rootps')).write_text(ps)
    suspects=busy(ps)
    if suspects:raise RuntimeError('busy/ambiguous device: '+ '\n'.join(suspects))
    print(command(ssh+[f'ls -ld {parent} && df -h {parent}']))
    print(command(ssh+['mkdir '+remote]))
    names=['runner','libattnops_skel.so','input.bin','gold_o.bin','gold_s.bin']
    print(command(['scp',*[out/n for n in names],'oneplus13-reverse:'+remote+'/']))
    rpc=out/'libcdsprpc.so'
    print(command(['scp','oneplus13-reverse:'+parent+'/attn-hexkl/lib/libcdsprpc.so',rpc]))
    print(command(['scp',rpc,'oneplus13-reverse:'+remote+'/']))
    rpchash=command(ssh+['sha256sum '+remote+'/libcdsprpc.so']);print(rpchash)
    assert rpchash.split()[0]==hashlib.sha256(rpc.read_bytes()).hexdigest()
    hashes=command(ssh+['sha256sum '+' '.join(remote+'/'+n for n in names)])
    print(hashes)
    for line in hashes.splitlines():
        digest,path=line.split();assert digest==hashlib.sha256((out/Path(path).name).read_bytes()).hexdigest()
    # Recheck immediately before launch. Do not infer idle from a canceled task.
    ps=command(ssh+['su -c "ps -A -o USER,PID,PPID,NAME,ARGS"'])
    (out/(tag+'.launch.rootps')).write_text(ps)
    suspects=busy(ps)
    if suspects:raise RuntimeError('busy/ambiguous device: '+ '\n'.join(suspects))
    args=f'{cfg["size"]} {a.iters} {cfg["hv"]} {cfg["t"]} {cfg["tp"]} {cfg["offsets"]["O"]} {cfg["offsets"]["S1"]} 0'
    body=f'cd {remote}; chmod 700 runner; export LD_LIBRARY_PATH={remote}:/system/lib64:/vendor/lib64; export ADSP_LIBRARY_PATH={remote}; ./runner {args} > {tag}.log 2>&1; rc=$?; echo "$rc" > {tag}.done'
    print(command(ssh+['nohup sh -c '+shlex.quote(body)+' </dev/null >/dev/null 2>&1 &']))
    for attempt in range(60):
        time.sleep(5)
        status=command(ssh+[f'if test -f {remote}/{tag}.done; then cat {remote}/{tag}.done; else echo PENDING; fi']).strip()
        if status!='PENDING':break
    else:raise RuntimeError('pending; not killed')
    print('TAG_DONE',tag,status)
    print(command(['scp',f'oneplus13-reverse:{remote}/{tag}.log',f'oneplus13-reverse:{remote}/{tag}.done',out]))
    print((out/(tag+'.log')).read_text())
    if status!='0':raise RuntimeError('phone exit '+status)
    outputnames=[f'output_{i:03d}.bin' for i in range(a.iters)]
    outputhashes=command(ssh+['sha256sum '+' '.join(remote+'/'+n for n in outputnames)])
    (out/(tag+'.outputs.sha256')).write_text(outputhashes)
    print(outputhashes)
    if a.iters>1:
        print(command(['scp',f'oneplus13-reverse:{remote}/'+outputnames[-1],out/(tag+'.last.bin')]))
    if a.iters>2:
        print(command(ssh+['rm -- '+' '.join(remote+'/'+n for n in outputnames[1:-1])]))
    print(command(['scp',f'oneplus13-reverse:{remote}/output_000.bin',out]))
    raw=np.fromfile(out/'output_000.bin',np.uint8);src=np.fromfile(out/'input.bin',np.uint8)
    for name in ('Q','K','V','g','beta','S0'):
        shape,dtype=cfg['specs'][name];start=cfg['offsets'][name];n=int(np.prod(shape))*np.dtype(dtype).itemsize
        assert np.array_equal(src[start:start+n],raw[start:start+n]),name+' input changed'
    for name,file in [('O','gold_o.bin'),('S1','gold_s.bin')]:
        shape,dtype=cfg['specs'][name];x=np.ndarray(shape,dtype,raw,cfg['offsets'][name])
        if name=='O':x=x[:,:cfg['t']]
        y=np.fromfile(out/file,np.float64).reshape(x.shape)
        finite,cos,rel=metrics(x,y)
        print(f'DEVICE_CHECK {name} finite={finite} cos={cos:.12f} rms_rel={rel:.6g}',flush=True)
        assert finite and cos>=.999
    print('SMOKE_FULL_CHECK_PASS',hashlib.sha256(raw).hexdigest(),flush=True)

if __name__=='__main__':main()
