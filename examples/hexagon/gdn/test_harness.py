"""Offline fixture contract tests; never claims DSP correctness."""
import json
import sys
from pathlib import Path
import numpy as np
from reference import inputs, recurrent

root=Path(sys.argv[1])
for shape in ('smoke','standard'):
    for mode in range(4):
        p=root/'fixtures'/f'{shape}-mode{mode}'
        meta=json.loads((p/'layout.json').read_text())
        nt,hk,hv,tp=(meta[k] for k in ('t','hk','hv','tp'))
        slab=np.fromfile(p/'input.bin',np.uint8)
        mask=np.fromfile(p/'guard.bin',np.uint8)
        expected=np.ones(len(slab),np.uint8)
        data=inputs(nt,hk,hv,mode=mode,seed=meta['seed'])
        for name,(dims,dtype) in meta['specs'].items():
            off=meta['offsets'][name]
            view=np.ndarray(dims,dtype,slab,off)
            expected[off:off+view.nbytes]=1 if name in ('Q','K','V','g','beta','S0') else 0
            if name in ('O','S1'):
                assert np.isnan(view).all()
                expected[off:off+view.nbytes]=2
        np.ndarray((hv,tp,128,2),np.uint8,expected,meta['offsets']['O'])[:,nt:]=3
        np.testing.assert_array_equal(mask,expected)
        for name,arr in zip(('Q','K','V','g','beta','S0'),data):
            view=np.ndarray(*meta['specs'][name],buffer=slab,offset=meta['offsets'][name])
            np.testing.assert_array_equal(view if name=='S0' else view[:,:nt],arr)
            if name!='S0':assert np.isnan(view[:,nt:]).all()
        assert np.any(data[-1]!=0)
        gold=recurrent(*data)
        for name,arr in zip(('gold_o.bin','gold_s.bin'),gold):
            np.testing.assert_array_equal(np.fromfile(p/name,np.float64),arr.ravel())
        print(f'FIXTURE_PASS {shape} mode={mode} seed={meta["seed"]} mapping=mod independent_FP64 nonzeroS0 poison guard',flush=True)
    for name in ('input.bin','guard.bin','gold_o.bin','gold_s.bin'):
        assert (root/(shape+'-off')/name).read_bytes()==(root/(shape+'-on')/name).read_bytes(),name
    seals=[json.loads((root/(shape+'-'+v)/'seal.json').read_text()) for v in ('off','on')]
    for path in seals[0].keys() & seals[1].keys():
        assert seals[0][path]==seals[1][path],path
    print(f'PAIR_PASS {shape} off/on identical input/gold/guard and shared source/toolchain seals (binaries separately hashed)',flush=True)
