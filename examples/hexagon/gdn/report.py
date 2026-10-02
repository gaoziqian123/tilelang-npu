"""Print raw per-run timing summaries and the explicit K3 DDR phase ledger."""
import json
from pathlib import Path
import re
import statistics
import sys
import numpy as np

NAMES=['gap','stateinit','snapshotpack','WS incl pack/readout','residual+transpose',
       'QKload','QS incl readout','QK incl Kpack/readout','causalgate',
       'PR incl pack/readout','QSgate+output','KR external pack','decay','ordered KR','statewrite','unused']
for arg in sys.argv[1:]:
    root=Path(arg)
    for done in sorted(root.glob('gdn3_*.done')):
        log=done.with_suffix('.log').read_text()
        rows=[tuple(map(float,x)) for x in re.findall(r'host_ms=([\d.]+) rpc1=([\d.]+) rpc2=([\d.]+) rpc3=([\d.]+)',log)]
        print('RUN',done.name,'done='+done.read_text().strip(),'n='+str(len(rows)))
        if rows:
            print('ALL_ITER_MEAN host/rpc1/rpc2/rpc3',*[f'{statistics.mean(x):.6f}' for x in zip(*rows)])
            print('HOST min/median/max',min(x[0] for x in rows),statistics.median(x[0] for x in rows),max(x[0] for x in rows))
    cfg=json.loads((root/'layout.json').read_text())
    if cfg.get('profile'):
        raw=np.fromfile(root/'output_000.bin',np.uint8)
        ledger=np.ndarray((34,),'uint64',raw,cfg['offsets']['LOG'])
        for i in sorted(range(16),key=lambda i:-int(ledger[2+2*i])):
            print('K3_PHASE',NAMES[i],'ms=',int(ledger[2+2*i])/1000,'count=',int(ledger[3+2*i]))
        print('K3_LEDGER_SUM_MS',sum(map(int,ledger[2::2]))/1000)
