#!/system/bin/sh
set -u
cd "$(dirname "$0")"
python3 - <<'PY'
import csv, json, os, re, shutil, subprocess
from pathlib import Path

items = json.load(open('manifest.json'))
binary = 'gemm_nt_test'
kernel_name = 'gemm_nt_kernel_kernel'
family_slowdown = 2.0
no_top3_limit = 12
canary_interval = 10
rounds = 2
ld = '.:/system/lib64:/vendor/lib64'

ms_re = re.compile("ms_iter\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)")
logs = Path('logs'); shutil.rmtree(logs, ignore_errors=True); logs.mkdir()
rows = []
state = {'best_ms': None, 'best_id': '', 'top3': [], 'family_seen': set(), 'family_skip': set(), 'measured': 0, 'no_top3': 0, 'early': False, 'canaries': 0}

def run_item(item, r, kind):
    env = os.environ.copy(); env.update({k: str(v) for k, v in item['env'].items()}); env['LD_LIBRARY_PATH'] = ld
    log = logs / f"{len(rows):04d}_{item['id']}_r{r}_{kind}.log"
    with log.open('w') as f:
        rc = subprocess.run([f"./{binary}", item['file'], kernel_name], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    txt = log.read_text(errors='replace')
    m = ms_re.search(txt)
    status = 'PASS' if 'PASS' in txt and 'FAIL' not in txt else ('FAIL' if 'FAIL' in txt else 'UNKNOWN')
    row = {'id': item['id'], 'round': r, 'kind': kind, 'rc': rc, 'log': str(log), 'skipped_reason': '', 'ms': float(m.group(1)) if m else None, 'status': status}
    rows.append(row)
    return row

def add_skip(item, r, reason):
    rows.append({'id': item['id'], 'round': r, 'kind': 'skipped', 'rc': '', 'log': '', 'skipped_reason': reason, 'ms': None, 'status': 'SKIPPED'})

def update_state(item, row):
    if row['kind'] != 'measure':
        return
    state['measured'] += 1
    fam = item['family_key']; cid = item['id']; ms = row['ms']
    entered = False
    if ms is not None and row['status'] == 'PASS' and row['rc'] == 0:
        old = list(state['top3'])
        state['top3'] = sorted(old + [{'id': cid, 'ms': ms}], key=lambda x: x['ms'])[:3]
        entered = any(x['id'] == cid and x['ms'] == ms for x in state['top3'])
        if state['best_ms'] is None or ms < state['best_ms']:
            state['best_ms'] = ms; state['best_id'] = cid
        if fam not in state['family_seen']:
            state['family_seen'].add(fam)
            if state['best_ms'] is not None and ms > state['best_ms'] * family_slowdown:
                state['family_skip'].add(fam)
    state['no_top3'] = 0 if entered else state['no_top3'] + 1
    if state['no_top3'] >= no_top3_limit:
        state['early'] = True

for r in range(rounds):
    seq = items if (r % 2 == 0) else list(reversed(items))
    for item in seq:
        if state['early']:
            add_skip(item, r, 'early_stopped'); continue
        if item['family_key'] in state['family_skip']:
            add_skip(item, r, 'skipped_family'); continue
        row = run_item(item, r, 'measure')
        update_state(item, row)
        if state['best_id'] and state['measured'] % canary_interval == 0:
            best = next(it for it in items if it['id'] == state['best_id'])
            can = run_item(best, r, 'canary')
            state['canaries'] += 1

with open('results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id','round','kind','rc','log','skipped_reason','ms','status'])
    w.writeheader(); w.writerows(rows)
meta = dict(state)
meta['family_seen'] = sorted(meta['family_seen']); meta['family_skip'] = sorted(meta['family_skip'])
json.dump(meta, open('run_meta.json','w'), indent=2, sort_keys=True)
PY
