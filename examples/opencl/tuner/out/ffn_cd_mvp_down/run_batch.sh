#!/system/bin/sh
set -u
cd "$(dirname "$0")"
rm -f results.csv
printf 'id,stage,round,rc,log
' > results.csv
rm -rf logs
mkdir -p logs
python3 - <<'PY' > run_items.sh
import json, shlex
items=json.load(open('manifest.json'))
for item in items:
    for r in range(2):
        env=dict(item['env']); env['LD_LIBRARY_PATH']='.:/system/lib64:/vendor/lib64'
        exports=' '.join(f"{k}={shlex.quote(str(v))}" for k,v in env.items())
        log=f"logs/{item['id']}_r{r}.log"
        cmd=f"{exports} ./ffn_chain_test {shlex.quote(item['gate'])} {shlex.quote(item['up'])} {shlex.quote(item['silu'])} {shlex.quote(item['down'])} > {shlex.quote(log)} 2>&1"
        print(f"{cmd}; rc=$?; printf '%s,%s,%s,%s,%s\n' {shlex.quote(item['id'])} {shlex.quote(item['stage'])} {r} $rc {shlex.quote(log)} >> results.csv")
PY
sh run_items.sh
