#!/system/bin/sh
set -u
cd "$(dirname "$0")"
rm -f results.csv
printf 'id,round,rc,log
' > results.csv
rm -rf logs
mkdir -p logs
python3 - <<'PY' > run_items.sh
import json, shlex
items = json.load(open('manifest.json'))
binary = 'gemm_nt_test'
kernel_name = 'gemm_nt_kernel_kernel'
for item in items:
    env = dict(item['env'])
    env['LD_LIBRARY_PATH'] = '.:/system/lib64:/vendor/lib64'
    exports = ' '.join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
    for r in range(2):
        log = f"logs/{item['id']}_r{r}.log"
        cmd = f"{exports} ./{shlex.quote(binary)} {shlex.quote(item['file'])} {shlex.quote(kernel_name)} > {shlex.quote(log)} 2>&1"
        print(f"{cmd}; rc=$?; printf '%s,%s,%s,%s\n' {shlex.quote(item['id'])} {r} $rc {shlex.quote(log)} >> results.csv")
PY
sh run_items.sh
