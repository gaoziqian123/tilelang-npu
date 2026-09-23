from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import sys
import tarfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GEMM_DIR = REPO_ROOT / "examples" / "opencl" / "gemm"
if str(GEMM_DIR) not in sys.path:
    sys.path.insert(0, str(GEMM_DIR))

from gemm_nt import make_grgemm_kernel  # noqa: E402
from examples.opencl.tuner.tune import (  # noqa: E402
    DeviceSpec,
    ProbeSpec,
    callable_source,
    cache_key,
    collect,
    default_parse_output,
    emit_all,
    run,
)


M = 960
K = 2560
FF = 9216
N2 = 2560
OUT_ROOT = Path(__file__).resolve().parent / "out"
FFN_OUT = REPO_ROOT / "examples" / "opencl" / "ffn" / "out"

TIME_RE = re.compile(
    r"FFN_CHAIN .*?gate\s+([0-9.]+)\s+up\s+([0-9.]+)\s+silu\s+([0-9.]+)\s+down\s+([0-9.]+)\s+total\s+([0-9.]+)"
)


def valid_cfg(stage: str, bm: int, bn: int, bk: int) -> dict[str, Any] | None:
    n = FF if stage in ("gate", "up") else N2
    k = K if stage in ("gate", "up") else FF
    if bm % 8 or bn % 8 or bk % 4:
        return None
    if M % bm or n % bn or k % bk:
        return None
    threads = (bm // 8) * (bn // 8)
    if threads <= 0 or threads > 256:
        return None
    return {
        "stage": stage,
        "M": M,
        "N": n,
        "K": k,
        "bm": bm,
        "bn": bn,
        "bk": bk,
        "threads": threads,
        "b_layout": "kn",
        "accum_dtype": "float32",
        "chunk_k": 256,
    }


def search_space(stage: str, coarse: bool = False) -> list[dict[str, Any]]:
    bns = (128, 256, 512) if stage in ("gate", "up") else (128, 256)
    if coarse:
        bks = (64,)
    else:
        bks = (16, 32, 64)
    configs = []
    for bm in (16, 32, 64):
        for bn in bns:
            for bk in bks:
                cfg = valid_cfg(stage, bm, bn, bk)
                if cfg is not None:
                    configs.append(cfg)
    return configs


def kernel_factory(config: dict[str, Any]):
    return make_grgemm_kernel(
        config["M"],
        config["N"],
        config["K"],
        config["bm"],
        config["bn"],
        config["bk"],
        config["threads"],
        config["b_layout"],
        config["accum_dtype"],
        config["chunk_k"],
    )


def stage_env(prefix: str, cfg: dict[str, Any]) -> dict[str, Any]:
    p = prefix.upper()
    return {f"TL_{p}_BM": cfg["bm"], f"TL_{p}_BN": cfg["bn"], f"TL_{p}_THREADS": cfg["threads"]}


def base_cfg(stage: str) -> dict[str, Any]:
    return valid_cfg(stage, 32, 128, 64)  # type: ignore[return-value]


def parse_chain(text: str) -> dict[str, Any]:
    m = TIME_RE.search(text)
    out: dict[str, Any] = {"status": "PASS" if "FAIL" not in text and "PASS" in text else "FAIL"}
    if m:
        out.update({
            "gate": float(m.group(1)),
            "up": float(m.group(2)),
            "silu": float(m.group(3)),
            "down": float(m.group(4)),
            "total": float(m.group(5)),
        })
        out["ms"] = out["total"]
    return out


def write_fixed_files(out_dir: Path, fixed: dict[str, Path]) -> dict[str, str]:
    names = {}
    for stage, src in fixed.items():
        dst = out_dir / f"fixed_{stage}.cl"
        dst.write_bytes(src.read_bytes())
        names[stage] = dst.name
    silu = out_dir / "fixed_silu.cl"
    silu.write_bytes((FFN_OUT / "silu_mul.cl").read_bytes())
    names["silu"] = silu.name
    return names


def make_manifest(stage: str, emits, fixed_names: dict[str, str], probe_iters: int, samples: int) -> list[dict[str, Any]]:
    rows = []
    for e in emits:
        if e.emit_error or not e.cl_path:
            continue
        gate = e.cl_path.name if stage == "gate" else fixed_names["gate"]
        up = e.cl_path.name if stage == "up" else fixed_names["up"]
        down = e.cl_path.name if stage == "down" else fixed_names["down"]
        env = {
            "TL_M": M,
            "TL_K": K,
            "TL_FF": FF,
            "TL_N2": N2,
            "TL_ITERS": probe_iters,
            "TL_CHECK_SAMPLES": samples,
        }
        for st in ("gate", "up", "down"):
            cfg = e.config if st == stage else base_cfg(st)
            # If a previous coordinate winner is fixed, its env is embedded in filename metadata below.
            if st != stage and f"{st}_cfg" in fixed_names:
                cfg = json.loads(fixed_names[f"{st}_cfg"])
            env.update(stage_env(st, cfg))
        rows.append({"id": e.config_id, "stage": stage, "gate": gate, "up": up, "silu": fixed_names["silu"], "down": down, "env": env})
    return rows


def deploy_and_run(out_dir: Path, tag: str, rows: list[dict[str, Any]], device: DeviceSpec) -> Path:
    (out_dir / "manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    script = '''#!/system/bin/sh
set -u
cd "$(dirname "$0")"
rm -f results.csv
printf 'id,stage,round,rc,log\n' > results.csv
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
        print(f"{cmd}; rc=$?; printf '%s,%s,%s,%s,%s\\n' {shlex.quote(item['id'])} {shlex.quote(item['stage'])} {r} $rc {shlex.quote(log)} >> results.csv")
PY
sh run_items.sh
'''
    (out_dir / "run_batch.sh").write_text(script, encoding="utf-8")
    files = [p for p in out_dir.glob("*.cl")] + [out_dir / "manifest.json", out_dir / "run_batch.sh"]
    md5 = "".join(f"{hashlib.md5(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files)
    (out_dir / "md5sums.txt").write_text(md5, encoding="utf-8")
    files.append(out_dir / "md5sums.txt")
    tar_path = out_dir / "bundle.tar"
    with tarfile.open(tar_path, "w") as tf:
        for p in files:
            tf.add(p, arcname=p.name)
    remote_base = f"{device.remote_dir.rstrip('/')}/{tag}"
    res = run(["ssh", device.ssh, f"rm -rf {shlex.quote(remote_base)} && mkdir -p {shlex.quote(remote_base)}"], timeout=60)
    if res.returncode != 0:
        raise RuntimeError(res.stdout)
    res = run(["scp", str(tar_path), f"{device.ssh}:{remote_base}/bundle.tar"], timeout=120)
    if res.returncode != 0:
        raise RuntimeError(res.stdout)
    cmd = f"cd {remote_base} && tar xf bundle.tar && if [ ! -x ./ffn_chain_test ] && [ -x ../../ffn_chain_test ]; then ln -sf ../../ffn_chain_test ./ffn_chain_test; fi && md5sum -c md5sums.txt"
    res = run(["ssh", device.ssh, cmd], timeout=120)
    if res.returncode != 0:
        raise RuntimeError(res.stdout)
    done = f"{remote_base}/{tag}.done"
    log = f"{remote_base}/{tag}.driver.log"
    launch = f"cd {remote_base} && rm -f {done} {log} && chmod +x run_batch.sh && nohup sh -c './run_batch.sh > {log} 2>&1; echo $? > {done}' >/dev/null 2>&1 &"
    res = run(["ssh", device.ssh, launch], timeout=30)
    if res.returncode != 0:
        raise RuntimeError(res.stdout)
    t0 = time.time()
    while time.time() - t0 < device.timeout_s:
        res = run(["ssh", device.ssh, f"test -f {done} && cat {done} || true"], timeout=30)
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        rc = next((ln for ln in reversed(lines) if ln.lstrip('-').isdigit()), "")
        if rc:
            if rc != "0":
                raise RuntimeError(f"remote rc={rc} log={log}")
            return collect(out_dir, tag, remote_base, device)
        time.sleep(device.poll_s)
    raise TimeoutError(tag)


def score_results(extract_dir: Path, stage: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with (extract_dir / "results.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (extract_dir / row["log"]).read_text(encoding="utf-8", errors="replace")
            parsed = parse_chain(text)
            parsed.update({"rc": int(row["rc"]), "round": int(row["round"]), "log": row["log"]})
            if parsed.get("status") == "PASS" and parsed.get(stage) is not None:
                parsed["ms"] = parsed[stage]
            out.setdefault(row["id"], []).append(parsed)
    return out


def run_stage(stage: str, fixed: dict[str, Path], fixed_cfg: dict[str, dict[str, Any]], tag: str, device: DeviceSpec, jobs: int, coarse: bool, samples: int = 256):
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = ProbeSpec(binary="ffn_chain_test", kernel_name="ffn_chain", env=lambda c: {}, parse=default_parse_output, rounds=2, iters=10, check_samples=samples)
    cfgs = search_space(stage, coarse=coarse)
    emits = emit_all(kernel_factory, cfgs, probe, out_dir, jobs=jobs)
    fixed_names = write_fixed_files(out_dir, fixed)
    for st, cfg in fixed_cfg.items():
        fixed_names[f"{st}_cfg"] = json.dumps(cfg, sort_keys=True)
    rows = make_manifest(stage, emits, fixed_names, 10, samples)
    extract = deploy_and_run(out_dir, tag, rows, device)
    meas = score_results(extract, stage)
    all_rows = []
    candidates = []
    for e in emits:
        good = [m for m in meas.get(e.config_id, []) if m.get("rc") == 0 and m.get("status") == "PASS" and m.get("ms") is not None]
        best = min(good, key=lambda m: m["ms"]) if good else None
        all_rows.append({"config_id": e.config_id, "config": e.config, "emit_error": e.emit_error, "measurements": meas.get(e.config_id, [])})
        if best:
            candidates.append((best["ms"], e, best))
    candidates.sort(key=lambda x: x[0])
    best = candidates[0] if candidates else None
    summary = {"stage": stage, "count": len(emits), "scored": len(candidates), "best": None, "top5": [], "bottom5": []}
    if best:
        summary["best"] = {"config_id": best[1].config_id, "config": best[1].config, "ms": best[0], "chain": best[2]}
        summary["top5"] = [{"config_id": e.config_id, "ms": ms, "config": e.config} for ms, e, _ in candidates[:5]]
        summary["bottom5"] = [{"config_id": e.config_id, "ms": ms, "config": e.config} for ms, e, _ in candidates[-5:]]
    (out_dir / f"{tag}_results.json").write_text(json.dumps(all_rows, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return out_dir, summary


def run_single_combo(tag: str, fixed: dict[str, Path], cfgs: dict[str, dict[str, Any]], device: DeviceSpec, samples: int, iters: int = 10):
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    names = write_fixed_files(out_dir, fixed)
    env = {"TL_M": M, "TL_K": K, "TL_FF": FF, "TL_N2": N2, "TL_ITERS": iters, "TL_CHECK_SAMPLES": samples}
    for st, cfg in cfgs.items():
        env.update(stage_env(st, cfg))
    rows = [{"id": "combo", "stage": "total", "gate": names["gate"], "up": names["up"], "silu": names["silu"], "down": names["down"], "env": env}]
    extract = deploy_and_run(out_dir, tag, rows, device)
    text = (extract / "logs" / "combo_r0.log").read_text(encoding="utf-8", errors="replace")
    parsed = parse_chain(text)
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(parsed, indent=2, sort_keys=True), encoding="utf-8")
    print(f"{tag} {parsed}")
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser(description="Coordinate-descent tune FFN chain on OnePlus 13")
    ap.add_argument("--tag", default="ffn_cd")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--ssh", default="oneplus13-reverse")
    ap.add_argument("--remote-dir", default="/data/data/com.termux/files/home/tl_ffn/tune")
    ap.add_argument("--timeout-s", type=float, default=2400)
    ap.add_argument("--coarse", action="store_true")
    args = ap.parse_args()
    device = DeviceSpec(ssh=args.ssh, remote_dir=args.remote_dir, timeout_s=args.timeout_s)
    fixed = {"gate": FFN_OUT / "ffn_gate.cl", "up": FFN_OUT / "ffn_gate.cl", "down": FFN_OUT / "ffn_down.cl"}
    cfgs = {"gate": base_cfg("gate"), "up": base_cfg("up"), "down": base_cfg("down")}
    run_single_combo(args.tag + "_baseline", fixed, cfgs, device, samples=256, iters=10)
    for stage in ("gate", "up", "down"):
        out_dir, summary = run_stage(stage, fixed, cfgs, args.tag + "_" + stage, device, args.jobs, args.coarse)
        best = summary.get("best")
        if not best:
            raise SystemExit(f"no best for {stage}")
        best_id = best["config_id"]
        fixed[stage] = out_dir / f"{best_id}.cl"
        cfgs[stage] = best["config"]
    run_single_combo(args.tag + "_best_sample", fixed, cfgs, device, samples=256, iters=10)
    run_single_combo(args.tag + "_best_full", fixed, cfgs, device, samples=0, iters=3)
    (OUT_ROOT / args.tag).mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / args.tag / "final_config.json").write_text(json.dumps(cfgs, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
