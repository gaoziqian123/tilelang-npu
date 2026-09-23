from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import re
import shlex
import subprocess
import tarfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import tilelang
import tilelang.opencl  # noqa: F401 - register OpenCL target
from tilelang import tvm


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = Path(__file__).resolve().parent / "out"
DEFAULT_FAMILY_SLOWDOWN = 2.0
DEFAULT_GLOBAL_NO_TOP3_LIMIT = 12
DEFAULT_CANARY_INTERVAL = 10
DEFAULT_TUNE_ROUNDS = 2


KernelFactory = Callable[[dict[str, Any]], Any]
EnvFactory = Callable[[dict[str, Any]], dict[str, Any]]
ParseOutput = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class ProbeSpec:
    binary: str
    kernel_name: str
    env: EnvFactory
    parse: ParseOutput
    rounds: int = 2
    iters: int = 10
    check_samples: int | None = None
    cos_min: float | None = None

    def cache_payload(self) -> dict[str, Any]:
        return {
            "binary": self.binary,
            "kernel_name": self.kernel_name,
            "rounds": self.rounds,
            "iters": self.iters,
            "check_samples": self.check_samples,
            "cos_min": self.cos_min,
        }


@dataclass(frozen=True)
class DeviceSpec:
    ssh: str = "oneplus13-reverse"
    remote_dir: str = "/data/data/com.termux/files/home/tl_gemm/tune"
    ld_library_path: str = ".:/system/lib64:/vendor/lib64"
    poll_s: float = 5.0
    timeout_s: float = 1800.0


@dataclass(frozen=True)
class TuneSpec:
    priors: list[dict[str, Any]] = field(default_factory=list)
    family_slowdown: float = DEFAULT_FAMILY_SLOWDOWN
    global_no_top3_limit: int = DEFAULT_GLOBAL_NO_TOP3_LIMIT
    canary_interval: int = DEFAULT_CANARY_INTERVAL
    rounds: int = DEFAULT_TUNE_ROUNDS


@dataclass
class EmitResult:
    config_id: str
    config: dict[str, Any]
    cl_path: Path | None = None
    source_sha256: str | None = None
    cache_key: str | None = None
    emit_error: str | None = None


@dataclass
class TuneResult:
    tag: str
    out_dir: Path
    emits: list[EmitResult]
    measurements: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    best: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def run(cmd: list[str], *, timeout: float | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd)


def git_rev() -> str:
    res = run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    return res.stdout.strip() if res.returncode == 0 else "unknown"


def callable_source(fn: Callable[..., Any]) -> str:
    try:
        return inspect.getsource(fn)
    except OSError:
        return repr(fn)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(factory_src: str, config: dict[str, Any], probe: ProbeSpec) -> str:
    payload = {
        "tilelang_git_rev": git_rev(),
        "kernel_factory_source": factory_src,
        "config": config,
        "probe": probe.cache_payload(),
    }
    return hashlib.sha256(stable_json(payload).encode()).hexdigest()


def lower_opencl(factory: KernelFactory, config: dict[str, Any]) -> str:
    pass_configs = config.get("pass_configs", {})
    if pass_configs is None:
        pass_configs = {}
    if not isinstance(pass_configs, dict):
        raise TypeError("config['pass_configs'] must be a dict when present")
    pcfg = {"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}
    for k, v in pass_configs.items():
        pcfg[k] = v
    built = factory(dict(config))
    if isinstance(built, tuple) and len(built) == 2:
        func, extra_pass_configs = built
        if extra_pass_configs:
            if not isinstance(extra_pass_configs, dict):
                raise TypeError("kernel factory pass_configs return must be a dict")
            for k, v in extra_pass_configs.items():
                pcfg[k] = v
    else:
        func = built
    with tvm.target.Target("opencl"), tvm.transform.PassContext(
        config=pcfg
    ):
        artifact = tilelang.lower(func, target="opencl", enable_device_compile=False)
    return artifact.kernel_source


def family_key(config: dict[str, Any]) -> str:
    drop = {"bk", "threads", "pass_configs"}
    return stable_json({k: v for k, v in config.items() if k not in drop})


def config_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return stable_json(a) == stable_json(b)


def order_configs(configs: Iterable[dict[str, Any]], spec: TuneSpec) -> list[dict[str, Any]]:
    remaining = [dict(c) for c in configs]
    ordered: list[dict[str, Any]] = []
    for prior in spec.priors:
        for i, cfg in enumerate(remaining):
            if config_match(cfg, prior):
                ordered.append(remaining.pop(i))
                break
    ordered.extend(remaining)
    return ordered


def emit_one(
    factory: KernelFactory,
    factory_src: str,
    probe: ProbeSpec,
    out_dir: Path,
    config_id: str,
    config: dict[str, Any],
) -> EmitResult:
    try:
        src = lower_opencl(factory, config)
        cl_path = out_dir / f"{config_id}.cl"
        cl_path.write_text(src, encoding="utf-8")
        return EmitResult(
            config_id=config_id,
            config=config,
            cl_path=cl_path,
            source_sha256=hashlib.sha256(src.encode()).hexdigest(),
            cache_key=cache_key(factory_src, config, probe),
        )
    except Exception:
        return EmitResult(
            config_id=config_id,
            config=config,
            cache_key=cache_key(factory_src, config, probe),
            emit_error=traceback.format_exc(),
        )


def emit_all(
    factory: KernelFactory,
    configs: Iterable[dict[str, Any]],
    probe: ProbeSpec,
    out_dir: Path,
    jobs: int = 8,
) -> list[EmitResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    factory_src = callable_source(factory)
    config_list = list(configs)
    results: list[EmitResult] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {
            ex.submit(emit_one, factory, factory_src, probe, out_dir, f"c{i:04d}", cfg): i
            for i, cfg in enumerate(config_list)
        }
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda r: r.config_id)
    return results


def write_manifest(out_dir: Path, emits: list[EmitResult], probe: ProbeSpec, spec: TuneSpec) -> None:
    manifest = []
    for e in emits:
        if e.cl_path is None or e.emit_error:
            continue
        env = probe.env(e.config)
        env["TL_ITERS"] = str(probe.iters)
        if probe.check_samples is not None:
            env["TL_CHECK_SAMPLES"] = str(probe.check_samples)
        if probe.cos_min is not None:
            env["TL_COS_MIN"] = str(probe.cos_min)
        manifest.append({"id": e.config_id, "file": e.cl_path.name, "env": env, "config": e.config, "family_key": family_key(e.config)})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "tune_spec.json").write_text(json.dumps({
        "priors": spec.priors,
        "family_slowdown": spec.family_slowdown,
        "global_no_top3_limit": spec.global_no_top3_limit,
        "canary_interval": spec.canary_interval,
        "rounds": spec.rounds,
    }, indent=2, sort_keys=True), encoding="utf-8")


def make_batch_script(probe: ProbeSpec, device: DeviceSpec, spec: TuneSpec, manifest_name: str = "manifest.json") -> str:
    return f'''#!/system/bin/sh
set -u
cd "$(dirname "$0")"
python3 - <<'PY'
import csv, json, os, re, shutil, subprocess
from pathlib import Path

items = json.load(open({manifest_name!r}))
binary = {probe.binary!r}
kernel_name = {probe.kernel_name!r}
family_slowdown = {spec.family_slowdown!r}
no_top3_limit = {spec.global_no_top3_limit!r}
canary_interval = {spec.canary_interval!r}
rounds = {spec.rounds!r}
ld = {device.ld_library_path!r}

ms_re = re.compile("ms_iter\\s*[=:]?\\s*([0-9]+(?:\\.[0-9]+)?)")
logs = Path('logs'); shutil.rmtree(logs, ignore_errors=True); logs.mkdir()
rows = []
state = {{'best_ms': None, 'best_id': '', 'top3': [], 'family_seen': set(), 'family_skip': set(), 'measured': 0, 'no_top3': 0, 'early': False, 'canaries': 0}}

def run_item(item, r, kind):
    env = os.environ.copy(); env.update({{k: str(v) for k, v in item['env'].items()}}); env['LD_LIBRARY_PATH'] = ld
    log = logs / f"{{len(rows):04d}}_{{item['id']}}_r{{r}}_{{kind}}.log"
    with log.open('w') as f:
        rc = subprocess.run([f"./{{binary}}", item['file'], kernel_name], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    txt = log.read_text(errors='replace')
    m = ms_re.search(txt)
    status = 'PASS' if 'PASS' in txt and 'FAIL' not in txt else ('FAIL' if 'FAIL' in txt else 'UNKNOWN')
    row = {{'id': item['id'], 'round': r, 'kind': kind, 'rc': rc, 'log': str(log), 'skipped_reason': '', 'ms': float(m.group(1)) if m else None, 'status': status}}
    rows.append(row)
    return row

def add_skip(item, r, reason):
    rows.append({{'id': item['id'], 'round': r, 'kind': 'skipped', 'rc': '', 'log': '', 'skipped_reason': reason, 'ms': None, 'status': 'SKIPPED'}})

def update_state(item, row):
    if row['kind'] != 'measure':
        return
    state['measured'] += 1
    fam = item['family_key']; cid = item['id']; ms = row['ms']
    entered = False
    if ms is not None and row['status'] == 'PASS' and row['rc'] == 0:
        old = list(state['top3'])
        state['top3'] = sorted(old + [{{'id': cid, 'ms': ms}}], key=lambda x: x['ms'])[:3]
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
'''


def deploy(out_dir: Path, tag: str, emits: list[EmitResult], probe: ProbeSpec, device: DeviceSpec, spec: TuneSpec) -> str:
    write_manifest(out_dir, emits, probe, spec)
    (out_dir / "run_batch.sh").write_text(make_batch_script(probe, device, spec), encoding="utf-8")
    files = [p for p in out_dir.glob("*.cl")] + [out_dir / "manifest.json", out_dir / "tune_spec.json", out_dir / "run_batch.sh"]
    md5_lines = []
    for p in files:
        md5_lines.append(f"{hashlib.md5(p.read_bytes()).hexdigest()}  {p.name}\n")
    (out_dir / "md5sums.txt").write_text("".join(md5_lines), encoding="utf-8")
    files.append(out_dir / "md5sums.txt")

    tar_path = out_dir / "bundle.tar"
    with tarfile.open(tar_path, "w") as tf:
        for p in files:
            tf.add(p, arcname=p.name)

    remote_base = f"{device.remote_dir.rstrip('/')}/{tag}"
    prep = f"rm -rf {shlex.quote(remote_base)} && mkdir -p {shlex.quote(remote_base)}"
    res = run(["ssh", device.ssh, prep], timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"remote mkdir failed:\n{res.stdout}")
    res = run(["scp", str(tar_path), f"{device.ssh}:{remote_base}/bundle.tar"], timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"scp bundle failed:\n{res.stdout}")
    bin_q = shlex.quote(probe.binary)
    extract = (
        f"cd {shlex.quote(remote_base)} && tar xf bundle.tar && "
        f"if [ ! -x ./{bin_q} ] && [ -x ../../{bin_q} ]; then ln -sf ../../{bin_q} ./{bin_q}; fi && "
        f"md5sum -c md5sums.txt"
    )
    res = run(["ssh", device.ssh, extract], timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"remote md5 failed:\n{res.stdout}")
    return remote_base


def start_and_wait(tag: str, remote_base: str, device: DeviceSpec) -> None:
    done = f"{remote_base}/{tag}.done"
    log = f"{remote_base}/{tag}.driver.log"
    launch = (
        f"cd {shlex.quote(remote_base)} && rm -f {shlex.quote(done)} {shlex.quote(log)} && "
        f"chmod +x run_batch.sh && nohup sh -c './run_batch.sh > {shlex.quote(log)} 2>&1; echo $? > {shlex.quote(done)}' >/dev/null 2>&1 &"
    )
    res = run(["ssh", device.ssh, launch], timeout=30)
    if res.returncode != 0:
        raise RuntimeError(f"remote launch failed:\n{res.stdout}")
    t0 = time.time()
    while True:
        if time.time() - t0 > device.timeout_s:
            raise TimeoutError(f"remote run timed out after {device.timeout_s}s: {remote_base}")
        res = run(["ssh", device.ssh, f"test -f {shlex.quote(done)} && cat {shlex.quote(done)} || true"], timeout=30)
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        txt = next((ln for ln in reversed(lines) if re.fullmatch(r"-?\d+", ln)), "")
        if txt:
            if txt != "0":
                raise RuntimeError(f"remote batch rc={txt}; see {log}")
            return
        time.sleep(device.poll_s)


def collect(out_dir: Path, tag: str, remote_base: str, device: DeviceSpec) -> Path:
    remote_tar = f"{remote_base}/{tag}_results.tar"
    pack = f"cd {shlex.quote(remote_base)} && tar cf {shlex.quote(remote_tar)} results.csv run_meta.json logs {tag}.driver.log"
    res = run(["ssh", device.ssh, pack], timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"remote result pack failed:\n{res.stdout}")
    local_tar = out_dir / f"{tag}_results.tar"
    res = run(["scp", f"{device.ssh}:{remote_tar}", str(local_tar)], timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"scp results failed:\n{res.stdout}")
    extract_dir = out_dir / "remote_results"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(local_tar, "r") as tf:
        tf.extractall(extract_dir)
    return extract_dir


_MS_RE = re.compile(r"ms_iter\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)")
_COS_RE = re.compile(r"cos(?:ine)?\s*[=: ]+([-+0-9.eE]+)")


def default_parse_output(text: str) -> dict[str, Any]:
    ms_m = _MS_RE.search(text)
    cos_m = _COS_RE.search(text)
    status = "PASS" if "PASS" in text and "FAIL" not in text else ("FAIL" if "FAIL" in text else "UNKNOWN")
    return {
        "ms": float(ms_m.group(1)) if ms_m else None,
        "cos": float(cos_m.group(1)) if cos_m else None,
        "status": status,
    }


def parse_measurements(extract_dir: Path, probe: ProbeSpec) -> dict[str, list[dict[str, Any]]]:
    flat: list[dict[str, Any]] = []
    with (extract_dir / "results.csv").open(newline="", encoding="utf-8") as f:
        for seq, row in enumerate(csv.DictReader(f)):
            log_rel = row["log"]
            if log_rel:
                text = (extract_dir / log_rel).read_text(encoding="utf-8", errors="replace")
                parsed = probe.parse(text)
            else:
                parsed = {"status": row.get("status", "SKIPPED")}
            if row.get("ms"):
                parsed["ms"] = float(row["ms"])
            if row.get("status"):
                parsed["status"] = row["status"]
            parsed.update({
                "seq": seq,
                "round": int(row["round"]) if row["round"] not in ("", None) else None,
                "kind": row.get("kind", "measure"),
                "rc": int(row["rc"]) if row.get("rc") not in ("", None) else None,
                "log": log_rel,
                "skipped_reason": row.get("skipped_reason", ""),
                "raw_ms": parsed.get("ms"),
            })
            flat.append((row["id"], parsed))
    canaries = [p for _, p in flat if p.get("kind") == "canary" and p.get("raw_ms") is not None and p.get("rc") == 0]
    can_ms = sorted(p["raw_ms"] for p in canaries)
    median = can_ms[len(can_ms)//2] if can_ms else None
    last_canary = None
    measurements: dict[str, list[dict[str, Any]]] = {}
    for cid, parsed in flat:
        if parsed.get("kind") == "canary" and parsed.get("raw_ms") is not None and parsed.get("rc") == 0:
            last_canary = parsed["raw_ms"]
        if parsed.get("raw_ms") is not None:
            if median is not None and last_canary:
                parsed["norm_ms"] = parsed["raw_ms"] / last_canary * median
            else:
                parsed["norm_ms"] = parsed["raw_ms"]
        measurements.setdefault(cid, []).append(parsed)
    return measurements


def choose_best(emits: list[EmitResult], measurements: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    emit_by_id = {e.config_id: e for e in emits}
    candidates = []
    for cid, rows in measurements.items():
        good = [r for r in rows if r.get("kind") == "measure" and r.get("rc") == 0 and r.get("status") == "PASS" and r.get("norm_ms") is not None]
        if not good:
            continue
        best_row = min(good, key=lambda r: r["norm_ms"])
        emit = emit_by_id[cid]
        candidates.append((best_row["norm_ms"], cid, best_row, emit))
    if not candidates:
        return None
    _, cid, row, emit = min(candidates, key=lambda x: x[0])
    return {"config_id": cid, "config": emit.config, "ms": row["norm_ms"], "raw_ms": row.get("raw_ms"), "cos": row.get("cos"), "cache_key": emit.cache_key}


def write_results(tag: str, out_dir: Path, emits: list[EmitResult], measurements: dict[str, list[dict[str, Any]]], best: dict[str, Any] | None, meta: dict[str, Any] | None = None) -> None:
    all_rows = []
    for e in emits:
        all_rows.append({
            "config_id": e.config_id,
            "config": e.config,
            "cache_key": e.cache_key,
            "source_sha256": e.source_sha256,
            "emit_error": e.emit_error,
            "measurements": measurements.get(e.config_id, []),
        })
    (out_dir / f"{tag}_results.json").write_text(json.dumps(all_rows, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / f"{tag}_best.json").write_text(json.dumps(best or {}, indent=2, sort_keys=True), encoding="utf-8")
    if meta is not None:
        (out_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def tune(
    *,
    factory: KernelFactory,
    configs: Iterable[dict[str, Any]],
    probe: ProbeSpec,
    device: DeviceSpec,
    tag: str,
    jobs: int = 8,
    deploy_only: bool = False,
    spec: TuneSpec | None = None,
) -> TuneResult:
    spec = spec or TuneSpec()
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered_configs = order_configs(configs, spec)
    emits = emit_all(factory, ordered_configs, probe, out_dir, jobs=jobs)
    ok = sum(1 for e in emits if not e.emit_error)
    print(f"EMIT_DONE ok={ok} fail={len(emits)-ok} out={out_dir}")
    if deploy_only:
        write_results(tag, out_dir, emits, {}, None, {"priors": spec.priors})
        return TuneResult(tag=tag, out_dir=out_dir, emits=emits)
    remote_base = deploy(out_dir, tag, emits, probe, device, spec)
    print(f"DEPLOY_DONE remote={remote_base}")
    start_and_wait(tag, remote_base, device)
    print("REMOTE_RUN_DONE")
    extract_dir = collect(out_dir, tag, remote_base, device)
    measurements = parse_measurements(extract_dir, probe)
    best = choose_best(emits, measurements)
    meta = {}
    meta_path = extract_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({"priors": spec.priors})
    write_results(tag, out_dir, emits, measurements, best, meta)
    print(f"RESULTS_WRITTEN {out_dir / (tag + '_results.json')}")
    if best:
        print(f"BEST {best['config_id']} ms={best['ms']} cos={best.get('cos')} config={best['config']}")
    return TuneResult(tag=tag, out_dir=out_dir, emits=emits, measurements=measurements, best=best, meta=meta)
