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
    with tvm.target.Target("opencl"), tvm.transform.PassContext(
        config={"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}
    ):
        artifact = tilelang.lower(factory(config), target="opencl", enable_device_compile=False)
    return artifact.kernel_source


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


def write_manifest(out_dir: Path, emits: list[EmitResult], probe: ProbeSpec) -> None:
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
        manifest.append({"id": e.config_id, "file": e.cl_path.name, "env": env})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def make_batch_script(probe: ProbeSpec, device: DeviceSpec, manifest_name: str = "manifest.json") -> str:
    # Keep phone-side dependencies minimal: parse manifest with python if present,
    # otherwise fall back to a line-oriented manifest is intentionally not supported.
    return f'''#!/system/bin/sh
set -u
cd "$(dirname "$0")"
rm -f results.csv
printf 'id,round,rc,log\n' > results.csv
rm -rf logs
mkdir -p logs
python3 - <<'PY' > run_items.sh
import json, shlex
items = json.load(open({manifest_name!r}))
binary = {probe.binary!r}
kernel_name = {probe.kernel_name!r}
for item in items:
    env = dict(item['env'])
    env['LD_LIBRARY_PATH'] = {device.ld_library_path!r}
    exports = ' '.join(f"{{k}}={{shlex.quote(str(v))}}" for k, v in env.items())
    for r in range({probe.rounds}):
        log = f"logs/{{item['id']}}_r{{r}}.log"
        cmd = f"{{exports}} ./{{shlex.quote(binary)}} {{shlex.quote(item['file'])}} {{shlex.quote(kernel_name)}} > {{shlex.quote(log)}} 2>&1"
        print(f"{{cmd}}; rc=$?; printf '%s,%s,%s,%s\\n' {{shlex.quote(item['id'])}} {{r}} $rc {{shlex.quote(log)}} >> results.csv")
PY
sh run_items.sh
'''


def deploy(out_dir: Path, tag: str, emits: list[EmitResult], probe: ProbeSpec, device: DeviceSpec) -> str:
    write_manifest(out_dir, emits, probe)
    (out_dir / "run_batch.sh").write_text(make_batch_script(probe, device), encoding="utf-8")
    files = [p for p in out_dir.glob("*.cl")] + [out_dir / "manifest.json", out_dir / "run_batch.sh"]
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
    pack = f"cd {shlex.quote(remote_base)} && tar cf {shlex.quote(remote_tar)} results.csv logs {tag}.driver.log"
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
    measurements: dict[str, list[dict[str, Any]]] = {}
    with (extract_dir / "results.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            log_rel = row["log"]
            text = (extract_dir / log_rel).read_text(encoding="utf-8", errors="replace")
            parsed = probe.parse(text)
            parsed.update({"round": int(row["round"]), "rc": int(row["rc"]), "log": log_rel})
            measurements.setdefault(row["id"], []).append(parsed)
    return measurements


def choose_best(emits: list[EmitResult], measurements: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    emit_by_id = {e.config_id: e for e in emits}
    candidates = []
    for cid, rows in measurements.items():
        good = [r for r in rows if r.get("rc") == 0 and r.get("status") == "PASS" and r.get("ms") is not None]
        if not good:
            continue
        best_row = min(good, key=lambda r: r["ms"])
        emit = emit_by_id[cid]
        candidates.append((best_row["ms"], cid, best_row, emit))
    if not candidates:
        return None
    _, cid, row, emit = min(candidates, key=lambda x: x[0])
    return {"config_id": cid, "config": emit.config, "ms": row["ms"], "cos": row.get("cos"), "cache_key": emit.cache_key}


def write_results(tag: str, out_dir: Path, emits: list[EmitResult], measurements: dict[str, list[dict[str, Any]]], best: dict[str, Any] | None) -> None:
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


def tune(
    *,
    factory: KernelFactory,
    configs: Iterable[dict[str, Any]],
    probe: ProbeSpec,
    device: DeviceSpec,
    tag: str,
    jobs: int = 8,
    deploy_only: bool = False,
) -> TuneResult:
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    emits = emit_all(factory, configs, probe, out_dir, jobs=jobs)
    ok = sum(1 for e in emits if not e.emit_error)
    print(f"EMIT_DONE ok={ok} fail={len(emits)-ok} out={out_dir}")
    if deploy_only:
        write_results(tag, out_dir, emits, {}, None)
        return TuneResult(tag=tag, out_dir=out_dir, emits=emits)
    remote_base = deploy(out_dir, tag, emits, probe, device)
    print(f"DEPLOY_DONE remote={remote_base}")
    start_and_wait(tag, remote_base, device)
    print("REMOTE_RUN_DONE")
    extract_dir = collect(out_dir, tag, remote_base, device)
    measurements = parse_measurements(extract_dir, probe)
    best = choose_best(emits, measurements)
    write_results(tag, out_dir, emits, measurements, best)
    print(f"RESULTS_WRITTEN {out_dir / (tag + '_results.json')}")
    if best:
        print(f"BEST {best['config_id']} ms={best['ms']} cos={best.get('cos')} config={best['config']}")
    return TuneResult(tag=tag, out_dir=out_dir, emits=emits, measurements=measurements, best=best)
