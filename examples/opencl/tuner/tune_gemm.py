from __future__ import annotations

import argparse
import json
import sys
import time
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
    TuneSpec,
    default_parse_output,
    run,
    tune,
)


M = 1024
N = 2560
K = 2560
PASS_TRUE = {"tl.UnrollLoop": {"explicit_unroll": True, "unroll_local_access": True}}
PASS_FALSE = {"tl.UnrollLoop": {"explicit_unroll": False, "unroll_local_access": True}}
CHAMPION = {"bm": 32, "bn": 256, "bk": 16, "b_layout": "kn"}
RUNNER_UP = {"bm": 32, "bn": 256, "bk": 32, "b_layout": "kn"}
CHAMPION_MS = 9.864


def valid_config(bm: int, bn: int, bk: int, b_layout: str, pass_configs: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if bm % 8 or bn % 8 or bk % 4:
        return None
    if M % bm or N % bn or K % bk:
        return None
    threads = (bm // 8) * (bn // 8)
    if threads <= 0 or threads > 256:
        return None
    return {
        "M": M,
        "N": N,
        "K": K,
        "bm": bm,
        "bn": bn,
        "bk": bk,
        "threads": threads,
        "b_layout": b_layout,
        "accum_dtype": "float32",
        "chunk_k": 256,
        "pass_configs": pass_configs or PASS_TRUE,
    }


def search_space() -> list[dict[str, Any]]:
    configs = []
    seen = set()
    for bm in (16, 32, 64):
        for bn in (64, 128, 256):
            for bk in (16, 32, 64):
                for b_layout in ("kn", "nk"):
                    for pass_cfg in (PASS_TRUE, PASS_FALSE):
                        cfg = valid_config(bm, bn, bk, b_layout, pass_cfg)
                        if cfg is None:
                            continue
                        key = (bm, bn, bk, b_layout, json.dumps(pass_cfg, sort_keys=True))
                        if key not in seen:
                            seen.add(key)
                            configs.append(cfg)
    return configs


def prior_config(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = valid_config(spec["bm"], spec["bn"], spec["bk"], spec["b_layout"])
    assert cfg is not None
    return cfg


def kernel_factory(config: dict[str, Any]):
    pass_configs = config.pop("pass_configs", {})
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
    ), pass_configs


def gemm_env(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "TL_M": config["M"],
        "TL_N": config["N"],
        "TL_K": config["K"],
        "TL_BM": config["bm"],
        "TL_BN": config["bn"],
        "TL_THREADS": config["threads"],
        "TL_B_LAYOUT": config["b_layout"],
        "TL_CHECK": "cosine",
    }


def summarize(out_dir: Path, tag: str) -> None:
    rows = json.loads((out_dir / f"{tag}_results.json").read_text(encoding="utf-8"))
    scored = []
    failures = []
    for row in rows:
        if row.get("emit_error"):
            failures.append((row["config_id"], "emit_error", row["config"]))
            continue
        good = [m for m in row.get("measurements", []) if m.get("kind") == "measure" and m.get("rc") == 0 and m.get("status") == "PASS" and m.get("norm_ms") is not None]
        if good:
            best = min(good, key=lambda m: m["norm_ms"])
            scored.append((best["norm_ms"], best.get("cos"), row["config_id"], row["config"]))
        else:
            failures.append((row["config_id"], "measure_fail", row["config"]))
    scored.sort(key=lambda x: x[0])
    summary = []
    summary.append(f"total={len(rows)} scored={len(scored)} failures={len(failures)}")
    if scored:
        best = scored[0]
        tflops = 2 * M * N * K / (best[0] * 1e-3) / 1e12
        summary.append(f"best={best[2]} ms={best[0]:.4f} tflops={tflops:.3f} cos={best[1]} cfg={best[3]}")
        summary.append(f"champion_reference bm=32 bn=256 bk=16 kn ms={CHAMPION_MS:.3f} tflops={2*M*N*K/(CHAMPION_MS*1e-3)/1e12:.3f}")
        summary.append("top5:")
        for ms, cos, cid, cfg in scored[:5]:
            summary.append(f"  {cid} ms={ms:.4f} cos={cos} cfg={cfg}")
        summary.append("bottom5:")
        for ms, cos, cid, cfg in scored[-5:]:
            summary.append(f"  {cid} ms={ms:.4f} cos={cos} cfg={cfg}")
    if failures:
        summary.append("failures_sample:")
        for f in failures[:10]:
            summary.append(f"  {f[0]} {f[1]} cfg={f[2]}")
    text = "\n".join(summary) + "\n"
    (out_dir / f"{tag}_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def full_check(best_path: Path, device: DeviceSpec, tag: str) -> None:
    best = json.loads(best_path.read_text(encoding="utf-8"))
    if not best:
        print("FULL_CHECK_SKIP no best config")
        return
    cid = best["config_id"]
    remote_base = f"{device.remote_dir.rstrip('/')}/{tag}"
    done = f"{remote_base}/full_{cid}.done"
    log = f"{remote_base}/full_{cid}.log"
    cl = f"{cid}.cl"
    env = gemm_env(best["config"])
    env.update({"TL_ITERS": 3, "TL_CHECK_SAMPLES": 0, "LD_LIBRARY_PATH": device.ld_library_path})
    exports = " ".join(f"{k}='{v}'" for k, v in env.items())
    cmd = (
        f"cd {remote_base} && rm -f {done} {log} && "
        f"nohup sh -c \"{exports} ./gemm_nt_test {cl} gemm_nt_kernel_kernel > {log} 2>&1; rc=\\$?; echo \\$rc > {done}\" >/dev/null 2>&1 &"
    )
    res = run(["ssh", device.ssh, cmd], timeout=30)
    if res.returncode != 0:
        raise RuntimeError(res.stdout)
    t0 = time.time()
    while time.time() - t0 < device.timeout_s:
        res = run(["ssh", device.ssh, f"test -f {done} && cat {done} || true"], timeout=30)
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
        rc_line = next((ln for ln in reversed(lines) if ln.lstrip("-").isdigit()), "")
        if rc_line:
            pull = run(["ssh", device.ssh, f"cat {log}"], timeout=120)
            (best_path.parent / f"{tag}_fullcheck.log").write_text(pull.stdout, encoding="utf-8")
            parsed = default_parse_output(pull.stdout)
            (best_path.parent / f"{tag}_fullcheck.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
            print(f"FULL_CHECK rc={rc_line} parsed={parsed} log={best_path.parent / (tag + '_fullcheck.log')}")
            return
        time.sleep(device.poll_s)
    raise TimeoutError("full check timed out")


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-tune TileLang OpenCL GR GEMM on OnePlus 13")
    ap.add_argument("--tag", default="gemm_gr_1024x2560x2560")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--ssh", default="oneplus13-reverse")
    ap.add_argument("--remote-dir", default="/data/data/com.termux/files/home/tl_gemm/tune")
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--deploy-only", action="store_true")
    ap.add_argument("--skip-full-check", action="store_true")
    args = ap.parse_args()

    configs = search_space()
    print(f"CONFIGS {len(configs)}")
    probe = ProbeSpec(
        binary="gemm_nt_test",
        kernel_name="gemm_nt_kernel_kernel",
        env=gemm_env,
        parse=default_parse_output,
        rounds=2,
        iters=10,
        check_samples=2048,
        cos_min=None,
    )
    device = DeviceSpec(ssh=args.ssh, remote_dir=args.remote_dir, timeout_s=args.timeout_s)
    spec = TuneSpec(priors=[prior_config(CHAMPION), prior_config(RUNNER_UP)])
    result = tune(factory=kernel_factory, configs=configs, probe=probe, device=device, tag=args.tag, jobs=args.jobs, deploy_only=args.deploy_only, spec=spec)
    summarize(result.out_dir, args.tag)
    if not args.deploy_only and not args.skip_full_check:
        full_check(result.out_dir / f"{args.tag}_best.json", device, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
