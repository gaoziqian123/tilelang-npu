# OpenCL remote tuner MVP

This directory contains an additive, kernel-independent remote auto-tuning
orchestrator for TileLang OpenCL kernels on the OnePlus 13.

## GR GEMM pilot

From the TileLang repo root:

```bash
source .venv/bin/activate
PYTHONPATH=/root/project/tilelang python examples/opencl/tuner/tune_gemm.py \
  --tag gemm_gr_1024x2560x2560
```

What it does:

1. builds the config list for `M=1024,N=2560,K=2560`, fp32 accumulation;
2. lowers all valid configs in parallel with TileLang's default cache enabled;
3. writes `.cl` files under `examples/opencl/tuner/out/<tag>/`;
4. deploys one tarball to `/data/data/com.termux/files/home/tl_gemm/tune/<tag>/` and verifies md5sums;
5. starts a detached phone-side batch script (`nohup ...; echo $? > done`);
6. pulls back `results.csv` and per-config logs;
7. writes `<tag>_results.json`, `<tag>_best.json`, and `<tag>_summary.txt`;
8. reruns the champion once with `TL_CHECK_SAMPLES=0` for full validation.

Phone prerequisites are the existing `~/tl_gemm/gemm_nt_test` binary and the
OpenCL runtime.  The runner uses:

```text
LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64
./gemm_nt_test <file.cl> gemm_nt_kernel_kernel
```

The generic orchestrator is `tune.py`.  It accepts a Python kernel factory
`config -> T.prim_func`, a config list, `ProbeSpec` (binary/kernel/env/parser),
`DeviceSpec` (ssh alias/remote dir), and a tag.  Cache keys are sha256 over the
TileLang git revision, kernel factory source, config, and probe spec.

## Phase-2 scheduler features

`tune.py` also has a kernel-independent `TuneSpec` used by `tune_gemm.py`:

- **Priors**: known-good config dicts are moved to the front before emit/run;
  the GR GEMM pilot seeds `bm=32,bn=256,bk=16,kn` and `bk=32`.
- **Early stop**: configs are grouped by a family key that drops `bk`,
  `threads`, and `pass_configs`.  If the first measured member of a family is
  slower than `2.0x` current best, the rest of the family is marked
  `skipped_family`; after 12 measured configs fail to enter top-3, the sweep is
  marked `early_stopped`.
- **Pass configs in the search space**: configs may carry a `pass_configs` dict;
  it is validated, recorded in results/cache keys, and merged into the
  TileLang `PassContext`.  The GEMM pilot demonstrates
  `{"tl.UnrollLoop": {"explicit_unroll": true|false, ...}}`.
- **Canary normalization**: rounds are interleaved forward/reverse, and every 10
  measured configs the current best is re-run as a canary.  Results keep both
  raw `ms` and canary-normalized `norm_ms`; ranking uses `norm_ms`.

Each run writes `<tag>_meta.json` with `measured`, `canaries`, `family_skip`,
`early`, `top3`, and `priors`.
