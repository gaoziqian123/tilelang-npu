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
