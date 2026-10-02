# Historical device evidence — compact checkpoint record

Transcribed from existing local run logs on 2026-10-02; **not tests run by this
documentation session**, not verification of the current complete tree.
Only benchmark records are retained; process lists, logcat, network endpoints
and unrelated application logs are omitted. Hashes identify deployed artifacts,
not a promise that a fresh checkout can reproduce their binaries. Exact compiler
and deployment command chains for these historical runs were not recovered in
this documentation batch; the emission recipes in the READMEs are explicitly
not substituted for missing historical commands.

## GEMM independent-verifier run

Source record: `vf_gemm_verify_20261002/run.log`, UTC 2026-10-01 16:13:01.
Formal emit / actual O2 build is the reported provenance of this verifier run.
Exact benchmark command/environment from its runner script (historical only;
device tasks must be authorized, detached and serialized):

```sh
cd /data/data/com.termux/files/home/vf_gemm_verify_20261002
sha256sum runner libwg_gemm_skel.so
export LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH=/data/data/com.termux/files/home/vf_gemm_verify_20261002
chmod u+x runner
time ./runner
```

Raw hash and complete sample output:

```text
79ee0f5042ab876af634bde491413eaa325afac213a71c23a10bf57695331002  runner
9f02f8d70ad26f0f700c54ba0048d7c0a82000a3431d7014fce5a3e3cfc147bf  libwg_gemm_skel.so
INIT rc=0 M=1024 N=12288 K=2560 full_fp64_reference=1 ABI=3 output=fp16
SAMPLE 0 rc=0 bad=0 nonfinite=0 checked=12582912 cosine=0.999999990309 maxrel=0.054209376 dsp_us=11991 RPC_us=12533.229 valid=1
SAMPLE 1 rc=0 bad=0 nonfinite=0 checked=12582912 cosine=0.999999990309 maxrel=0.054209376 dsp_us=11953 RPC_us=12452.188 valid=1
SAMPLE 2 rc=0 bad=0 nonfinite=0 checked=12582912 cosine=0.999999990309 maxrel=0.054209376 dsp_us=11907 RPC_us=12308.854 valid=1
SAMPLE 3 rc=0 bad=0 nonfinite=0 checked=12582912 cosine=0.999999990309 maxrel=0.054209376 dsp_us=11253 RPC_us=11894.062 valid=1
SAMPLE 4 rc=0 bad=0 nonfinite=0 checked=12582912 cosine=0.999999990309 maxrel=0.054209376 dsp_us=11233 RPC_us=13599.480 valid=1
CLOSE rc=0
```

Median RPC = 12.452188 ms; `2*M*N*K / (ms*1e9)` = 5.17375 TFLOPS.
Slowest RPC = 13.599480 ms = 4.737T. Prepacked RPC includes device work and
RPC overhead, **not host packing**. This is not a kernel-only or cold end-to-end
throughput number. Full fp64, cosine ≥ .999, max-rel < .1, NaN/finite and
canary checks remain the GEMM contract.

## FA historical r5 single-state-worker coreMAX

Source record: `kw_tl_rpc_fa_r5_coremax_frozen_20261001_coreaba_B/run.log`,
UTC 2026-10-01 15:05:34. Exact benchmark command/environment:

```sh
cd /data/data/com.termux/files/home/kw_tl_rpc_fa_r5_coremax_frozen_20261001_coreaba_B
sha256sum runner libwg_gemm_skel.so
export LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64
export ADSP_LIBRARY_PATH=/data/data/com.termux/files/home/kw_tl_rpc_fa_r5_coremax_frozen_20261001_coreaba_B
chmod u+x runner
time ./runner
```

Raw hash, contract and all ten sample lines (per-head detail omitted; each
sample retains its minimum-head cosine):

```text
c46070c991101e0691a56dbe4762c9d5e66e7e1ae4e5fedb1c7ef837f0c97ada  runner
efd4c41aec5fd30cc0c16caa98c5420227fd01adcda014c6f85e82d581bd39f2  libwg_gemm_skel.so
CONTRACT FA_HALF_V4 B1 HQ16 HKV4 Q1024 KV1024 D256 causal=1 samples=5 seeds=2 cosine_global_and_head>=.999 maxrel_diagnostic_only=1 reference_AP_only=1
SAMPLE seed=20260926 sample=0 rc=0 bad=0 nonfinite=0 checked=4194304 firstrow_bad=0 cosine=0.999999906172 minhead=0.99999989636 maxrel_diagnostic=0.0693161743 dsp_us=50734 RPC_us=52957.343 valid=1 first=1
SAMPLE seed=20260926 sample=1 rc=0 bad=0 nonfinite=0 checked=4194304 firstrow_bad=0 cosine=0.999999906172 minhead=0.99999989636 maxrel_diagnostic=0.0693161743 dsp_us=50685 RPC_us=53390.000 valid=1 first=0
SAMPLE seed=20260926 sample=2 rc=0 bad=0 nonfinite=0 checked=4194304 firstrow_bad=0 cosine=0.999999906172 minhead=0.99999989636 maxrel_diagnostic=0.0693161743 dsp_us=50698 RPC_us=53167.969 valid=1 first=0
SAMPLE seed=20260926 sample=3 rc=0 bad=0 nonfinite=0 checked=4194304 firstrow_bad=0 cosine=0.999999906172 minhead=0.99999989636 maxrel_diagnostic=0.0693161743 dsp_us=50770 RPC_us=53079.948 valid=1 first=0
SAMPLE seed=20260926 sample=4 rc=0 bad=0 nonfinite=0 checked=4194304 firstrow_bad=0 cosine=0.999999906172 minhead=0.99999989636 maxrel_diagnostic=0.0693161743 dsp_us=50747 RPC_us=51212.135 valid=1 first=0
SAMPLE seed=20260927 sample=0 rc=0 bad=0 nonfinite=0 checked=4194304 firstrow_bad=0 cosine=0.999999904766 minhead=0.999999895552 maxrel_diagnostic=0.0724223881 dsp_us=50812 RPC_us=53275.364 valid=1 first=1
SAMPLE seed=20260927 sample=1 rc=0 bad=0 nonfinite=0 checked=4194304 firstrow_bad=0 cosine=0.999999904766 minhead=0.999999895552 maxrel_diagnostic=0.0724223881 dsp_us=50708 RPC_us=53060.260 valid=1 first=0
SAMPLE seed=20260927 sample=2 rc=0 bad=0 nonfinite=0 checked=4194304 firstrow_bad=0 cosine=0.999999904766 minhead=0.999999895552 maxrel_diagnostic=0.0724223881 dsp_us=50726 RPC_us=56434.427 valid=1 first=0
SAMPLE seed=20260927 sample=3 rc=0 bad=0 nonfinite=0 checked=4194304 firstrow_bad=0 cosine=0.999999904766 minhead=0.999999895552 maxrel_diagnostic=0.0724223881 dsp_us=50726 RPC_us=53526.511 valid=1 first=0
SAMPLE seed=20260927 sample=4 rc=0 bad=0 nonfinite=0 checked=4194304 firstrow_bad=0 cosine=0.999999904766 minhead=0.999999895552 maxrel_diagnostic=0.0724223881 dsp_us=50699 RPC_us=53019.375 valid=1 first=0
CLOSE rc=0
```

Median of all ten = 53.1239585 ms (historically reported 53.123958 ms).
Raw host-pack / first-call totals make the excluded cost explicit:

```text
HOST_PACK seed=20260926 Q_us=3237.343 K_us=808.907 V_us=723.281 Android_actual=1
COLD_PACK_PLUS_FIRST_RPC seed=20260926 us=57726.874 excludes_init_ref_alloc=1
HOST_PACK seed=20260927 Q_us=4603.698 K_us=1244.896 V_us=1118.698 Android_actual=1
COLD_PACK_PLUS_FIRST_RPC seed=20260927 us=60242.656 excludes_init_ref_alloc=1
```

This is not the current default-source/dual-worker/fusion result, and the
21.55 ms hand-edited diagnostic is not a formal compiler result. No test
contract changes are made in this documentation batch.
