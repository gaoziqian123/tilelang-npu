# Standard Hexagon FA checkpoint

[Unified index](../../../docs/hexagon/README.md) ·
[Historical evidence](../../../docs/hexagon/checkpoint_evidence_20261002.md).

Regular source: `fa_worker_pipeline.py` → `build_worker_standard.py` → generated
query-block kernels, dispatcher and packers → root standalone FA v4 RPC package.
This is not retired `fa_std_v2` / `fa_online_v2`.

Build recipe from TileLang (not run in this docs batch):

```sh
python examples/hexagon/fa/build_worker_standard.py \
  --output /tmp/opencode/fa_checkpoint_new --state-workers 1 --intrinsic-rows
```

Output must be fresh. Override `--compiler` if the installed SDK path differs.
The builder performs O2 object/ASM compilation for
B1/HQ16/HKV4/S1024/D256/BQ128/BK256. Root `package.py --kind fa` also needs
independently generated two-seed fp64 input/reference files, not included in
this small docs checkpoint. Never derive references from DSP intermediates.

Historical **r5 single-state-worker + coreMAX**: two seeds × five samples,
median **53.123958 ms prepacked RPC**, excluding host pack. Same DSL/builder
lineage, **not a device measurement of current default source**. Current
`--state-workers 2` and fusion candidates remain device UNVERIFIED.
The half-state contract retains full 4,194,304-element fp64 reference,
global/per-head cosine ≥ .999, exact causal first row, NaN/finite and canary
checks; max-rel is diagnostic. No gate changes here.

Root `reference/fa_Btwo_spin1024` approximately **21.55 ms** hand-edited
diagnostic is not formal compiler output. Historical BUILD/CONTINUATION files
are earlier experiment status, not current-tree validation. Next gates include
collective/partition and terminal-effect proofs, extern-call audit, preserved
rounding across fusion, object/ASM review and independent device revalidation.
