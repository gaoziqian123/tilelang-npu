# Standard Hexagon GEMM

## Current prepacked checkpoint

[Unified index](../../../docs/hexagon/README.md) ·
[Historical evidence](../../../docs/hexagon/checkpoint_evidence_20261002.md).
The measured recipe uses `gemm_prepacked_dma.py`, not the `gemm_nt.py` example
below. From the TileLang checkout with matching Python/compiler libraries and
an existing output directory (recipe, not executed in this docs batch):

```sh
python examples/hexagon/gemm/gemm_prepacked_dma.py \
  --M 1024 --N 12288 --K 2560 --BM 64 --BN 256 --BK 1280 \
  --schedule async --reuse-b --merge-edges --load 1 --transform-output \
  --output /tmp/opencode/standard.c \
  --host-packer-output /tmp/opencode/standard_pack.h
```

This emits C, JSON physical/ABI manifest and host packers, without deployment.
Use root `backend/npu/attn/tl_standalone/package.py --kind gemm` with
`--generated` / `--packer` pointing to those outputs and a fresh `--output`
directory for actual O2 compilation. Never hand-edit generated C. Keep BK1280:
changing BK changes partial-sum rounding. Packed inputs and fp16 row-major C
are ABI requirements.

Historical verifier median **12.452188 ms / 5.17375T**, worst **4.737T**, is
full prepacked RPC excluding host pack. Full fp64: 12,582,912 outputs,
cosine **0.999999990309**, max-rel **0.054209376**. These numbers belong to
the evidence hashes, not an unmeasured new build.

## Other standard example

`gemm_nt.py` uses the standard `tilelang.hexagon` backend (`target="hexagon"`),
ordinary `T.copy` / `T.gemm`, shared AH/WH layouts, and native HMX fp16 readout.
The legacy statement emitter and experimental v2 backend have been deleted;
their target names are rejected, not aliases for this backend.

The persistent-panel DSL stages A once and streams B panels through cooperative
workers, with HMX operations serialized on the owner thread. The runtime must
provide that owner and cooperative worker pool. Hardware primitives remain in
`src/tl_templates/hexagon/`.

Regenerate the six-worker configuration without deployment:

```bash
python examples/hexagon/gemm/gemm_nt.py --workers 6 --out /tmp/opencode/gemm_nt.c --skip-clang
```

Omit `--skip-clang` with `HEXAGON_SDK` configured to compile a DSP object.
This script does not deploy or run on a phone. Existing `out/gemm_nt*.c`
standard-backend artifacts are retained; historical performance evidence and
reference implementations are not replacement generators.
