# Hexagon compiler / intrinsic checkpoint — 2026-10-02

This is a work-in-progress checkpoint, not completion of the full intrinsic
architecture. Only regular `target="hexagon"` is active. Retired legacy/v2
targets/selectors must fail closed, never become fallback paths.

## Entry points

- [GEMM DSL](../../examples/hexagon/gemm/README.md): prepacked DMA, resident B.
- [FA DSL](../../examples/hexagon/fa/README.md): worker pipeline and standard builder.
- [Historical evidence](checkpoint_evidence_20261002.md): small raw sample record,
  exact runner commands and hashes, readable without private temporary files.
- [Standalone runtime](../../../backend/npu/attn/tl_standalone/README.md) and
  [root paths](../../../npu-hetero-mlir/docs/CURRENT_PATHS.md) require the companion
  root checkout below.
- Contracts: [vector lowering](logical_vector_lowering.md),
  [math modes](logical_math_modes.md), [transform/cast](transform_cast_contract.md),
  [worker ABI](workergroup_v4.md), [prepacked DMA](prepacked_dma.md).
- [Unified HVX proposal](HVX_INTRINSICS_README.md) is a historical design document,
  not proof that every proposed API is implemented.

## Results and limits

GEMM M1024/N12288/K2560, BM64/BN256/BK1280, async/reuse-b/merge-edges/load1/
transform-output has a historical independent-verifier O2 device run: median
**12.452188 ms / 5.17375 TFLOPS**, slowest **4.737 TFLOPS**. These are complete
**prepacked RPC** times excluding host packing. Full fp64: 12,582,912 outputs,
cosine 0.999999990309, max-rel 0.054209376. This docs session did not rerun it.

FA r5 **single-state-worker + coreMAX** historical median is **53.123958 ms**
over two seeds / ten samples. Current dual-worker and fusion candidates are
**device UNVERIFIED**; 53 ms is not a measurement of current default source.
The approximately 21.55 ms hand-edited root reference diagnostic is **not
formal compiler output**.

The new standalone runtime is hexkl-free and defaults coreMAX on; explicitly
use `--no-core-max` for a control. Old independent handwritten backends still
depend on hexkl. Async worker scheduling does not prove nonblocking DMA.

## Checkout dependencies

Root branch `main` (remote alias `ternary`) and TileLang branch `gaoziqian`
(remote alias `npu`) are independent Git repositories. Root `tilelang/` is
**not a submodule**. Put this checkout there for root package builds and links.
Record both commit IDs when the owner later commits; this batch does not commit
or push. TVM is pinned at local revision `541b8b761`, which may not be available
upstream. Obtaining that source remains a reproducibility blocker until
published through an authorized channel. A clean clone is not promised to work.
Matching Python dependencies, rebuilt TileLang/TVM libraries, Hexagon SDK/tools
(v79, HVX128, HMX, actual O2 objects), Android NDK and authorized FastRPC/firmware
are required. Stale installed libraries and `out/` artifacts are not rebuilds.

## Remaining gates

Complete intrinsic coverage, effect/ownership proofs and collective validation
remain ongoing. The reported three `call_extern` escape sites require a fresh
lowering audit, classification and replacement or explicit justification; this
checkpoint does not claim zero extern calls. Validate partition, barriers,
causal last-active-K, tails, dual-worker state, fusion rounding, exceptional
math/casts, reduction order and absence of scalar fallback using independent
tests, real object/ASM review and full device runs. Native DMA submit/wait
lifecycle exists; cancellation/drain integration and overlap are separate gates.

No tests, fixtures or tolerances changed here. Historical red structural tests
must be reported, not weakened. Security review and independent verifier PASS
are still required before treating the pending source checkpoint as complete.
