# Explicit HF operations (experimental, not production enabled)

`tilelang.hexagon.language.hf` exposes separate IR operations. Neither `T.exp`
nor ordinary reductions are rewritten to these operations.

## Representation and boundaries

* DSL inputs/outputs are IEEE binary16. `sub_hf(a,b)` performs HF -> QF16
  subtraction -> HF conversion. QF16 is **not** an IEEE value: its register
  encoding is confined to `qfloat16_register` inside the target leaf. It cannot
  be stored in a DSL float16 buffer. No public QFloat arithmetic chain exists yet.
* `exp2_hf(x)` computes an explicit approximate base-two exponential, not exp(x).
  On [-14,0] it uses the upstream QF16 polynomial, with HF conversion before
  exponent reconstruction. Proposed acceptance gate: absolute error <= two
  binary16 ulps versus independently rounded double `exp2`. This is a gate to
  verify, **not a measured guarantee**. Exhaustive 65536-bit-pattern testing and
  device stage comparisons are required before enabling this path.
* Outside [-14,0], including subnormal-result inputs, a cold libm double exp2
  fallback rounds to HF. NaN -> NaN (payload unspecified), +inf -> +inf,
  -inf -> +0, both signed zeros -> 1. Half underflow uses round-to-nearest,
  ties-to-even as supplied by the target conversion. Target libm linkage and
  conversion behavior remain device validation gates. Scratch in this fallback
  is thread-local ordinary memory, never VTCM.
* Codegen currently accepts scalar, 32 and 64 lanes only. Inactive lanes are zero
  padded and never stored. Arbitrary tails must be guarded by the caller.

## Mapping to upstream fcc891

Reference: `ggml/src/ggml-hexagon/htp/flash-attn-ops.c:1280-1469`.
Existing compiler facilities: HF/SF casts, packed AH/WH copy and layout shuffle,
HMX GEMM, reductions, paired cyclic worker ownership. New interrupted-session
WIP: explicit rounded subtract and exp2 HF builtins/native C++ codegen/leaves.
Missing: public typed QFloat chain, HF vector max/reduction, explicit QF32 sum
chain and matching upstream reduction order, ordinary mxmem D@O+P@V variant.

`fa_tile_schedule.make_fa(explicit_hf=True)` is a default-off diagnostic DSL
variant. It exposes base-2 score HF rounding, rounded subtraction, HF exp2,
HF probabilities widened for sum, and persistent SF m/l/output. The independent
`round_local_sum=True` switch adds upstream's local sum HF -> SF boundary.
It intentionally retains SF reduction/output accumulation to isolate error;
it is **not yet an implementation equivalent to the full upstream flow**.
Upstream Q/O are F32; this project's Q/O remain F16. Score scaling placement
and HMX accumulation rounding therefore need independent stage comparison.

All original FA checks remain mandatory (full 4194304 outputs, fp64 rms < .1,
old relative < .1, cosine, NaN poisoning, nonfinite scan). The FP32 exponential
contract stays `2e-5*abs(reference)+2 ulp`. No new tolerance permits a failed
FA candidate to pass. No timing or correctness claim is made here.
