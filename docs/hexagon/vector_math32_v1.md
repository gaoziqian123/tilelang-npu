# Independent HVX binary32 math leaves — contract v1

Integration surface (C++17, Hexagon v79, 128-byte HVX):

```cpp
#include "tl_templates/hexagon/vector_math32.h"
HVX_Vector y = tl::hvx_math32_v1::exp_f32(x);
// constexpr false:
tl::hvx_math32_v1::has_single_rounding_fma_f32;
// Instantiating fma_f32(a,b,c) produces a compile-time diagnostic.
```

One register contains 32 binary32 bit patterns, unchanged lane order. No memory,
alignment, shape, FA, worker, or scalar fallback requirements. No shared headers,
builtins or codegen are changed by this addition. Integration is a separate task.
Do not silently replace a strict exp operation with this approximate API.

## Exp numerical contract

This is **approximate**, not strict IEEE exp, correctly rounded exp, or an FMA
implementation. Version 1 accepts every binary32 input bit pattern:

* Both signed zeros return exactly `1.0f`.
* `+inf -> +inf`, `-inf -> +0`.
* NaNs preserve sign/payload and set quiet bit `0x00400000`. Signaling NaNs are
  sanitized before floating arithmetic; no fenv/exception-flag guarantee.
* Positive finite input above bits `0x42b17217` returns infinity. The next float,
  `0x42b17218`, exceeds the binary32 finite exponential range.
* Negative finite input at or below `-104` returns positive zero. In the remaining
  underflow tail, integer significand shifts implement gradual underflow with
  ties-to-even rounding of the **approximation**, not the exact exponential.
  There is no intentional FTZ output policy. Close to half-min-subnormal the
  approximation may choose the adjacent representable value.
* Accuracy acceptance target on finite non-overflow inputs:
  `abs(y - exp(x)) <= 2e-6 * exp(x) + 2^-149`. This is a fixed v1 target tested
  on a host IEEE-operation model; **device conformance remains UNVERIFIED**.
  It is not a proof over all binary32 inputs or all target arithmetic modes.
* Requires ordinary round-to-nearest binary32 sf arithmetic, no reassociation;
  validation builds use `-ffp-contract=off`, not fast-math. Input subnormals may
  be flushed by target arithmetic without materially affecting exp near one;
  output subnormals are constructed entirely as integer bits.

Range reduction uses nearest integer `k`, split ln(2), and degree-eight Taylor
on approximately `[-ln(2)/2, ln(2)/2]`. Integer exponent reconstruction handles
normal and subnormal output without floating scaling/FTZ. Special values are
handled with vector predicates and integer muxes, not per-lane repair. The
constant eight-step polynomial loop is unrolled by the target compiler.

Sources reviewed: reference ggml `ggml-hexagon/htp/hvx-exp.h`, and the current
`src/tl_templates/hexagon/exp.h`. The former uses a floor-based qfloat polynomial
and exponent insertion; the latter retains a cold per-lane libm repair. This
new implementation instead uses split reduction and sf polynomial operations
to permit a transparent host operation model and integer gradual underflow.
It does not claim bit identity with either prior implementation.

## Explicit single-rounding FP32 FMA: rejected on this HVX target

SDK examined:
`/root/hexagon-deps/HEXAGON_TOOLS/Tools/lib/clang/19/include/hvx_hexagon_protos.h`.
At lines 5005–5024 it exposes:

```
Vxx32.sf += vmpy(Vu32.hf,Vv32.hf)  Q6_Wsf_vmpyacc_WsfVhfVhf
Vd32.sf  = vmpy(Vu32.sf,Vv32.sf)   Q6_Vsf_vmpy_VsfVsf
```

The accumulating instruction takes **half inputs**, not arbitrary binary32
inputs. No sf*sf+sf vector intrinsic is exposed. qfloat multiply/add sequences
do not establish IEEE single-rounding FMA and are not substituted. Scalar
`hexagon_protos.h:2754–2777` does expose `Q6_R_sfmpyacc_RR` / `_lib` / scale
variants (`F2_sffma`); those are scalar, not a pure HVX implementation. Thus this
is a rejection of **supported pure-HVX FP32 FMA mode in this SDK/target**, not a
claim that the Hexagon scalar ISA has no FMA, or that software integer emulation
is impossible. No software fused emulation is implemented here.

Counterexample checked with exact Decimal arithmetic:
`a=1+2^-23, b=1-2^-23, c=-1`. A single final rounding gives `-2^-46`;
binary32 multiply followed by add gives zero. The public API cannot silently
use the latter, including when explicitly instantiated with template `<true>`.

## Reproduction and evidence

From `tilelang/`:

```
python testing/python/backend/test_hexagon_vector_math32_v1.py
```

The standalone suite imports no TileLang/TVM. It prints the exact target commands
and their raw compiler output. It performs actual `-O2 -c` **and** `-O2 -S`, then
checks for vector arithmetic and absence of `call`, `callr`, `vextract`. A negative
compilation instantiates FMA and requires the explicit unsupported diagnostic.
Temporary target objects/assembly are removed after the check. Override compiler
via `HEXAGON_CXX`; missing compiler is a skip, not target verification.

Compiler used: QuIC LLVM Hexagon Clang 19.0.02, target hexagon. Recorded host run:

```
IEEE-model Decimal70 samples=21255 max_normal_relative=8.94837239e-08
Ran 3 tests in 2.627s
OK
```

The 70-digit Decimal reference is independent of the polynomial. Samples include
raw random bits, uniform domain samples, neighboring floats at reduction
boundaries, overflow, normal/subnormal, and subnormal-to-zero transitions.
Special bit patterns and the FMA counterexample are checked separately.
The model uses IEEE rounded scalar operations, **not an HVX emulator**; host
results must not be reported as device numerical results. No device run,
performance measurement, deployment, or commit was made. Independent verifier
is unavailable in this session (no subagent tool); final acceptance is UNVERIFIED
until separately reviewed and, for device numerical claims, tested on target.
