# Physical HVX leaf implementation status

This is a **partial implementation**, not completion of HVX_INTRINSICS_README.md.
No device validation or independent verifier verdict has been obtained.

## Callable API

`from tilelang.hexagon.language.vector import leaf, permute, permute2`

All values are explicit `uint8x128` raw register views; the operation names carry
the numerical interpretation. `leaf(name, *args, mode=..., immediate=0)` emits
`tl.hexagon.vector_leaf(name, mode, immediate, *args)`. Both frontend and C++
codegen check the closed operation set, arity, type, immediate and mode. Calls
remain opaque because domain rejection may trap. No kernel-name matching or
whole-operator helper is used. This API does not yet provide memory proofs or
logical vector lowering, and must not be presented to FA integration as such.

| Operation | Implemented contract |
|---|---|
| rotate | `bits_v1`, static byte shift 0..127 |
| align | `bits_v1`, output byte j = [second,first][j+shift] |
| interleave16 / deal16 | `bits_v1`, halfword shuffle and its inverse |
| permute | frontend maps only cyclic lane maps, halfword interleave/deal |
| permute2 | frontend maps only contiguous windows of [a,b] |
| select | `bits_v1`, each byte mask == 255 selects a, otherwise b; not a typed lane predicate |
| abs16/32, neg16/32 | `bits_v1`, clear/toggle sign; preserves payload bits |
| add/sub/mul/min/max 16/32 | explicit `native_target_v1`; no strict NaN/zero promise |
| sum32_asc/max32_asc | `native_target_v1`, shifts 4,8,16,32,64, no seed |
| widen_linear | `strict_exact_v1`, part 0/1; reuses cast_layout.h |
| narrow_evenodd | `rne_gradual_quiet_payload_v1`; reuses cast_layout.h |
| rcp32_nr2 | `nr2_positive_bounded_v1`; vector guard for [2^-60,2^60], then two Newton steps; no new error bound claimed |

## Still missing (not silently delegated away)

### Follow-up implementation

Added `broadcast16/32` (static lane immediate), `widen_evenodd`,
`narrow_linear`, `div32_nr2`, and `exp2_16_bounded`. The last uses the existing
HF/QF polynomial unchanged, through `exp2_hf_impl<true>`, and traps on inputs
outside [-14,0] (including NaN/infinity). Mode is
`qf_poly6_neg14_0_v1`; no measured or proven total numerical error bound is
claimed. The legacy `exp2_hf` still calls `<false>` and retains its repair.
Approximate division explicitly means `mul32(a,rcp32_nr2(b))`, NOT correctly
rounded IEEE division. Both new cast orders retain their existing mode names.

Read upstream `reference/llama-cpp-upstream-htp/ggml/src/ggml-hexagon/htp/hvx-exp.h`:
its exp2 clamps at -24, and its exp tail reads a complete HVX_UVector before
storing a partial result. Neither is a valid substitute for strict subnormal
semantics or no-overread masked loads. The SDK prototype search found
predicated stores, but no corresponding byte-masked load. A safe vector tail
implementation still needs a demonstrated alternative, not this upstream read.

The added math target-object test compiles the complete bounded exp2 path,
cast-order adapters, broadcast, and approximate division at -O2 to object/ASM.
Both test files together: four tests passed. No device evidence or independent
verifier result. These additions do not resolve the full architecture blockers.

Aligned/masked memory IR with allocation/alias/owner proofs, safe no-overread
tails, scalar splat/extract, typed compare/predicates/bitcast, FMA with a
proven rounding contract, exp32 and full-domain exp2 without libm repair,
logical VectorLowering and function-wide
`vector_required` fail-closed verification. Unsupported names/modes in this
new API reject, but this does **not** police existing generic TIR paths.

The original helpers retain scalar fallback; they are not safe substitutes for
these missing strict paths. Reductions extract lane zero once and splat it after
the butterfly, ensuring bit-identical output lanes despite floating-point
lane-order differences. Seeded and cross-chunk reductions remain unimplemented.

## Evidence and gaps

`cmake --build build -j4` built the modified builtin/codegen successfully.
The new test file's three tests passed using the command in the implementation
report. Actual Hexagon `-O2 -c` and `-O2 -S` compile all added C++ leaves.
The ASM gate checks presence of vror/vadd and absence of named exp/div helpers;
it is **not** a complete per-instruction scalar/spill audit or numeric oracle.
Mapping tests check frontend selection and offsets, not execution on Hexagon.
End-to-end generated-C compilation and exhaustive cast bit tests remain pending
for this API. Existing tests/fixtures/tolerances were not modified.
