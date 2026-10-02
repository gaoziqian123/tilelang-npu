# Public T.transform support and cast contract

`T.transform(src,dst)` has logical elementwise copy/cast semantics. Equal explicit
logical shapes are required. Actual annotated layout maps determine addressing;
buffer names and kernel names play no role. Distinct storage is never elided.
Overlapping storage is rejected, including identity aliases.

Current Hexagon matrix:

| Source -> destination | Layout-only / dtype-only / fused |
|---|---|
| half -> half | bit-preserving layout-only |
| half -> float | dtype-only or supported permutation then exact widening |
| float -> half | exact RNE narrowing then supported permutation |
| float -> float, other dtypes | rejected |

Layout domain: explicit compact static rank-two regions, <=65536 elements,
columns divisible by 64; contiguous destination vectors; identity or even/odd
deinterleave from at most two source blocks of 64 elements. Destination must be
injective shared/VTCM storage. Sources may be shared, single-worker local or
nonreplicated fragments with one proven writer owner per complete vector.
Bounds, owner crossings, incomplete vectors and unsupported permutations fail
closed, not a scalar VTCM fallback. General interleave/transpose is not promised.

Narrowing uses existing `half_bits_rne`: round-to-nearest ties-to-even, gradual
underflow, signed zero, infinity, overflow to infinity, NaNs quieted with sign
and retained high payload bits. Widening uses integer vector exponent/mantissa
expansion: every finite half is exact, including subnormals; NaNs are quieted
with sign/payload preserved. Neither operation depends on floating rounding
environment or native conversion FTZ behavior. No qf numeric cast is required:
integer HVX vector shifts/selects implement the exact semantics. Do not claim
native sf/qf conversion instructions were used for this implementation.

## O2 audit correction (2026-09-30)

The earlier integer-only codegen claim described source intent, not adequate
ASM proof. SDK19 scalarized 64B unpack and aggregate payload repair into scalar
loads plus vinsert. Current widening uses explicit Q6_Wuw_vunpack_Vuh on a full
128B source vector; narrowing now uses cast_linear_f32_f16 -> half_pair_rne ->
Q6_Vhf_vcvt_VsfVsf plus vector sign/NaN repair and vdeal. Empty inline-asm `+v`
register constraints prevent LLVM SROA from scalarizing these vector inputs;
they introduce no instruction, runtime helper or kernel-specific match.
Thus narrowing IS native sf/qf32/hf conversion plus integer repair; the earlier
integer-only statement above is superseded for the current Hexagon codegen.
Host narrowing retains the exact integer reference. Widening remains integer
vector expansion, not FP32 arithmetic used to simulate the half GEMM.

Fresh evidence: /tmp/opencode/transform_hvx_formal_20260930. In
float32_float16_False.s lines 110-165 the dataflow is vector shifts/masks,
qf32->hf, vmux sign/payload repair, vdeal and vmem store. In
float16_float32_True.s lines 105-129 the transform begins after input staging:
vmem -> vdeal -> vshuff -> vunpack. The scalar memuh/memh loop at lines 86-104
is the separate T.copy input-layout staging, NOT T.transform; the complete
microkernel therefore is not claimed scalar-free. audit.json lists all scalar
memory sites for independent provenance review, not an automatic proof.
The narrowing cases and widening identity have no vinsert/vextract/memh/memuh
matches after the fix. Complete scalar-address/spill provenance review is still
required before declaring every supported path fully audited.

Conversion and shuffle compose as vector expressions, without added DDR/VTCM
scratch allocations. Compiler stack spills remain possible and are not claimed
absent. The half-only path does not insert float conversion; standard BN256/512
generated C hashes remain unchanged.

Validation artifacts: `/tmp/opencode/compiler_finalize_20260930_cast/`.
`emit_cast.py` emits four dtype-only/fused cases; all compiled with Hexagon SDK19
`-O2 -x c++ -std=c++17 -mv79 -mhmx -mhvx -mhvx-length=128B` to object and ASM.
`widen.cpp` + `reference.py` compare the integer widening implementation against
NumPy for all 65536 encodings, explicitly quieting reference NaNs.
Host cast regression: 20 passed; neighboring worker/transform regression: 34
passed. Initial host attempts failed because clang++ was absent and g++ lacked
_Float16; the successful check uses a shared C++ implementation and independent
NumPy reference instead. First full object/ASM batch timed out at 200s; rerun
with 600s completed. No device execution or performance claim is made.

The four generated transforms were compiled, not executed on DSP. Full fused
layout+cast end-to-end numeric verification and independent verifier review of
these new changes remain outstanding. Runtime singleton/DMA changes are external
to this compiler task and must be integrated/sealed by the runtime owner.
