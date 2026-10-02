# Standard TIR VectorLowering support

`tilelang.hexagon.vector_lowering.VectorLowering()` is a PrimFunc pass, applied
at the end of the Hexagon pipeline before VerifyVectorRequired. Opt in using
`tl.vector_required=1`. Arithmetic additionally requires the explicit function
attribute `tl.hvx_arithmetic_mode="native_target_v1"`; absent/strict modes
reject rather than silently selecting target arithmetic.

The pass consumes standard vector BufferStore expression trees, not Python
chunk APIs or uint8 physical calls. Supported nodes: BufferLoad, Broadcast of
FloatImm, Cast between half/float, Add/Sub/Mul/Min/Max, Select of floating
EQ/NE/LT/LE/GT/GE, and static Shuffle. Standard != uses unordered semantics.
Loads/stores require compact rank-one buffers, static unit-stride Ramp indices,
128B offsets and allocation bounds. All logical lengths are whole 128B chunks;
half/float casts require complete paired chunks where appropriate. Dynamic
addresses, nonconstant broadcasts, arbitrary gather/permutation, scalar vector
tails and in-place expressions reject. Shuffle supports cyclic single-chunk
maps, halfword interleave/deal, and contiguous windows from two chunks.

Example standard TIR `b[ramp(0,1,128)] = cast<float16x128>(
cast<float32x128>(a[ramp(0,1,128)]))` lowers to two half loads, four
`widen_linear` results, two `narrow_linear` results and two stores, all physical
128B values. Half rounding is not optimized away. Arithmetic expressions are
recursively lowered without reassociation or FMA contraction.

Reductions are NOT yet supported: standard Reduce/unknown expression nodes
reject, because their tree/seed contract cannot be silently replaced by the
existing asc tree. Vector Let bindings and oversized vector-valued calls also
remain unsupported. This is a bounded expression-tree lowering, not an
arbitrary-map or full reduction implementation.

Final verifier whitelists named HMX/worker/DMA effect calls, while checking their
argument subexpressions. Explicit extraction is restricted to one terminal
Evaluate outside loops; use as arithmetic/cast input or repeated extraction
rejects. This deliberately limits useful scalar consumption as well, until a
verified terminal-result ABI is available.

Evidence: five standard-TIR -> pass -> production C codegen -> Hexagon -O2
object tests cover half128 cast roundtrip, half128/float96 elementwise select,
and half128/float96 shuffle; strict-mode rejection is a sixth test. No device
numeric evidence or independent verifier verdict. Pipeline end-to-end DSL,
reduction, symbolic address proof and alias-sensitive scheduling remain gaps.
