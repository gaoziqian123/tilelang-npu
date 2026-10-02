# Logical math integration

Standard `T.exp(float32 vector)` lowers to `exp32_v1` only with function attr
`tl.hvx_exp_mode="hvx_math32_v1"`. This is the approximate contract in
vector_math32_v1.md, including its still-unverified device accuracy target.
Validation must use `-ffp-contract=off`, no fast-math. Integration does not
change the independently supplied vector_math32.h.

Standard `T.exp2(float16 vector)` requires
`tl.hvx_exp2_mode="ggml_qf16_clamp24_v1"`. This preserves the reference HF/QF
evaluation, accepts nonpositive finite values and -inf, rejects NaN/positive
inputs and is not strict IEEE exp2.

Standard FP32 division requires
`tl.hvx_div_mode="nr2_positive_bounded_v1"`. It means multiply by the existing
two-Newton reciprocal with denominator [2^-60,2^60]. `1/x` uses this same
explicit mode; no strict division is silently approximated. FP32 single-round
FMA remains unsupported. All missing modes reject during VectorLowering.

Raw predicate leaves `pred_and`, `pred_or`, `pred_not` use `bits_v1` and accept
128B bit masks. Typed compare produces 0xff/0 bytes, so these operations preserve
its canonical masks. Arbitrary byte values are bitwise operands, not implicit
boolean conversion. Existing select chooses a only for a 0xff mask byte.

Layout support remains cyclic single-chunk maps, halfword interleave/deal and
two-chunk contiguous windows. No arbitrary gather is claimed.

Cross-worker reduction is NOT the register asc tree: existing reduce.h
AllReduce uses descending xor partner offsets threads/2 down to scale, with
publish barrier before every read and consumption barrier before partial reuse.
It uses full-team barriers; it must not be called by only one worker subgroup.
Its rank-dependent operand ordering is not guaranteed bit-identical across
workers. Unifying this with a canonical subgroup tree/lifetime remains pending;
do not substitute it for the logical asc reduction contract.
