# Native FA server checkpoint

2026-09-26: full DSL lowers and compiles to an actual v79 object. This is
NOT device correctness, a fast HVX softmax implementation, or the FA >3T gate.

Implemented native ReduceOp registration using the shared fragment ownership
planner and Hexagon AllReduce ABI; local reductions use serial standard TIR
(including clear=False, avoiding the common LowerLocal single-SeqStmt bug).
Also implemented infinity lowering/printing, scalar min/max printing, and
lane-wise fp32 exp code generation. Exp currently calls libm expf per lane,
preserving exp(-inf)=0. It is a correctness-first baseline, not vector exp.
Official HTP hvx-exp.h was inspected: range reduction by floor(x*log2(e)),
degree-seven polynomial, qf32 intermediate normalization and exponent insertion.
Its -88 clamp must not be copied blindly: causal -inf requires exactly zero.
No approximate HVX exp has been added in this checkpoint.

Multiple fragment GEMMs previously emitted nondeterministic scratch bindings
when their names collided. Scratch names now include the destination buffer;
no GEMM arithmetic/lifecycle was changed. Shared-output GEMM is untouched.

## Commands and evidence (working directory tilelang)

```
cmake --build build -j8
[100%] Built target tilelang
.venv/bin/python -m pytest testing/python/backend/test_hexagon_reduce.py testing/python/backend/test_tilelang_backend_module.py testing/python/backend/test_hexagon_shared_gemm.py testing/python/backend/test_hexagon_removed_targets.py -q --tb=short
44 passed in 8.37s
```

The 33 existing cases remain unchanged; 10 new cases execute generated scalar
reduction C++ against NumPy (sum/max/min/abssum/absmax, clear on/off), plus one
full FA structural/repeat-emission check.

```
.venv/bin/python examples/hexagon/fa/fa.py --out /tmp/opencode/fa_native.c
.venv/bin/python examples/hexagon/fa/fa.py --out /tmp/opencode/fa_native_repeat.c
sha256sum /tmp/opencode/fa_native.c /tmp/opencode/fa_native_repeat.c
f3c3e43aba5becfbce7c458717e30e348782e203eaf8733dee12a9e02b42c472  /tmp/opencode/fa_native.c
f3c3e43aba5becfbce7c458717e30e348782e203eaf8733dee12a9e02b42c472  /tmp/opencode/fa_native_repeat.c
/root/hexagon-deps/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang++ -mv79 -mhvx -mhvx-length=128B -mhmx -O2 -fPIC -fstack-usage -x c++ -std=c++17 -fno-exceptions -fno-rtti -I/root/project/tilelang/src -c /tmp/opencode/fa_native.c -o /tmp/opencode/fa_native.o
```

Compiler exit 0, no diagnostics; object text 11416 bytes. Stack-usage report:
22400 bytes static. Generated VTCM allocation: 229376 bytes, 128-byte alignment.
Launch: 512 jobs (head fastest, q-block next), four workers.

```
.venv/bin/python examples/hexagon/gemm/gemm_nt.py --m 1024 --n 12288 --k 2560 --np 384 --workers 6 --out /tmp/opencode/gemm_reduce_regression.c --skip-clang
sha256sum /tmp/opencode/gemm_reduce_regression.c
7f29bbfe435c90cbe6d2f35a4910808ef32e81f05b68d85262e6670d38a9c118  /tmp/opencode/gemm_reduce_regression.c
```

## Next ABI / performance work

* Actual generated argument order is **K, O, Q, V**, not DSL declaration order.
* New registry wrapper needs 512 jobs, four cooperative workers, 229376-byte
  VTCM slab and >22400-byte stacks plus call/runtime margin. Existing worker
  scratch/barrier and `tl_hex_hmx_acc_read_f16` owner-thread ABI are required.
* Undefined math/conversion symbols include expf, __extendhfsf2,
  __truncsfhf2 and memset: link audit is still required.
* FP32 online state is retained, but HMX fragment readout is still fp16 before
  widening. This existing GEMM contract requires full fp64 FA device validation.
* Scalar exp, conversions, local reductions, generic transpose/layout staging
  and fragment readout synchronization remain optimization work. Object compile
  is not evidence that VTCM accesses are efficiently vectorized throughout.
* No RPC registry, IDL, runtime, device artifact, generated C or existing test
  was manually edited. No commit. Independent verifier unavailable in this
  delegated tool set; parent must request verification.

VERIFIED: host checks and server object compile above; GEMM source hash unchanged.
UNVERIFIED: FA numerical device gate, link/deploy, >3T, optimized HVX exp,
cross-worker numerical execution, independent verification.
