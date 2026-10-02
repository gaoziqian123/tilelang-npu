# Native FA continuation check

## Larger tile is not a deployed variant

The pending default change from q32/k128 to q128/k256 was inconsistent with
the collected source, registry symbol, and host launch (512 jobs). Restored
the q32/k128 defaults and added explicit `--block-q` / `--block-k` emission
arguments. The external tensor shape and numerical tests are unchanged.

Command (tilelang working directory):

```
.venv/bin/python examples/hexagon/fa/fa.py --block-q 128 --block-k 256 --out /tmp/opencode/fa_q128_k256.c
```

Raw result: `EMIT_OK /tmp/opencode/fa_q128_k256.c`.
Generated source requires 128 jobs and 852480 bytes of VTCM (lines 15/17),
not the registry's 512 jobs and 279040 bytes. Do not collect/deploy this
variant without updating the complete launch contract.

Actual object compilation, not syntax-only:

```
/root/hexagon-deps/HEXAGON_SDK/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/Tools/bin/hexagon-clang++ -mv79 -mhvx -mhvx-length=128B -mhmx -O2 -fPIC -fstack-usage -x c++ -std=c++17 -fno-exceptions -fno-rtti -I/root/project/tilelang/src -c /tmp/opencode/fa_q128_k256.c -o /tmp/opencode/fa_q128_k256.o
```

Exited zero with no diagnostics. Raw `.su` output:

```
/tmp/opencode/fa_q128_k256.c:14:fa_causal_b1_h16_g4_s1024_d256_q128_k256_w4_kernel	103808	static
```

This is the compiled function's static frame, not a bound on complete
runtime stack usage. No phone run or performance claim for this variant.

## Regression remains red

```
.venv/bin/python -m pytest testing/python/backend/test_hexagon_reduce.py testing/python/backend/test_hexagon_shared_gemm.py -q --tb=short
```

Raw result: `1 failed, 12 passed in 5.65s`.
Failure: `test_fa_full_lower_reproducible`, line 68,
`assert "tl::AllReduce<tl::MaxOp" in src`.
The existing row-owned lowering removed cross-worker AllReduce, but the
structural test still requires it. No assertion or tolerance was changed.
Next maintenance must preserve collective-reduction coverage separately and
add row-owned numerical/layout coverage, rather than simply deleting checks.

Independent verifier unavailable in this session (no subagent tool).
Overall task, larger-tile correctness, and >3T target remain UNVERIFIED.
