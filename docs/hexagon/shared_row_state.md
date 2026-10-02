# Shared row state: supported alternative to fragment recurrence

Use `T.alloc_shared((rows,), dtype)` for alpha/maximum/denominator state,
not `local.fragment`, when the intended ownership is explicitly by logical
row. Every stage accessing that state must use the same `physical_owner`,
worker count, and zero-based `T.Parallel(rows)` domain. Scalar state accesses
use exactly `state[row]`; a shifted index or different domain is rejected.

Initialize with a complete dominating fill. Same-owner persistent state is not
a ring-buffer edge: it stays allocated across pipeline iterations; ownership
wraps initialization in the owning group. Collective full-region copy/transform
boundaries receive group barriers before and after, permitting a collective
partition to hand data to a different, fixed row partition. Per-row staging
must remain genuinely worker-private; no shared staging reused by both workers.

Do not merely change `workers=1` to `2` with serial row loops. Do not use a
whole-vector state update spanning rows belonging to both workers. Explicit
`local.fragment` remains rejected for this recurrence because logical row
ownership alone does not prove physical Fragment layout ownership.

FA integration recommendation (DSL owner changes, not performed here): select
shared scope for the five row state arrays, retain matching Parallel row loops,
and rerun full emit plus target compile. This is an interface recommendation,
NOT a claim that the full FA workers2 kernel has passed those gates.

Host evidence: test_worker_shared_state_execution.py verifies actual DSL
planner/ownership acceptance and wrong-partition/different-owner rejection.
Its separate pthread oracle exercises two workers, 97 iterations, 32 repeated
launches, collective-to-cyclic-row handoff, and private stack staging. The
oracle is independently written C, not execution of generated worker code;
therefore it does not establish that every downstream allocation/layout pass
preserves this contract. That generated-code execution gate remains required.

Additional development probe (not yet a permanent regression):
`/tmp/opencode/shared_rows_codegen.py` emits a two-stage shared FP32 recurrence
through the full worker pipeline and compiles it with Hexagon clang -O2.
`/tmp/opencode/shared_rows_host.py` uses the same DSL with the existing
`tirx.disable_vectorize` option and emits unedited scalar C++ for host execution.
`/tmp/opencode/shared_rows_harness.c` runs that generated callback on the real
pthread adapter: 32 launches, seven iterations, two workers, poisoned output,
and a canary after the 512-byte state allocation. This probe passed.

The actual inferred row assignment in this probe is **32-row blocks**, not
cyclic scalar rows: `(iteration * 64) + local_worker * 32 + lane`. Initialization,
producer, and consumer use the identical mapping; the state is a depth-one
512-byte VTCM allocation with a fixed base throughout the callback. This is
evidence for this probe only. In particular, independently inferred layouts
for a FA matrix row loop and a scalar row-state loop must still be compared:
equal logical Parallel domains alone are NOT evidence of equal final mappings.
Do not claim full FA correctness before that check and full FA emit/compile.
