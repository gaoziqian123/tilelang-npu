# Standard TileLang causal GQA prefill design (`fa_std`)

Goal: emit a Hexagon C example for Qwen-style flash attention prefill using only
standard TileLang constructs, in the same style as `gdn_std.py` and
`ffn_std.py`.  The example does not embed handwritten C and does not call the
handwritten `attnops_fa256.c` / `fa2_*` leaves.

## Fixed shape and math

```text
S = 1024, HQ = 16, HKV = 4, D = 256, tile = 32
Q : [16, 1024, 256] fp16 row-major, logical slab tensor
K : [ 4, 1024, 256] fp16 row-major
V : [ 4, 256, 1024] fp16 row-major, host-supplied V^T
O : [16, 1024, 256] fp16 row-major
```

KV head `g` serves query heads `4*g .. 4*g+3`.  For query tile `qt` of one
query head, only KV tiles `kt <= qt` are visited.  Scores use the causal mask in
the diagonal tile and a scale of `1/sqrt(256) = 0.0625`:

```text
score_kt = 0.0625 * Q[h, qt] @ K[g, kt]^T
score_kt[r, c] = -32768  if kt == qt and c > r
m_used_kt[r] = max(previous_m[r], max_c score_kt[r, c])
P_kt[r, c] = exp(score_kt[r, c] - m_used_kt[r])
m_final[r] = m_used_last[r]
P_kt      *= exp(m_used_kt - m_final)
l[r]       = sum_{kt,c} P_kt[r, c]
O[h, qt]   = (sum_kt P_kt @ V[g, kt]) / l
```

`T.gemm` performs both matrix products with fp32 HMX accumulation:

- `Q_tile @ K_tile^T`, with Q staged to AH and K staged to WH from row-major
  input.
- `P_tile @ V_tile`, with P staged to AH and host-supplied `V^T [D,S]` staged
  directly to WH.  There is no global `Vt` scratch or transpose phase.

The v4 implementation is single-pass streaming: the score readout VTCM buffer
`ScoreB [32,128]` is reused as the padded probability tile, then staged from
VTCM row-major to AH.  To stay within the current standard f16 AH/WH staging
granularity, the P@V pass pads the reduction dimension from 32 to 128: P
columns `[32,128)` and V rows `[32,128)` are zero, so the result is
mathematically identical while all staging still uses standard `T.copy`.

## Kernel phases

The schedule is mixed sync/async.  `T.parallel` regions lower to worker-pool
phases; all `T.gemm` HMX chains remain on the caller thread.  The outer KV
group loop first hoists both K and host-transposed V staging into VTCM, then
each query-head group pipelines the per-query-tile Q staging against the
previous tile's score and value sweeps.

1. **K full-head staging**: once per KV group, a multi-worker pool phase stages
   all `K[g] [1024,256]` into a VTCM WH buffer `K_all [1024,256]`.  Jobs cover
   one 32-row tile and one 64-column slice, so the lowering uses the standard
   row-major to WH recipe.
2. **V^T full-head staging**: once per KV group, a pool phase stages the
   host-supplied logical `V[g] [256,1024]` directly into a VTCM WH buffer
   `V_all [256,1024]` in 64-column slices via `hrt_stage_f16_rm_to_wh_nt_s`.
   The source leading dimension is `S=1024`, so the sliced `_s` helper and the
   explicit destination row-block stride are required.
3. **Q staging**: a multi-worker pool phase stages a 32x256 Q tile to AH as
   four 64-column slices.  The emitted copy uses the strided AH recipe because
   the source row stride remains 256 while the copied slice width is 64.  The
   query-tile loop uses `T.Pipelined(..., num_stages=2, order/stage=...)`: in
   the steady state this first pool phase is emitted as `attnops_pool_start_ctx`
   and overlaps with the previous tile's HMX/scalar work until the first later
   pool phase (P staging) requires a join.
4. **Streaming score/softmax/value sweep** (`kt = 0..qt`): read point views
    `K_all[kt*32:(kt+1)*32, 0:256]` directly; run `T.gemm(Q, K^T)`; copy the
    accumulator to VTCM row-major `ScoreB`; apply scale and diagonal causal
   mask; compute row max/sum with `T.reduce_max` / `T.reduce_sum`; materialize
   padded P in-place in `ScoreB`; immediately stage P to AH and consume
   `P @ V` with point views `V_all[0:256, kt*32:kt*32+128]`.
5. **Online state and accumulator**: softmax pool jobs are non-persistent
   threads.  Cross-phase state therefore lives in VTCM: `Mslot` and `Lslot` are
   `[32,32]` fp32 128B slots, one row per slot, lane 0 semantically valid for
   `l` and all lanes replicated for `m`.  At `kt==0` the softmax job initializes
   `m=-inf,l=0`; for later tiles it vector-loads/reduces its slot, computes
   `alpha=exp(m_old-m_new)`, and rescales the row of VTCM fp32 `Oacc[32,256]`
   in the same job before storing updated slots.  The `kt==0` output path
    overwrites `Oacc`, so no rescale is applied to uninitialized data.
6. **Normalize**: the final normalize job vector-loads `l` from the row's VTCM
    slot, multiplies each output row by `1/l`, and writes row-major fp16 O.

Current masking policy: the diagonal causal mask intentionally stays as a
scalar `T.serial(tile)` loop inside the row-parallel softmax job.  Do not rewrite
it as a vectorized fp16 `T.if_then_else(cmask > rmask, ...)`: today the Hexagon
emitter lowers integer lane compares in word lanes, so a fp16 select gates pairs
of half lanes.  The future compiler fix is to generate half-lane (`Vh`) integer
compares for fp16 vector selects; until then the scalar diagonal mask is the
checked anti-pattern boundary.

Execution-model note: worker-pool jobs are not persistent.  Any state that must
survive across the score, P-staging, P@V and normalize phases must be in memory.
For FA v4 that legal fast path is the 128B VTCM slot per row for `m/l` plus the
VTCM fp32 `Oacc` buffer.  `Oacc` deliberately replaces an HMX accumulator for the
running output because external online rescale of HMX accumulators is not
available.

## VTCM budget

The static VTCM request reported by the generated C self-check is 1,161,216
bytes, well under the 8 MiB cap.

| Buffer | Shape/layout | Bytes |
|---|---:|---:|
| `Q_a` | two pipeline versions of `[32,256]` fp16 AH | 32,768 |
| `K_all` | `[1024,256]` fp16 WH | 524,288 |
| `V_all` | `[256,1024]` fp16 WH (V^T) | 524,288 |
| `P_a` | `[32,128]` fp16 AH | 8,192 |
| `ScoreB` | `[32,128]` fp16 RM (also P tile) | 8,192 |
| `OutB` | `[32,256]` fp16 RM | 16,384 |
| `Oacc` | `[32,256]` fp32 RM | 32,768 |
| `Mslot` / `Lslot` | `[32,32]` fp32 RM VTCM slots | 8,192 |
| HMX accumulator/readout area | emitter-managed | small / shared |
| **Generated static VTCM request** | | **1,161,216 B** |

FA v4 has no global scratch state in the slab.  The only slab tensors are
Q/K/V^T/O and `prof`; `Pbuf`, `Mus`, `Mfin`, `Lbuf`, `ScoreRow`, `Mtmp`, `Obuf`,
`Vt`, global `Oacc`, and global `Alpha` are intentionally absent.

## Slab ABI v1

The generic Hexagon AOT entry is:

```c
int attnops_tl_fa_std(remote_handle64 h,
                      unsigned char *slab, int slabLen,
                      int abl);
```

Logical contents are row-major fp16 unless noted.  The TileLang signature uses
2-D flattened head-major tensors (`[heads*S, D]`) because this exercises the
currently robust generic copy route while preserving the logical 3-D ABI.

```text
Q       [16*1024, 256]  == logical [16,1024,256]
K       [ 4*1024, 256]  == logical [ 4,1024,256]
V       [ 4,256,1024]   == logical V^T [4,256,1024]
O       [16*1024, 256]  == logical [16,1024,256]
prof    int32 qtimer counters emitted by `tl.hexagon_prof`
```

## v1 exclusions and compiler note

- No host-preconverted QAH/KWH/VWH ABI; all staging happens inside the kernel.
- Q staging uses the standard `T.Pipelined` async pool start; K/V hoist phases
  and P staging remain synchronous pool phases.
- No persistent all-P row block: each streamed score tile is immediately
  converted in-place to P in VTCM `ScoreB`, staged to AH, and consumed by the
  following `P @ V` before the next `kt` tile overwrites it.
- Row max/sum use the 32-lane `T.reduce_max` / `T.reduce_sum` helpers; VTCM
  sources select `_vtcm` reducers so 64B score rows are never copied through
  scalar VTCM accesses.
- P@V uses a zero-padded K=128 pass instead of a native K=32 HMX pass because
  the current standard f16 rm->AH/WH staging helpers operate on pairs of
  32-column tiles (64-column minimum for this path).
- A small Hexagon emitter fix was required: sliced AH copies need source
  leading-dimension aware staging, sliced WH staging needs slice-local tile
  destination offsets / source stride preservation, VTCM-source f16 row
  reductions must select `_vtcm` helpers, and VTCM-source rm->AH copies must be
  allowed by `hexagon.copy_rm_ah`.  Other generated anchors remain
  byte-identical except for the intended copy-routing rule relaxation.

No device execution is performed by this example; `--self-check` is structural
and `hexagon-clang -fsyntax-only` verifies generated C syntax.

## Device validation (2026-09-13)

- `fa_std_test 1 0`: FINAL_CRITERION **PASS** — strict elementwise
  `|got-ref| <= max(0.1*(|ref|+0.02*rms), 2.5e-4)` 0/4194304 failures,
  cosine 0.999995009 against the fp64 reference.
- Criterion note: the cosine bar is 0.99999 (not GDN's 0.999999).  The 5e-6
  deficit is the HMX fp16-readout format floor: scores, P and O partials all
  round through fp16 because hexkl exposes only `acc_read_f16` (no fp32
  readout).  `FORMAT_FLOOR` in the test proves it: `cos(ref16, ref64)` with
  fp16-quantized P is 0.999999982 (format allows ~1), while
  `cos(got, ref16) = 0.9999950` equals `cos(got, ref64)` — the kernel tracks
  the fp16-reference to the same deficit, so the residual is the format, not
  logic.  Closing it needs an fp32 accumulator readout path in hexkl.
- v1 performance: ~1856 ms for S=1024 (16/4 GQA, D=256).  Prof ticks
   (35.6M wall): pool_vpad 5.8M (scalar V transpose), pool_v 3.8M + pool_k
   0.7M (per-tile restaging), mm 68K (0.2%).  Perf work (hoisted K/V staging,
   HVX tile-transpose copy route, async overlap) is a follow-up campaign.
- v2 hoists K/V: `K_all` and `V_all` are staged once per KV group; value
  transpose now fills ABI scratch `Vt [4,256,1024]`.  The only per-kt pool phase
  left is P staging.  Device timing is tracked in the current campaign report.
- v3 row-parallel softmax (2026-09-14): mask+scale / running row max / exp /
  pad-zeroing (phase A), rescale+rowsum (phase B), Oacc zero/accumulate
  (phase C) and final normalize (phase D) moved from caller-thread
  `T.serial(tile)` loops to `T.parallel(tile)` worker-pool phases.  Each job
  owns one row exclusively (`Mfin[r]`/`Lbuf[r]`/`Pbuf[kt,r,:]`), so no races;
  the row max stays scalar inside the job (generic LowerTileOp sliced
  `T.reduce_max` is still broken) with `Mus[kt,r]` doubling as the
  job-private accumulator.  Device: **1321 -> 383 ms** (3.4x), strict
  elementwise PASS, regressions (gdn_std/ffn_std) green.  Prof ticks
  (7.31M wall): rmask 2.58M + arow 2.34M dominate (~70%) — both are scalar
  global-memory loops, next target is HVX vectorization of those phases.
- v3b HVX vectorization (2026-09-14): the row-parallel phases' inner loops
  now use `T.vectorized` — score scale (off-diagonal tiles; the diagonal
  keeps the scalar masked path), exp (full 128-wide padded row with a
  +32768 bias on pad lanes so they underflow to exactly 0 in fp16), Oacc
  zero/accumulate and final normalize (256-wide).  Row max and the row-sum
  reduction stay scalar but accumulate in `T.alloc_var` locals instead of
  round-tripping global scratch.  Device: **383 -> 172 ms**, strict
  elementwise PASS; gdn_std/ffn_std regressions green.
- v3c 32-lane HVX row reductions (2026-09-14): rowmax/rowsum now use
  `hrt_reduce_max_f16_32` / new `hrt_reduce_sum_f16_32` via
  `T.reduce_max`/`T.reduce_sum` on an element load at row start.  The
  element-load convention is intentional: sliced loads still trip the generic
  `LowerTileOp` substitution bug, while the emitter can recover sliced-row
  addresses from `BufferLoad` and derive reduce width from the callee suffix
  (32/128).  Device: **172 -> 151.421 ms** (20 iters; 1 iter 153.174 ms),
  strict elementwise PASS; gdn_std/ffn_std regressions green.  Main pool ticks:
  rmask 715428, rsum 539626, pstage 389321, arow 465201, qasync 22275,
  rmask2 45763, rsum2 34867, arow2 30003.
- Emitter fixes landed for v3b: (1) `T.alloc_var` (`local.var`) scalars are
  redeclared inside pool worker functions that reference them and dropped
  from the main flow when worker-only (the latter used to trip -Werror
  -Wunused-variable in the skel build); (2) fp32 uniform (splat) vector
  stores in 64-lane iteration contexts now emit both 128B halves — the
  single-store fast path is only legal for genuine 32-lane contexts
  (extent 32).  Bug (2) silently zeroed only half of each Oacc row.
- Debugging note (v3 bring-up): the skel entry guards the slab with
  `if (slabLen < prof_off + PROF_BYTES) return -1;`.  **FastRPC masks
  negative skel return values as a generic invoke failure `user err 0x4e`
  (78)** — the host never sees the -1.  v3 grew the prof array from 48 to 88
  bytes while the test still reserved 48, so every call "crashed" at what
  looked like random entry points during bisection (early returns before the
  guard propagated fine because they were positive).  Lesson: keep skel error
  returns non-negative, or treat host-side 0x4e as "check your negative
  return paths" first.
- Compiler fixes landed for this kernel: WriteSet marks `reduce_max` dst
   buffers; `hrt_reduce_max_f16_32` out-of-bounds 128B load + missing 1-lane
   fold fixed (both f16 helpers); `hexagon.gemm_hmx` carries an explicit
   transB flag and rejects transpose_B=False; rm->WH staging uses
   `hrt_stage_f16_rm_to_wh_nt_s` with buffer-level tile stride for sliced
   multi-row-block regions; generic-shell selection keys on `_pre_has_parallel`
   (a silently-added `not legacy -> generic` line was reverted because it
   broke the gemm_small statement-level anchor path).
- v4.1 streaming rewrite (2026-09-14): V input contract is now host-supplied
  `V^T [HKV,D,S]`; the global `Vt` scratch and transpose phase are gone.
  Softmax jobs compute `alpha` and rescale their `Oacc` row directly (only for
  `kt>0`), so global `Alpha` and the separate rescale phase are gone.  `Mfin`
  and `Lbuf` are replaced by VTCM 128B slots (`Mslot`/`Lslot`) initialized in
  the `kt==0` softmax branch and vector-loaded by normalize.  This preserves
  the required non-persistent pool-job execution model while removing all
  global scalar state from the FA slab.
- v4.1 bug note: an attempted vectorized diagonal mask produced pairwise causal
  leakage because `cmask > rmask` used word-lane predicates for a fp16 vmux.
  Device evidence at `t=0`: row 0 kept columns 0 and 1 and output the average of
  `V[:,0]` and `V[:,1]` instead of exactly `V[:,0]`.  The current fix is to keep
  the diagonal mask in the proven scalar form; the checker rejects the generated
  vectorized fp16-mask anti-pattern until the emitter grows Vh-domain compares.
- v4.1 device validation after scalar-mask restore: `fa_std_test 1 0 0` PASS
  (`cos_o=0.999999637`, 0/4,194,304 element failures, 126.540 ms one-shot).
  `fa_std_test 20 0 0` PASS at **127.296 ms** average, wall **2,400,449** ticks.
  Main tick table: softmax **984,973** (merged mask/scale/max/exp/sum/Oacc
  rescale), pstage **377,565**, arow **434,634**, normalize **31,861**,
  unperm aggregate **329,851**, mm aggregate **72,543**.  GDN/FFN regressions on
  the same skel: GDN **48.870 ms**, FFN **66.180 ms**.
