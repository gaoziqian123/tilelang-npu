# Standard TileLang fused SwiGLU FFN design (`ffn_std`)

Goal: emit a Hexagon C example for the fixed Qwen FFN anchor using only
standard TileLang constructs, mirroring `examples/hexagon/gdn_std/gdn_std.py` in style:
`T.copy`, `T.gemm`, `T.parallel`, `T.serial`, `T.vectorized`, standard scalar /
elementwise arithmetic, and `T.exp`.  The example is a code-generation and
structure anchor; it does not embed handwritten C strings and does not call
user-visible `T.hexagon` leaf intrinsics.

Fixed shape:

```text
M = 1024, K = 2560, FF = 9216
x  : [M, K]   fp16 row-major
Wg : [FF, K]  fp16 row-major
Wu : [FF, K]  fp16 row-major
Wd : [K, FF]  fp16 row-major
y  : [M, K]   fp16 row-major
```

Math:

```text
gate = x @ Wg^T       # fp32 HMX accumulation, fp16 staged storage
up   = x @ Wu^T       # fp32 HMX accumulation, fp16 staged storage
h    = silu(gate) * up = gate / (1 + exp(-gate)) * up
y    = h @ Wd^T       # fp32 HMX accumulation, fp16 output storage
```

The handwritten reference for semantics is
`/root/project/backend/npu/attn/skel/src/attnops_ffn.c`; this example does not
copy its handwritten worker code or leaf intrinsics.

## Kernel phases

The row-block sweeps use `T.Pipelined(..., num_stages=3)` with explicit stage
annotations so the Hexagon emitter can use the fused pool-phase async pattern
from `00_syntax_and_rules.md` §6.5.  In the steady-state body each loop is shaped
as `[T.parallel producer, HMX GEMM/readout chain..., T.parallel writeback]`:

- phase 1 uses `stage=[0,1,1,1,1,2]`, where stage 0 stages `X`, stage 1 runs the
  gate/up HMX chains and accumulator readouts, and stage 2 computes SwiGLU and
  stores `h`;
- phase 2 uses `stage=[0,1,1,2]`, where stage 0 stages `h`, stage 1 runs the down
  HMX chain and readout, and stage 2 stores `y`.

For each steady-state loop the two `T.parallel` phases are emitted as one fused
async worker array: jobs `[0,E0)` run the producer and jobs `[E0,E0+E1)` run the
writeback from the delayed buffer version.  The caller thread starts the fused
array with one `attnops_pool_start_ctx`, runs the HMX chain, then joins once at
the loop back-edge.  Prologue/epilogue phases stay synchronous
`attnops_pool_run_ctx`.  The weight-panel loops remain ordinary `T.serial` loops
so Wg/Wu/Wd are not double-buffered in VTCM.

### Phase 1: gate/up panels and SwiGLU scratch

The FF axis is split into static `FF_PANEL = 256` panels.  For each panel:

1. Stage `Wg[p:p+FF_PANEL, :]` and `Wu[p:p+FF_PANEL, :]` into one fused VTCM WH
   buffer `W_b[0:2*FF_PANEL, :]` with a single 16-job pool phase (jobs 0..7 for
   Wg, jobs 8..15 for Wu).  This avoids two separate 8-job phases on the 6-worker
   pool and reduces tail-worker bubbles.
2. For each 32-row block of `x`, the pipelined stage-0 producer stages the
   activation block to VTCM AH once; in steady state this staging for the next
   logical block is fused with the delayed stage-2 SwiGLU/writeback of the
   previous block in the same async job array.
3. The caller thread runs the stage-1 gate/up `T.gemm` calls and copies
   accumulator tiles to versioned VTCM fp16 temporaries while the fused pool
   array runs.  The stage-2 `T.parallel` computes `silu(gate) * up` with standard
   math (`T.exp`) and writes the result to global `h` scratch.

`h` is global scratch in row-major `[M, FF]`.  A fully raw HMX-AH global scratch
with no accumulator unpermute would require a standard TileLang global-layout
annotation that is not currently exposed; the v1 example therefore uses the
compiler's standard accumulator-to-fp16 copy and documents this as an
expressiveness gap rather than adding handwritten C.

### Phase 2: down projection

For each output-channel panel:

1. Stage `Wd[k:k+K_PANEL, :]` to WH once and keep it resident.
2. The pipelined row-block loop stages a row-major 32-row block of `h` as the HMX
   A operand.  In steady state, H staging for the next logical block is fused
   with the delayed final-store phase for the previous block.
3. The caller thread runs `T.gemm(h, Wd_panel^T)` and copies the accumulator to a
   versioned VTCM fp16 tile; the fused async pool writes the delayed tile to the
   row-major `y` region of the slab.

## VTCM budget

The panels are chosen by a simple under-8MB budget check, matching the spirit of
`gdn_std`'s self-check.

| Buffer | Shape/layout | Bytes |
|---|---:|---:|
| `X_a` | 2 × `[32, 2560]` fp16 AH (pipeline versions) | 327,680 |
| `W_b` | `[2*256, 2560]` fp16 WH (`Wg` then `Wu`) | 2,621,440 |
| `Gate`, `Up` | 2 × 2 × `[32,256]` fp16 RM (delayed writeback versions) | 65,536 |
| `H_a` | 2 × `[32, 9216]` fp16 AH (pipeline versions) | 1,179,648 |
| `Wd_b` | `[128, 9216]` fp16 WH | 2,359,296 |
| `Ytile` | 2 × `[32,128]` fp16 RM (delayed final-store versions) | 16,384 |
| HMX accumulator/readout area | emitter-managed | ~64 KiB |
| **Total static VTCM request** | | **6,606,848 bytes (~6.30 MiB) generated** |

`FF_PANEL = 512` would require two resident phase-1 weight panels of about
5.0 MiB before the down-projection panel and full-row `h` staging buffers are
counted.  Keeping all buffers statically allocated in one standard TileLang
function would be too close to / above the 8 MiB VTCM cap, so v1 uses
`FF_PANEL = 256`.

## Slab ABI

The emitted AOT entry follows the generic TileLang Hexagon slab ABI:

```c
int attnops_tl_ffn_std(remote_handle64 h,
                       unsigned char *slab, int slabLen,
                       int abl);
```

Logical contents:

```text
x_off    = 0
x_bytes  = 1024*2560*2, row-major [1024,2560]
wg_off   = align128(x_off + x_bytes)
wg_bytes = 9216*2560*2, row-major [9216,2560]
wu_off   = align128(wg_off + wg_bytes)
wu_bytes = 9216*2560*2, row-major [9216,2560]
wd_off   = align128(wu_off + wu_bytes)
wd_bytes = 2560*9216*2, row-major [2560,9216]
h_off    = align128(wd_off + wd_bytes)
h_bytes  = 1024*9216*2, row-major [1024,9216]
y_off    = align128(h_off + h_bytes)
y_bytes  = 1024*9216*2, row-major-padded [1024,9216]
           valid y is y[row, 0:2560]; columns [2560:9216) are padding
prof_off = align128(y_off + y_bytes)
prof     = int32 qtimer counters emitted by tl.hexagon_prof
```

The padded `y` region is an ABI convenience for the standard 2-D TileLang slab
signature used in this example.  A production wrapper can expose `y` as the
valid leading `K` columns.

The generated function also accepts `abl`, as in other Hexagon examples.  The
profile tail uses the same convention as `gdn_std`: slots 0..4 are aggregate
staging/HMX/writeback/wall counters and slots 5..N describe worker-pool phases in
source order.

## Performance notes

- 2026-09-14 OnePlus 13, skel with R5 staging optimization: fusing the phase-1
  Wg/Wu staging phases changed the first weight-stage pool from two 8-job phases
  (`p5=175792`, `p6=173536` ticks baseline) to one 16-job phase (`p5=281862`
  ticks).  Staging span improved from 349,328 to 281,862 ticks (19.3% lower;
  close to the expected 4-wave -> 3-wave tail-bubble reduction).  End-to-end
  `ffn_std_test 3`: 70.0 ms baseline -> 67.673 ms.

## v1 exclusions

- No weight-panel pipelining; double-buffering the large Wg/Wu/Wd panels would
  exceed the 8 MiB VTCM cap.
- No host-side WH preconversion ABI; weights are staged from row-major tensors by
  standard `T.copy`.
- No handwritten raw-AH global writeback for `h`.  The standard route stores
  tile-major logical fp16 scratch and re-stages it for phase 2.
- No hardware execution or numerical validation is performed by this example;
  `--self-check` is structural and `hexagon-clang -fsyntax-only` only.
