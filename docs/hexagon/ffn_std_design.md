# Standard TileLang fused SwiGLU FFN design (`ffn_std`)

Goal: emit a Hexagon C example for the fixed Qwen FFN anchor using only
standard TileLang constructs, mirroring `examples/hexagon/gdn_std.py` in style:
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

The row-block sweeps use `T.Pipelined(..., num_stages=2)` to overlap activation
staging with caller-thread HMX work.  In the steady-state loop the first
worker-pool `T.parallel` phase is emitted as `attnops_pool_start_ctx`, while the
caller thread continues into the HMX GEMM / readout chain.  The outstanding pool
is joined before the next worker-pool phase (the SwiGLU/store or final store),
and prologue/epilogue phases stay synchronous.  The weight-panel loops remain
ordinary `T.serial` loops so Wg/Wu/Wd are not double-buffered in VTCM.

### Phase 1: gate/up panels and SwiGLU scratch

The FF axis is split into static `FF_PANEL = 256` panels.  For each panel:

1. Stage `Wg[p:p+FF_PANEL, :]` and `Wu[p:p+FF_PANEL, :]` to VTCM WH layout with
   `T.copy`.
2. For each 32-row block of `x`, the pipelined stage-0 `T.parallel` phase stages
   the activation block to VTCM AH once; in steady state this staging for block
   `mb+1` overlaps the stage-1 gate/up `T.gemm` calls for block `mb`.
3. Copy accumulator tiles to VTCM fp16 temporaries, join the outstanding staging
   pool, then a synchronous `T.parallel` phase computes `silu(gate) * up`
   elementwise using standard math (`T.exp`) and writes the result to global `h`
   scratch.

`h` is global scratch in row-major `[M, FF]`.  A fully raw HMX-AH global scratch
with no accumulator unpermute would require a standard TileLang global-layout
annotation that is not currently exposed; the v1 example therefore uses the
compiler's standard accumulator-to-fp16 copy and documents this as an
expressiveness gap rather than adding handwritten C.

### Phase 2: down projection

For each output-channel panel:

1. Stage `Wd[k:k+K_PANEL, :]` to WH once and keep it resident.
2. The pipelined row-block loop stages a row-major 32-row block of `h` as the HMX
   A operand.  In steady state, H staging for `rb2+1` overlaps the down-projection
   `T.gemm(h, Wd_panel^T)` for `rb2`.
3. Copy the accumulator to a VTCM fp16 tile, join the outstanding staging pool,
   and a synchronous worker-pool phase writes it to the row-major `y` region of
   the slab.

## VTCM budget

The panels are chosen by a simple under-8MB budget check, matching the spirit of
`gdn_std`'s self-check.

| Buffer | Shape/layout | Bytes |
|---|---:|---:|
| `X_a` | 2 × `[32, 2560]` fp16 AH (pipeline versions) | 327,680 |
| `Wg_b` | `[256, 2560]` fp16 WH | 1,310,720 |
| `Wu_b` | `[256, 2560]` fp16 WH | 1,310,720 |
| gate/up/h panel tmp | 3 × `[32,256]` fp16 RM | 49,152 |
| `H_a` | 2 × `[32, 9216]` fp16 AH (pipeline versions) | 1,179,648 |
| `Wd_b` | `[128, 9216]` fp16 WH | 2,359,296 |
| HMX accumulator/readout area | emitter-managed | ~64 KiB |
| **Total static VTCM request** | | **< 7.2 MiB generated** |

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

## v1 exclusions

- No weight-panel pipelining; double-buffering the large Wg/Wu/Wd panels would
  exceed the 8 MiB VTCM cap.
- No host-side WH preconversion ABI; weights are staged from row-major tensors by
  standard `T.copy`.
- No handwritten raw-AH global writeback for `h`.  The standard route stores
  tile-major logical fp16 scratch and re-stages it for phase 2.
- No hardware execution or numerical validation is performed by this example;
  `--self-check` is structural and `hexagon-clang -fsyntax-only` only.
