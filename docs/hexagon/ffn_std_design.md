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

The v1 schedule is deliberately synchronous: worker-pool `T.parallel` phases
are barriers, and all `T.gemm` HMX chains run on the caller thread.  There is no
async overlap.

### Phase 1: gate/up panels and SwiGLU scratch

The FF axis is split into static `FF_PANEL = 256` panels.  For each panel:

1. Stage `Wg[p:p+FF_PANEL, :]` and `Wu[p:p+FF_PANEL, :]` to VTCM WH layout with
   `T.copy`.
2. For each 32-row block of `x`, stage the activation block to VTCM AH once and
   run two `T.gemm` calls sharing it: gate and up.
3. Copy accumulator tiles to VTCM fp16 temporaries, then a `T.parallel` phase
   computes `silu(gate) * up` elementwise using standard math (`T.exp`) and
   writes the result to global `h` scratch.

`h` is global scratch in row-major `[M, FF]`.  A fully raw HMX-AH global scratch
with no accumulator unpermute would require a standard TileLang global-layout
annotation that is not currently exposed; the v1 example therefore uses the
compiler's standard accumulator-to-fp16 copy and documents this as an
expressiveness gap rather than adding handwritten C.

### Phase 2: down projection

For each 32-row block:

1. `T.copy(..., layout=("rm", "ah"))` stages a row-major 32-row block of `h` as
   the HMX A operand.
2. The output K axis is split into `K_PANEL = 128` panels.  For each panel,
   stage `Wd[k:k+K_PANEL, :]` to WH and run `T.gemm(h, Wd_panel^T)`.
3. Copy the accumulator to a VTCM fp16 tile and a worker-pool phase writes it to
   the row-major `y` region of the slab.

## VTCM budget

The panels are chosen by a simple under-8MB budget check, matching the spirit of
`gdn_std`'s self-check.

| Buffer | Shape/layout | Bytes |
|---|---:|---:|
| `X_a` | `[32, 2560]` fp16 AH | 163,840 |
| `Wg_b` | `[256, 2560]` fp16 WH | 1,310,720 |
| `Wu_b` | `[256, 2560]` fp16 WH | 1,310,720 |
| gate/up/h panel tmp | 3 × `[32,256]` fp16 RM | 49,152 |
| `H_a` | `[32, 9216]` fp16 AH | 589,824 |
| `Wd_b` | `[128, 9216]` fp16 WH | 2,359,296 |
| HMX accumulator/readout area | emitter-managed | ~64 KiB |
| **Total static VTCM request** | | **< 6.5 MiB generated** |

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

- No async pool/HMX overlap; every `T.parallel` phase is synchronous.
- No host-side WH preconversion ABI; weights are staged from row-major tensors by
  standard `T.copy`.
- No handwritten raw-AH global writeback for `h`.  The standard route stores
  tile-major logical fp16 scratch and re-stages it for phase 2.
- No hardware execution or numerical validation is performed by this example;
  `--self-check` is structural and `hexagon-clang -fsyntax-only` only.
