# Explicit prepacked external ABI (experimental)

## TEST-CONTRACT-CHANGES: FP16 compute/output requirement

The user explicitly replaced the previous FP32 cross-K accumulator/output
contract. `gemm_prepacked_dma.py` now uses FP16 inputs, HMX partial, persistent
HVX accumulator and row-major output. HMX internal accumulator precision is
hardware-defined; this is not a claim that its internal arithmetic is FP16.
The two-K partial schedule and separate consumer stage remain intact. The
fragment explicitly follows contiguous physical producer chunks; clear, update
and final copy share that map through ordinary layout inference/vectorization.
The prior `gemm_worker_pipeline.py` FP32 example is unchanged.

Runner must use `__fp16 *C`, allocate/poison **2*M*N bytes**, and decode FP16
before its independent FP64 reference comparison. Keep rms max-relative <0.1,
cosine >=0.999, finite/NaN scans and input guards unchanged. FP16 rounding may
fail those criteria; do not adjust tolerances to hide it. Device precision has
not been tested in this compiler change.

Fresh artifacts: `/tmp/opencode/prepacked_fp16/{tiny,medium,standard}` with
.c/.o/.s/.json and _pack.h. Output sizes: tiny4096, medium131072,
standard25165824 bytes. Medium/standard accumulator storage is 8192 half
elements =16384 bytes per worker (excluding compiler spills/frame overhead).
O2 assembly shows hf operands into qf16 vadd followed by hf conversion, not
FP32 accumulation. Final logical output copy is currently scalar half stores;
no vector-output-store performance claim is made.

Use public `T.Layout` and `T.annotate_layout` on both external tensors and local
shared buffers. `T.copy(..., annotations={"hexagon.dma_direct": 1})` requests a
physical copy, and rejects an unproved mapping rather than repacking silently.
Without that request the existing conversion path is retained. An annotated
external pointer MUST already point to packed bytes; annotations do not pack
row-major host input automatically.

`tilelang.hexagon.physical_contract.contract(layout)` exports version, dtype,
logical shape, physical extent/strides, alignment, zero-padding policy, byte
capacity, exact element map and SHA256. `pack`/`unpack` are bit-copy CPU reference
implementations. Allocation alignment is the caller's obligation; a NumPy byte
array returned by pack is not guaranteed to be DMA-aligned. Copy to an aligned
runtime allocation. Record cold host pack + RPC separately from warm RPC timing.

`src/hexagon/op/copy.cc:DirectPhysicalCopy` proves bounded panel mappings from the
two Layout expressions, sorts physical offsets, coalesces contiguous spans and
then regular 2D spans. It does not compare layout names or assume a panel is one
contiguous span. Static proof is currently limited to 1048576 panel elements;
the Python reference exporter has a separate explicit capacity limit. This
enumeration implementation is a first version, not scalable symbolic lowering.

Completion: the group leader calls a synchronous copy-and-wait adapter, then
every group lane enters its group barrier. No cross-stage DMA overlap is claimed.
HMX and accumulator semantics remain unchanged. The low-level adapter must be
provided by the backend; object compilation is NOT proof of device linkage.

## Backend coordination (not implemented here)

Header: `src/tl_templates/hexagon/dma_copy.h`.

```c
int tl_hex_dma_copy_2d_wait(void *dst, const void *src,
    uint32_t width, uint32_t rows, uint32_t src_stride, uint32_t dst_stride);
```

Return zero only after destination visibility/completion; negative on failure.
The compiler emits an opaque statement intrinsic with checked status: a nonzero
adapter result returns immediately from the int worker callback. Runtime execute
error handling must cancel/wake the other group lanes. Noncallback compilation
is rejected (no implicit trap or discarded result). All addresses, widths and strides are conservatively 128-byte
aligned. Backend must validate/split descriptors against **v79 SDK** limits;
these uint32 fields are a software interface, not a hardware register format.
Do not implement by importing an unavailable `qurt_user_dma_dmsyncht` symbol.
No claim is made here that this new adapter is already linked or device usable.

Example: `examples/hexagon/gemm/gemm_prepacked_dma.py --output path.c --host-packer-output pack.h` generates
the C and adjacent JSON manifest, including input maps and required symbol.
The compact v2 input contract stores a C index expression, not per-element
indices. `compact_contract` uses Layout inverse and range proofs, admitting only
nonnegative bounded arithmetic whose C translation has the same semantics.
The generated C packer accepts `(const void *logical, void *packed,
size_t logical_cap, size_t packed_cap)` and returns int; it validates byte
capacities, alignment, range overflow and aliasing, zeros padding, and copies
bits with memcpy. There is no hidden cache. Callers must repack changed inputs;
weight cache keys must include input version/digest and layout fingerprint.

Compiler and wrapper check v79 width/strides <=0xFFFFFF and rows<=65535.
The adapter still validates registered resources. The intrinsic name is
`tl.hexagon.dma_copy_2d_wait`; it lowers only in an int worker callback.

Fresh tiny/medium/standard artifacts are in `/tmp/opencode/wg_round_v3/` with
stems `prepacked`, `prepacked_medium`, `prepacked_standard` (.c/.o/.json and
_pack.h). Standard M1024 N12288 K2560, BM64 BN256 BK1280 uses A width81920,
rows2, src_stride163840,dst_stride81920; B uses identical width/strides and
rows8. VTCM1704192, DDR scratch0, round_count3072. Host packed A5242880 bytes,
B62914560 bytes, both aligned128. Link the backend-provided adapter implementation
into the skel; object compilation here does not establish device linkage.
