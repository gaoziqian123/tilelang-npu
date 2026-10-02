# Logical transform: partial implementation, not release-ready

`T.transform(src, dst)` is registered as `tl.tileop.transform`. It uses the
Copy operator's access regions (source read/destination write), not Python
code generation or a kernel-name match. Equal explicit logical shapes are
required; unlike copy's convenience normalization, broadcasting is rejected.

## Current Hexagon support

- FP16 to FP16, explicit rank-two annotated layouts, compact storage.
- Source: shared, single-worker local, or nonreplicated fragment with proven
  per-vector writer ownership. Destination: injective shared layout with
  physically contiguous 128-byte output vectors.
- Static regions, at most 65536 elements; columns divisible by 64.
- Identity vectors or even/odd half deinterleave from two source vectors.
  The latter emits native `Q6_Vh_vdeal_Vh` followed by vector concatenation.
- Distinct storage is copied, never elided. Aliased storage is rejected.
- Unsupported permutations/dtypes fail closed on Hexagon, not scalar fallback.
- No float conversion is inserted in the half-only path. Permutations preserve
  half bits, including signed zero, subnormals and NaN payloads.

NOT implemented: dtype-only/fused dtype-layout conversion, general interleave,
dynamic mappings, overlapping/in-place transforms, compact loop synthesis
(the current proof emits one statement per output vector). No claim of an
all-layout API or complete requested support matrix is made. Other targets
currently share Copy lowering; their transform semantics are not verified.

## Experimental example and DMA contract

`gemm_prepacked_dma.py --transform-output` opts into a shared output stage;
default baseline remains unchanged. Standard configuration allocates 32768
bytes for this stage. Generated v3 plan total VTCM is 1736960 bytes, with
output stage at 1703936 and the HMX scale storage following it. This resource
information is not yet exported in the compact manifest.

Direct DMA now accepts either global-to-shared or shared-to-global with the
same physical-span proof. Shared-to-global inserts a group publication barrier,
one leader submit+wait, then a group completion barrier before reuse. Errors
return from the callback. No backend or qurt implementation was changed.

Backend handoff (must be implemented before running the candidate):

- Register C as a permitted DDR **destination**, with its exact capacity, and
  register VTCM as a permitted source as well as destination. Do not relax to
  arbitrary addresses or merely test the starting pointer.
- For each direction validate the complete span `(rows-1)*stride+width` in both
  registered allocations with overflow-safe arithmetic, read/write permission,
  and 128-byte address/width/stride alignment.
- Enforce width and both strides <= 0xFFFFFF, rows <= 65535; check byte offsets
  and pointer arithmetic before conversion to the native uint32 descriptor.
- Standard output descriptor: width=512, rows=64, source stride=512,
  destination stride=24576, destination byte offset=m*1572864+n*512.
- Input and output DMA owners belong to different groups but execute in
  nonoverlapping phases under the existing round drain; no asynchronous event
  scheduling change or multiple simultaneous submitters is intended.

Artifacts under `/tmp/opencode/transformstd/` are experimental. B reuse is NOT
merged: the original two-K producer/slot schedule remains unchanged. Lifetime
validation was not bypassed. Independent FP64 device comparison, original
precision thresholds, poisoning and input guards remain mandatory.

Host tests check API shape rejection, access-region construction, all 65536
half bit patterns against an independent lane permutation, and worker codegen.
They do not execute the DSP instructions. No phone run or independent verifier
has been performed for this candidate.
