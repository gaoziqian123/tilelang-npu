# Explicit vector batch API (not a general TIR vectorizer)

Import from `tilelang.hexagon.language.vector`.

* `load(buffer, offset=0, valid_lanes=None, fill_bits=0)` / `store(buffer,
  value, offset=0, valid_lanes=None)`: compact rank-one FP16/FP32 or 16/32-bit
  integer buffers, static extent and offset, zero buffer elem_offset. Offset
  must be a multiple of 128B. Full 128B storage must exist even for logical
  tails; unpadded tails reject. Runtime base-address alignment traps before
  access. Memory codegen independently rechecks bounds of address_of operands.
  Predicated store uses Q6_vmem_QRIV; it does not read/modify neighbouring bytes.
* `splat_bits(bits, dtype)` / `extract_bits(value, lane)`: scalar bit-pattern
  APIs, not implicit numerical conversions. Extract returns uint32; half uses
  its low 16 bits. Static lane is checked. `bitcast(value,dtype)` is 128B only.
* `leaf('fma16', a,b,c, mode='native_target_v1')`: Q6 half multiply-accumulate.
  Strict single-rounding mode is NOT claimed. FP32 FMA and unknown modes reject.
* `compare_chunks(a,b,relation=eq/ne/lt/le/gt/ge,nan_policy=ordered/unordered)`:
  bitwise IEEE ordering, subnormals and signed zero handled explicitly, byte
  masks returned. Unordered means true if either input is NaN, including eq.
* `load_chunks(buffer,lanes,offset=0,fill_bits=0)` splits a static logical
  region into typed 128B values. `map_chunks(name,*values,mode,immediate=0)`
  lowers each physical chunk through the closed leaf registry. `widen`
  doubles FP32 chunk count, preserving half rounding already performed.
  `narrow_chunks(values,mode=...,order='linear'/'evenodd')` consumes FP32 pairs.
  `store_chunks(buffer,values,lanes,offset=0)` emits a SeqStmt of stores.

This is explicit frontend chunk lowering, **not** a pass translating arbitrary
oversized vector TIR expressions. That requested pass is still missing.

`tl.vector_required=1` on a function invokes the final pipeline verifier. It
rejects generic BufferLoad/Store, unknown calls and generic vector/float
arithmetic, permitting address_of and explicit leaves. This conservative gate
is not a complete scalar integer data-flow verifier: it cannot distinguish an
integer extracted value used for data arithmetic from an integer address.
Do not claim full fail-closed data-taint validation until that gap is closed.

Tests cover new memory bounds rejection, explicit chunk cast construction,
scalar-load rejection, and real v79 -O2 object/ASM for masked memory, comparisons
and half multiply-accumulate. Generated-C end-to-end and device numerical
verification remain outstanding. No existing tests or thresholds changed.
