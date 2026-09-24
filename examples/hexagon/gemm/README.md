# Hexagon-v2 GEMM example

`gemm_nt_v2.py` is the Phase-1 vertical-slice GEMM entry.  It keeps a standard
`T.gemm` TIR builder (`get_tir`) and emits deterministic DSP C through the new
`tilelang.hexagon_v2` codegen bridge.

Key properties:

- new path only: does not modify `tilelang/hexagon/` or `examples/hexagon-legacy/`;
- HMX core is explicit deep-chain asm (`mxclracc.hf`, `mxmem:deep`, `mxmem2`,
  `:after.hf`), split into `<=32` K-tile segments;
- staging/writeback reuse the existing `hexagon_rt.h` support helpers for Phase 1.

Example:

```bash
PYTHONPATH=/root/project/tilelang \
python examples/hexagon/gemm/gemm_nt_v2.py --m 1024 --n 12288 --k 2560
```
