# Experimental consumer read staging

Enable through PrimFunc attributes `tl.workergroup_stage_reads=1` and an explicit
`tl.workergroup_max_ddr_bytes` budget. Default is off. The production example
exposes `--stage-reads --cap-ddr N`.

`StageWorkerReads` runs after the existing numerical chain and final physical
allocation planner. It snapshots read-only, flattened, static versioned shared
buffers inside HVX ordinal regions. It excludes writes, address escapes (including
address_of BufferLoad), and multiple Buffer views sharing a data pointer. Full
physical backing must be contiguous, zero-offset and a multiple of 128 bytes.
Unsupported regions retain the original safe access path. This is structural,
not a match on GEMM names, roles, shapes, or AH index expressions.

Only BufferLoad nodes are redirected. Consumer indices, predicates, casts and
addition order are unchanged. HMX output, packing targets and scatter-release
remain VTCM operations. The emitted primitive is a 128-byte HVX copy loop, not a
whole-operator helper. The accumulator update is still scalar in the inspected
medium assembly: do NOT interpret staging vmem or pack vadd as an HVX add.

Each consuming worker has a distinct 128-byte-aligned DDR slice, refreshed within
its ordinal scope before use. Slices are monotonic and not reused between stage
sites. Producer event/round lifetime still spans this copy and its consumer.
The descriptor reports total DDR bytes; the unchanged owner API forwards the
runtime DDR backing through an added private callback argument. The owner rejects
null/misaligned DDR and DDR/VTCM overlap. Runtime must provide validated backing
and caps; an old DDR=0 runner is NOT compatible when enabled.

Default small example requires 4096 DDR bytes; M128 N512 K2560 with
BM64 BN256 BK1280 requires 65536 DDR bytes (32768 per consumer, two consumers).
The persistent accumulator remains local. Neither this lowering nor its host
layout bit-copy oracle proves device performance or end-to-end precision; that
requires the isolated device A/B and full independent reference checks.

Diagnostic basis: the supplied fixed-stage patch changed only consumer reads;
its assembly has a vmem copy followed by memuh and scalar sfadd. Reported device
speedup belongs to that isolated patch, not yet to regenerated compiler output.
