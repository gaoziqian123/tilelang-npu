# Standard FA v4 half-state candidate

Build with `.venv/bin/python examples/hexagon/fa/build_worker_standard.py
--output /tmp/opencode/<fresh-directory>` from tilelang. This generates all
eight query-block functions, typed dispatcher, Layout-derived bit packers,
manifest, v79 O2 objects/assembly/stack reports and a relocatable combined object.
Completed candidate: `/tmp/opencode/fa_fp16_standard/r2`. The parent directory
contains a FAILED earlier generation; do not use those files.

## TEST-CONTRACT-CHANGES (explicit user authorization)

The user changed FA precision requirements to half state and cosine >=0.999.
All persistent online m/l/alpha/out and score/P/PV storage is now half; score
scale is applied after HMX readout with a half rounding boundary. Independent
FP64 reference, input hashes, output NaN poisoning and nonfinite rejection remain
mandatory. Maximum relative error is diagnostic, NOT the old <0.1 gate. No
repository test or tolerance was modified, and this package has no numerical
pass claim. Runtime test/runner enforcement is the backend owner's task.

Not all instructions are half: panel-local exp widens to the existing FP32 HVX
math implementation then narrows; two-row max/sum scratch uses the existing
FP32 row-reduction leaf then rounds to half. This temporary reduction is not
cross-panel state. Row-state arrays are small private local memory; scalar
row broadcasts/reduction result stores therefore do not access VTCM. Generic
division may use wider intermediate arithmetic. HMX internal accumulation is
hardware-defined. This is a half-storage/state contract, not all-HF arithmetic.

## Backend interface

Use generated `dispatcher.h`, `manifest.json`, `pack.h`, `fa_standard.o`.
The full call handles B1/HQ16/HKV4/S1024/D256, BQ128/BK256, with 128 BLOCKING
local owner launches inside one RPC; keep the caller on the HMX-open thread.
There is no RPC per query block and no arithmetic in the dispatcher.

Q: 128 independently packed 128x256 blocks in head/query-block order, 8MiB.
K: four independently packed 1024x256 heads, 2MiB.
V: four 1024x256 logical heads composed with transpose into the kernel's
WH 256x1024 view, 2MiB. Packers preserve raw half bits, no arithmetic.
O: row-major [16,1024,256] half, 8MiB, write-only DMA destination.
Register Q/K/V read-only and O write-only; alignment is 128 bytes. All four
spans must be disjoint. Runtime adapter must also enforce disjointness from
scratch, registered bounds and permissions. Sync bytes are runtime-calculated.

Every case has 3 workers, 5 stages, 24 events, depth two and VTCM 1,311,232
bytes (plus reserved 768 <4MiB). Cases use 1,1,2,2,3,3,4,4 visible panels,
with the exact row mask inside the last panel. Q is redundantly DMA-loaded
each panel and K/V are not retained between owner calls. These are explicit
optimization debts; no 10ms or sufficient HVX parallelism claim.
