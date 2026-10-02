# Worker scheduling modes

Worker-role `T.Pipelined` now defaults to `round_sync`. The explicit spelling is
`annotations={"tl.workergroup_schedule": "round_sync"}`. The previous independent
ready/free protocol remains available as experimental `"async"`; example CLI
uses `--schedule async`. Non-worker CUDA pipelines are unchanged.

Offsets are longest-path distances in the validated forward role DAG. At tick t,
role r executes logical iteration t-offset[r] when in bounds. Slot addresses
use that logical iteration plus the outer invocation ordinal, not the tick.
All physical workers, including inactive prologue/drain lanes, reach one team
barrier outside the role branches. Each outer tile drains before the next clear.
For each edge, depth must exceed consumer_offset-producer_offset: otherwise
producer and consumer could access the same slot in the same tick. Depth 1 is
therefore rejected for a nonempty dependency edge, not silently serialized.

Runtime coordination: canonical ABI adds only
`int tl_wg_team_barrier(const tl_wg_worker *);` (no descriptor size changes).
This is a whole-launch collective, NOT a group barrier. Runtime must implement
uniform cancellation/deadline handling and wake all participants on failure.
The compiler emits error propagation at the round boundary, outside the HMX
issue/readout body. The runtime implementation and device behavior have not
been verified in this compiler change. Software deadlines cannot recover a
hardware hang or prove outstanding HMX work quiescent.

TEST-CONTRACT-CHANGES: previous depth-1 and independent-event tests now explicitly
select async, preserving their original coverage. New tests check round codegen
and reject depth-1 slot collisions. No numerical threshold changes.
