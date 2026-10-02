"""Explicit worker-group roles; scheduling is performed by the C++ compiler."""

from tilelang import _ffi_api


def pipeline_stage(name: str, *, engine: str, workers: int | None = None,
                   weight: int | None = None, physical_owner: str | None = None):
    """Annotate a role inside T.Pipelined, without deriving semantics from names.

    ``workers`` is a positive integer count. Alternatively an HVX role may
    request a positive integer ``weight``; see docs/hexagon/workergroup_abi.md.
    HMX must explicitly request one worker. Counts and weights are exclusive.
    This is an IR-builder scope, not a Python scheduler or runtime context.
    ``physical_owner`` explicitly aliases equally sized logical stages onto one
    physical group. Their dependency edges remain logical-stage edges. Aliased
    plans use stage-aware ABI v4, async depth-two rectangular iteration contracts.
    """
    if not isinstance(name, str) or not isinstance(engine, str):
        raise TypeError("name and engine must be strings")
    if (workers is None) == (weight is None):
        raise ValueError("specify exactly one of workers or weight")
    for value in (workers, weight):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("workers and weight must be positive integers")
    if physical_owner is not None and (not isinstance(physical_owner, str) or not physical_owner):
        raise ValueError("physical_owner must be a nonempty string")
    if physical_owner is None:
        return _ffi_api.PipelineStage(name, engine, workers or 0, weight or 0)
    return _ffi_api.PipelineStageWithOwner(name, engine, workers or 0, weight or 0, physical_owner)
