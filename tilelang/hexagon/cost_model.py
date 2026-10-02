"""Initial launch estimates, intended to be calibrated by an external tuner.

Costs are bootstrap priors, not performance guarantees. Shape information uses
``jobs`` (independent jobs) and ``work_per_job`` (normalized work units).
An external orchestrator can sweep PassContext overrides without code changes.
"""

from dataclasses import dataclass
from math import ceil, isfinite
from types import MappingProxyType

from tvm import tirx
from tilelang.transform import PassConfigKey, get_pass_context


@dataclass(frozen=True)
class OpCost:
    max_workers: int
    unit_cost: float
    worker_overhead: float


# Relative cost units, deliberately not labelled as measured milliseconds.
# HVX pool capacity is six; HMX issue must remain on its owner thread.
OP_COSTS = MappingProxyType({
    "elementwise": OpCost(6, 1.0, 0.05),
    "copy": OpCost(6, 1.0, 0.05),
    "gemm": OpCost(1, 1.0, 0.0),
})


@dataclass(frozen=True)
class LaunchConfig:
    num_workers: int
    job_partition: int
    estimated_cost: float


def estimate_launch(op_kind, shape_info, target) -> LaunchConfig:
    """Estimate a wave launch; explicit tuner overrides use PassContext.

    ``tl.hexagon.num_workers`` overrides worker count and
    ``tl.hexagon.job_partition`` overrides contiguous jobs per worker.
    Both must be positive integers. Unknown operations fail closed.
    """
    from .target import target_is_hexagon
    if not target_is_hexagon(target):
        raise ValueError("launch estimation requires the new Hexagon target")
    cost = OP_COSTS[op_kind]
    jobs = int(shape_info.get("jobs", 1))
    work = float(shape_info.get("work_per_job", 1.0))
    if jobs < 1 or work <= 0 or not isfinite(work):
        raise ValueError("jobs and work_per_job must be positive and finite")
    def score(workers):
        return ceil(jobs / workers) * work * cost.unit_cost + workers * cost.worker_overhead

    workers = min(range(1, min(jobs, cost.max_workers) + 1), key=score)
    config = get_pass_context().config
    workers = int(config.get(PassConfigKey.TL_HEXAGON_NUM_WORKERS.value, workers))
    if workers < 1:
        raise ValueError("tl.hexagon.num_workers must be positive")
    partition = int(config.get(PassConfigKey.TL_HEXAGON_JOB_PARTITION.value, ceil(jobs / workers)))
    if partition < 1:
        raise ValueError("tl.hexagon.job_partition must be positive")
    return LaunchConfig(workers, partition, score(workers))


def AnnotateLaunchEstimates():
    """Attach estimates only when thread bindings do not already exist.

    TODO: T.Kernel currently erases whether threads=128 was explicit. Until the
    common frontend preserves that provenance, all thread bindings are treated
    as explicit. Never silently replace an explicit user launch. These attrs
    are planning metadata, not a worker-pool launch implementation.
    """
    @tirx.transform.prim_func_pass(opt_level=0)
    def annotate(func, mod, ctx):
        bindings = []
        def visit(node):
            if isinstance(node, tirx.For) and node.thread_binding is not None:
                if node.thread_binding.thread_tag.startswith("threadIdx."):
                    bindings.append(node)
            if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent":
                if getattr(node.node, "thread_tag", "").startswith("threadIdx."):
                    bindings.append(node)
        tirx.stmt_functor.post_order_visit(func.body, visit)
        if bindings:
            return func
        attrs = func.attrs
        launch = estimate_launch(
            str(attrs.get("hexagon.op_kind", "elementwise")),
            {"jobs": attrs.get("hexagon.jobs", 1), "work_per_job": attrs.get("hexagon.work_per_job", 1)},
            attrs["target"],
        )
        return func.with_attrs({
            "hexagon.num_workers": launch.num_workers,
            "hexagon.job_partition": launch.job_partition,
        })
    return annotate
