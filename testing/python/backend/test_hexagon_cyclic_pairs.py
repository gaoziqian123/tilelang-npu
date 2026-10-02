"""Cyclic paired-row ownership and divergent-barrier regression checks."""
import ast
from collections import Counter
from pathlib import Path
import pytest


@pytest.mark.parametrize('workers', range(1, 7))
@pytest.mark.parametrize('bm', [2, 6, 32, 64, 96, 128, 192, 256, 1024])
def test_exact_cover(workers, bm):
    visits=[]
    for tx in range(workers):
        # The serial upper bound is the predicate p < BM/2 expressed as
        # an iteration domain; no barrier is inside this divergent domain.
        extent=max(0,(bm//2-tx+workers-1)//workers)
        actual=[(it*workers+tx)*2+r for it in range(extent) for r in range(2)]
        predicated=[(it*workers+tx)*2+r
                    for it in range((bm//2+workers-1)//workers)
                    if it*workers+tx<bm//2 for r in range(2)]
        assert actual==predicated
        assert all(0<=row<bm for row in actual)
        visits.extend(actual)
    assert Counter(visits)==Counter(range(bm))


def reject_divergent_barriers(source):
    """DSL structural audit, not a replacement for compiler verification.

    Track tx dependencies through enclosing loop bounds and predicates.
    A later compiler test must also cover automatically inserted barriers.
    """
    def dependent(node):
        return any(isinstance(n,ast.Name) and n.id=='tx' for n in ast.walk(node))
    def walk(node, divergent=False):
        if isinstance(node,ast.If):
            divergent |= dependent(node.test)
        if isinstance(node,ast.For):
            divergent |= dependent(node.iter)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            if node.func.attr=='sync_threads' and divergent:
                raise ValueError('barrier in worker-dependent control flow')
        for child in ast.iter_child_nodes(node):
            walk(child,divergent)
    walk(ast.parse(source))


@pytest.mark.parametrize('source', [
    'if tx < 4:\n T.sync_threads()',
    'for p in T.serial(T.ceildiv(64-tx,workers)):\n T.sync_threads()',
    'for p in T.serial(11):\n if p*workers+tx<64:\n  T.sync_threads()',
])
def test_divergent_barrier_negative(source):
    with pytest.raises(ValueError,match='worker-dependent'):
        reject_divergent_barriers(source)


def test_persistent_barriers_outside_tail():
    root=Path(__file__).resolve().parents[3]
    source=(root/'examples/hexagon/fa/fa_persistent.py').read_text()
    reject_divergent_barriers(source)
    assert 'rp*8+tx*2' not in source
    assert 'workers=4' in source
    assert 'pairs_per_worker*2' in source


def test_four_worker_old_mapping_identical():
    for rp in range(16):
        for tx in range(4):
            for r in range(2):
                assert (rp*4+tx)*2+r==rp*8+tx*2+r
                assert ((rp*4+tx)*2+r)%4==(tx*2+r)%4
