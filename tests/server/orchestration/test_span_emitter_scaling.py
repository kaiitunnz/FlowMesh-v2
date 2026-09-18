"""Span synthesis must not add serialization to a ledger transition.

Every emitter entry point runs under the runtime lock, so its cost is paid by the
workflow it observes. These drive a wide spawn region -- the shape where a per-emit
ledger rescan compounds -- and bound the ledger records the emitter reads rather than
the seconds it takes, because the defect is algorithmic and a wall-clock bound on a
shared machine measures the machine.
"""

from typing import Any

from server.orchestration import OrchestrationEngine
from server.orchestration.state import BoundaryEvent
from server.orchestration.telemetry import TelemetrySpanEmitter
from server.task.v2.representations.operators import BoundaryEventKind
from shared.telemetry.config import TelemetryLevel
from tests.server.orchestration.helpers import (
    emitter,
    engine,
    recursive_agent_bundle,
    rehydrated,
    span_signature,
    spawning_agent_bundle,
)

_CHILDREN = 750

# Doubling the width of a region doubles the work a linear emitter does and quadruples
# a scanning one, so the ratio separates them whatever the absolute counts are -- and
# it stays honest if someone adds another collection the emitter walks.
_MAX_GROWTH_FACTOR = 3.0


class _Reads:
    """Shared tally of ledger records read, across every collection the emitter walks.

    The cost this guards is algorithmic, so it is asserted as work done rather than as
    elapsed time: a wall-clock bound on a shared machine measures the machine. It must
    count every attached collection, because the quadratic had two halves -- rescans of
    the event trace, and scans of the work-item and activation dicts.
    """

    def __init__(self) -> None:
        self.count = 0


class _CountingTrace(list):
    def __init__(self, reads: _Reads, *args: Any) -> None:
        super().__init__(*args)
        self._reads = reads

    def __iter__(self) -> Any:
        self._reads.count += len(self)
        return super().__iter__()

    def __reversed__(self) -> Any:
        self._reads.count += len(self)
        return super().__reversed__()

    def __getitem__(self, index: Any) -> Any:
        item = super().__getitem__(index)
        self._reads.count += len(item) if isinstance(index, slice) else 1
        return item


class _CountingDict(dict):
    def __init__(self, reads: _Reads, *args: Any) -> None:
        super().__init__(*args)
        self._reads = reads

    def __iter__(self) -> Any:
        self._reads.count += len(self)
        return super().__iter__()

    def values(self) -> Any:
        self._reads.count += len(self)
        return super().values()

    def items(self) -> Any:
        self._reads.count += len(self)
        return super().items()

    def __getitem__(self, key: Any) -> Any:
        self._reads.count += 1
        return super().__getitem__(key)


def _spawn_children(eng: OrchestrationEngine, count: int) -> list[str]:
    eng.on_dispatched("A", "w1")
    for i in range(count):
        eng.route_boundary_event(
            "A",
            BoundaryEvent(
                kind=BoundaryEventKind.SPAWN,
                call_correlation=f"s{i}",
                child_region_ref="worker",
            ),
        )
    return [a.activation_id for a in eng._activations.values() if a.kind == "child"]


def _seal(eng: OrchestrationEngine) -> None:
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="seal",
            child_region_ref="worker",
        ),
    )


def _drive(level: TelemetryLevel, children_count: int) -> tuple[int, int, int]:
    """Records read during the settle loop and during the seal, plus spans emitted."""
    span_emitter, exporter = emitter(level)
    eng = engine(spawning_agent_bundle(), emitter=span_emitter)
    reads = _Reads()
    # The emitter reads the collections the engine handed it at attach, so the counting
    # ones have to replace those before anything binds to them.
    eng._trace = _CountingTrace(reads, eng._trace)
    eng._work_items = _CountingDict(reads, eng._work_items)
    eng._activations = _CountingDict(reads, eng._activations)
    eng._scopes = _CountingDict(reads, eng._scopes)
    span_emitter.attach(
        activations=eng._activations,
        scopes=eng._scopes,
        work_items=eng._work_items,
        attempts=eng._attempts,
        invocations=eng._invocations,
        trace=eng._trace,
        released_scopes=eng._released_scopes,
    )
    children = _spawn_children(eng, children_count)
    assert len(children) == children_count

    before = reads.count
    for child in children:
        eng.on_dispatched(child, "w1")
        eng.on_succeeded(child)
    settle = reads.count - before

    before = reads.count
    _seal(eng)
    seal = reads.count - before
    return settle, seal, len(exporter.get_finished_spans())


def _traced_reads(children_count: int) -> tuple[int, int]:
    """The settle and seal reads telemetry adds, over the same run with it off."""
    quiet_settle, quiet_seal, quiet_spans = _drive(TelemetryLevel.OFF, children_count)
    traced_settle, traced_seal, traced_spans = _drive(
        TelemetryLevel.FINE, children_count
    )

    assert quiet_spans == 0
    assert traced_spans > children_count, "the traced run must really synthesize spans"
    return traced_settle - quiet_settle, traced_seal - quiet_seal


def test_span_synthesis_reads_the_ledger_linearly_in_a_regions_width() -> None:
    narrow_settle, narrow_seal = _traced_reads(_CHILDREN)
    wide_settle, wide_seal = _traced_reads(_CHILDREN * 2)

    assert narrow_settle > 0 and narrow_seal > 0, "sanity: telemetry reads something"
    assert wide_settle < _MAX_GROWTH_FACTOR * narrow_settle, (
        f"doubling the region grew settle reads {narrow_settle} -> {wide_settle}, "
        "which is superlinear"
    )
    assert wide_seal < _MAX_GROWTH_FACTOR * narrow_seal, (
        f"doubling the region grew seal reads {narrow_seal} -> {wide_seal}, "
        "which is superlinear"
    )


def test_rehydrate_rebuilds_the_indexes_to_the_same_spans() -> None:
    live_emitter, live_exporter = emitter(TelemetryLevel.FULL)
    live = engine(spawning_agent_bundle(), emitter=live_emitter)
    for child in _spawn_children(live, 8):
        live.on_dispatched(child, "w1")
        live.on_succeeded(child)
    _seal(live)
    live.on_dispatched("A", "w1")
    live.on_succeeded("A")

    restart_emitter, restart_exporter = emitter(TelemetryLevel.FULL)
    rehydrated(live, emitter=restart_emitter)

    incremental = {span_signature(s) for s in live_exporter.get_finished_spans()}
    rebuilt = {span_signature(s) for s in restart_exporter.get_finished_spans()}
    assert incremental, "expected spans from the live run"
    assert rebuilt == incremental


def test_indexes_pick_up_records_added_after_the_first_emit() -> None:
    """A second spawn wave settles correctly against indexes built for the first."""
    span_emitter, exporter = emitter(TelemetryLevel.FULL)
    eng = engine(spawning_agent_bundle(), emitter=span_emitter)

    first = _spawn_children(eng, 3)
    for child in first:
        eng.on_dispatched(child, "w1")
        eng.on_succeeded(child)
    after_first = len(exporter.get_finished_spans())

    for i in range(3, 6):
        eng.route_boundary_event(
            "A",
            BoundaryEvent(
                kind=BoundaryEventKind.SPAWN,
                call_correlation=f"s{i}",
                child_region_ref="worker",
            ),
        )
    second = [
        a.activation_id
        for a in eng._activations.values()
        if a.kind == "child" and a.activation_id not in set(first)
    ]
    assert len(second) == 3
    for child in second:
        eng.on_dispatched(child, "w1")
        eng.on_succeeded(child)

    assert len(exporter.get_finished_spans()) > after_first
    episode_work_items = {
        dict(s.attributes or {})["flowmesh.physical.work_item_id"]
        for s in exporter.get_finished_spans()
        if s.name == "flowmesh.episode"
    }
    settled = {
        wi.work_item_id
        for wi in eng._work_items.values()
        if wi.activation_id in set(first) | set(second)
    }
    assert settled <= episode_work_items

    _seal(eng)
    assert eng.region_closed("worker:spawn:join")


def test_every_child_activation_gets_a_closed_extent() -> None:
    span_emitter, exporter = emitter(TelemetryLevel.FINE)
    eng = engine(spawning_agent_bundle(), emitter=span_emitter)
    children = _spawn_children(eng, 4)
    for child in children:
        eng.on_dispatched(child, "w1")
        eng.on_succeeded(child)
    _seal(eng)

    operator_spans = [
        s for s in exporter.get_finished_spans() if s.name == "flowmesh.operator"
    ]
    child_ids = set(children)
    covered = {
        dict(s.attributes or {})["flowmesh.logical.activation_id"]
        for s in operator_spans
    }
    assert child_ids <= covered
    for span in operator_spans:
        assert span.start_time is not None and span.end_time is not None
        assert span.end_time >= span.start_time


_DEPTH = 10

# A scope-owning activation's extent covers its whole subtree, so each generation's
# extent subsumes every generation below it. Recomputing that per ancestor doubles the
# work per level added -- exponential, not merely quadratic. Emitting D activations and
# reading each one's subtree once is inherently quadratic in total, so doubling the
# depth may quadruple the work; this separates that from the explosion.
_MAX_DEPTH_GROWTH_FACTOR = 5.0


def _seal_region(eng: OrchestrationEngine, task_id: str) -> None:
    eng.route_boundary_event(
        task_id,
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="seal",
            child_region_ref="worker",
        ),
    )


def _spawn_chain(eng: OrchestrationEngine, depth: int) -> list[str]:
    """One child per generation: a scope tree ``depth`` deep and one wide."""
    chain: list[str] = []
    known: set[str] = set()
    parent = "A"
    for i in range(depth):
        eng.on_dispatched(parent, "w1")
        eng.route_boundary_event(
            parent,
            BoundaryEvent(
                kind=BoundaryEventKind.SPAWN,
                call_correlation=f"s{i}",
                child_region_ref="worker",
            ),
        )
        child = next(
            a.activation_id
            for a in eng._activations.values()
            if a.kind == "child" and a.activation_id not in known
        )
        known.add(child)
        chain.append(child)
        parent = child
    eng.on_dispatched(parent, "w1")
    return chain


def _extent_computations(depth: int, monkeypatch: Any) -> int:
    """Extent derivations run while a chain ``depth`` deep settles innermost-first.

    Counts the derivation itself rather than ledger records, because this defect lives
    entirely in one recursion: a memo that stops working would be invisible under the
    linear reads the rest of a settle does.
    """
    span_emitter, exporter = emitter(TelemetryLevel.FINE)
    eng = engine(recursive_agent_bundle(), emitter=span_emitter)
    chain = _spawn_chain(eng, depth)
    assert len(chain) == depth

    computed = 0
    original = TelemetrySpanEmitter._compute_activation_extent

    def counting(self: Any, activation_id: str, memo: Any) -> Any:
        nonlocal computed
        computed += 1
        return original(self, activation_id, memo)

    monkeypatch.setattr(
        TelemetrySpanEmitter,
        "_compute_activation_extent",
        counting,
    )
    eng.on_succeeded(chain[-1])
    for activation_id in reversed(chain[:-1]):
        _seal_region(eng, activation_id)
        eng.on_succeeded(activation_id)
    _seal_region(eng, "A")
    eng.on_succeeded("A")

    assert len(exporter.get_finished_spans()) > depth, "the run must synthesize spans"
    return computed


def test_span_synthesis_does_not_explode_in_a_regions_depth(
    monkeypatch: Any,
) -> None:
    """Width is guarded above; this is the same defect along the other axis."""
    shallow = _extent_computations(_DEPTH, monkeypatch)
    deep = _extent_computations(_DEPTH * 2, monkeypatch)

    assert shallow > 0, "sanity: the settle derives extents"
    assert deep < _MAX_DEPTH_GROWTH_FACTOR * shallow, (
        f"doubling the nesting grew extent derivations {shallow} -> {deep}, "
        "which is worse than the quadratic a per-activation subtree read costs"
    )
