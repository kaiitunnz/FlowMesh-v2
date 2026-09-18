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
from server.task.v2.representations.operators import BoundaryEventKind
from shared.telemetry.config import TelemetryLevel
from tests.server.orchestration.helpers import (
    emitter,
    engine,
    rehydrated,
    span_signature,
    spawning_agent_bundle,
)

_CHILDREN = 1500

# A per-settle rescan reads the whole trace each time, so the records it touches grow
# with the square of the region's width -- around a million at this one. A bound that
# is generous per settle is still three orders of magnitude below that.
_MAX_RECORDS_READ_PER_SETTLE = 40


class _CountingTrace(list):
    """A ledger trace that counts the records read from it.

    The cost this guards is algorithmic, so it is asserted as work done rather than as
    elapsed time: a wall-clock bound on a shared machine measures the machine.
    """

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.records_read = 0

    def __iter__(self) -> Any:
        self.records_read += len(self)
        return super().__iter__()

    def __getitem__(self, index: Any) -> Any:
        item = super().__getitem__(index)
        self.records_read += len(item) if isinstance(index, slice) else 1
        return item


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
    return [
        a.activation_id
        for a in eng._activations.values()  # noqa: SLF001 - test inspection
        if a.kind == "child"
    ]


def _seal(eng: OrchestrationEngine) -> None:
    eng.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="seal",
            child_region_ref="worker",
        ),
    )


def _drive(level: TelemetryLevel) -> tuple[int, int, int]:
    """Records read during the settle loop and during the seal, plus spans emitted."""
    span_emitter, exporter = emitter(level)
    eng = engine(spawning_agent_bundle(), emitter=span_emitter)
    # The emitter reads the trace list the engine handed it at attach, so the counting
    # list has to replace that one before anything binds to it.
    trace = _CountingTrace(eng._trace)  # noqa: SLF001 - test instrumentation
    eng._trace = trace  # noqa: SLF001 - test instrumentation
    span_emitter.attach(
        activations=eng._activations,  # noqa: SLF001 - test instrumentation
        scopes=eng._scopes,  # noqa: SLF001 - test instrumentation
        work_items=eng._work_items,  # noqa: SLF001 - test instrumentation
        attempts=eng._attempts,  # noqa: SLF001 - test instrumentation
        invocations=eng._invocations,  # noqa: SLF001 - test instrumentation
        trace=trace,
        released_scopes=eng._released_scopes,  # noqa: SLF001 - test instrumentation
    )
    children = _spawn_children(eng, _CHILDREN)
    assert len(children) == _CHILDREN

    before = trace.records_read
    for child in children:
        eng.on_dispatched(child, "w1")
        eng.on_succeeded(child)
    settle = trace.records_read - before

    before = trace.records_read
    _seal(eng)
    seal = trace.records_read - before
    return settle, seal, len(exporter.get_finished_spans())


def test_settling_a_wide_region_reads_the_ledger_a_bounded_number_of_times() -> None:
    quiet_settle, quiet_seal, quiet_spans = _drive(TelemetryLevel.OFF)
    traced_settle, traced_seal, traced_spans = _drive(TelemetryLevel.FINE)

    assert quiet_spans == 0
    assert traced_spans > _CHILDREN, "the traced run must really synthesize spans"

    budget = _MAX_RECORDS_READ_PER_SETTLE * _CHILDREN
    assert traced_settle - quiet_settle < budget, (
        f"settling {_CHILDREN} children read {traced_settle - quiet_settle} extra "
        f"ledger records at fine, over a {budget} budget"
    )
    assert traced_seal - quiet_seal < budget, (
        f"sealing a {_CHILDREN}-child region read {traced_seal - quiet_seal} extra "
        f"ledger records at fine, over a {budget} budget"
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
        for a in eng._activations.values()  # noqa: SLF001 - test inspection
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
        for wi in eng._work_items.values()  # noqa: SLF001 - test inspection
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
