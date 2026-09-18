"""Span synthesis must not add measurable serialization to a ledger transition.

Every emitter entry point runs under the runtime lock, so its cost is paid by the
workflow it observes. These drive a wide spawn region -- the shape where a per-emit
ledger rescan compounds -- and compare the settle loop against the same loop with
telemetry off, so the bound holds regardless of how fast the host is.
"""

import time

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

# Enough headroom for a loaded CI host and the emitter's own constant factor, far
# below the tens-to-hundreds multiple a per-emit ledger rescan costs at this width.
_MAX_OVERHEAD_FACTOR = 8.0
_TIMER_FLOOR_SEC = 0.05


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


def _drive(level: TelemetryLevel) -> tuple[float, float, int]:
    """Seconds spent in the settle loop and in the seal, plus the spans emitted."""
    span_emitter, exporter = emitter(level)
    eng = engine(spawning_agent_bundle(), emitter=span_emitter)
    children = _spawn_children(eng, _CHILDREN)
    assert len(children) == _CHILDREN

    started = time.perf_counter()
    for child in children:
        eng.on_dispatched(child, "w1")
        eng.on_succeeded(child)
    settle = time.perf_counter() - started

    started = time.perf_counter()
    _seal(eng)
    seal = time.perf_counter() - started
    return settle, seal, len(exporter.get_finished_spans())


def test_settling_a_wide_region_costs_about_the_same_with_traces_on() -> None:
    quiet_settle, quiet_seal, quiet_spans = _drive(TelemetryLevel.OFF)
    traced_settle, traced_seal, traced_spans = _drive(TelemetryLevel.FINE)

    assert quiet_spans == 0
    assert traced_spans > _CHILDREN, "the traced run must really synthesize spans"

    assert traced_settle < _MAX_OVERHEAD_FACTOR * quiet_settle + _TIMER_FLOOR_SEC, (
        f"settling {_CHILDREN} children took {traced_settle:.3f}s at fine vs "
        f"{quiet_settle:.3f}s off"
    )
    assert traced_seal < _MAX_OVERHEAD_FACTOR * quiet_seal + _TIMER_FLOOR_SEC, (
        f"sealing a {_CHILDREN}-child region took {traced_seal:.3f}s at fine vs "
        f"{quiet_seal:.3f}s off"
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
