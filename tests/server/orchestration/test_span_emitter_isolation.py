"""A failing span synthesis must not reach the ledger transition that triggered it.

Every emitter entry point is called from inside a transition that has already begun
mutating durable state, so these drive real settles with synthesis forced to fail and
assert the ledger outcome -- the slot published, the successor released, the restart
completed -- rather than merely that nothing propagated out.
"""

import pytest

from server.orchestration import WorkItemStatus
from server.orchestration import telemetry as telemetry_module
from server.orchestration.state import BoundaryEvent
from server.orchestration.telemetry import TelemetrySpanEmitter
from server.task.v2.representations.operators import BoundaryEventKind
from shared.telemetry.config import TelemetryLevel
from tests.server.orchestration.helpers import (
    chain_bundle,
    emitter,
    engine,
    rehydrated,
    spawning_agent_bundle,
)


class _Boom(RuntimeError):
    pass


@pytest.fixture
def failing_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every span derivation raise, at the first durable value it reads."""

    def boom(value: str) -> int:
        raise _Boom(f"synthesis failed on {value!r}")

    monkeypatch.setattr(telemetry_module, "iso_to_ns", boom)


def test_a_failing_emit_still_publishes_the_slot_and_releases_successors(
    failing_synthesis: None,
) -> None:
    span_emitter, _exporter = emitter(TelemetryLevel.FULL)
    eng = engine(chain_bundle(), emitter=span_emitter)

    eng.on_dispatched("A", "w1")
    advance = eng.on_succeeded("A")

    wi = eng.work_item("A")
    assert wi is not None and wi.status is WorkItemStatus.SETTLED
    assert eng.resolve_output("out:A") is not None, "the result slot must be published"
    assert advance.ready == ["B"], "the successor must be released"

    # The released successor is really runnable: driving it to its own settle closes
    # the workflow's remaining output.
    eng.on_dispatched("B", "w1")
    eng.on_succeeded("B")
    assert eng.resolve_output("out:B") is not None


def test_a_failing_emit_leaves_the_ledger_identical_to_no_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def drive(span_emitter: TelemetrySpanEmitter | None) -> tuple[list[str], object]:
        eng = engine(chain_bundle(), emitter=span_emitter)
        eng.on_dispatched("A", "w1")
        eng.on_succeeded("A")
        eng.on_dispatched("B", "w1")
        eng.on_succeeded("B")
        published = eng.resolve_output("out:B")
        assert published is not None
        return (
            [kind for kind, _ in eng.contract_trace()],
            published.model_dump(exclude={"at"}),
        )

    quiet_trace, quiet_output = drive(None)

    def boom(value: str) -> int:
        raise _Boom(value)

    monkeypatch.setattr(telemetry_module, "iso_to_ns", boom)
    failing_emitter, _exporter = emitter(TelemetryLevel.FULL)
    failing_trace, failing_output = drive(failing_emitter)

    assert failing_trace == quiet_trace
    assert failing_output == quiet_output


def test_a_failing_export_does_not_break_a_settle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise _Boom("export failed")

    monkeypatch.setattr(telemetry_module, "_export_span", boom)
    span_emitter, _exporter = emitter(TelemetryLevel.FULL)
    eng = engine(chain_bundle(), emitter=span_emitter)

    eng.on_dispatched("A", "w1")
    advance = eng.on_succeeded("A")

    assert advance.ready == ["B"]
    assert eng.resolve_output("out:A") is not None


def test_an_unclassifiable_activation_drops_its_span_instead_of_wedging_the_settle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ActivationClassificationError`` is a real trigger raised by the emitter."""
    original = telemetry_module.TelemetrySpanEmitter._activation_extent

    def refuse(self: object, activation_id: str) -> tuple[int, int] | None:
        raise telemetry_module.ActivationClassificationError(activation_id)

    monkeypatch.setattr(
        telemetry_module.TelemetrySpanEmitter, "_activation_extent", refuse
    )
    assert original is not refuse

    span_emitter, exporter = emitter(TelemetryLevel.FULL)
    eng = engine(chain_bundle(), emitter=span_emitter)
    eng.on_dispatched("A", "w1")
    advance = eng.on_succeeded("A")

    assert advance.ready == ["B"]
    assert eng.resolve_output("out:A") is not None
    assert not [
        s for s in exporter.get_finished_spans() if s.name == "flowmesh.operator"
    ], "the unclassifiable activation's span is dropped, not guessed"


def test_a_failing_attach_still_builds_a_restartable_engine(
    failing_synthesis: None,
) -> None:
    live_emitter, _exporter = emitter(TelemetryLevel.FULL)
    live = engine(spawning_agent_bundle(), emitter=live_emitter)
    live.on_dispatched("A", "w1")
    live.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN,
            call_correlation="s0",
            child_region_ref="worker",
        ),
    )
    child = next(
        a.activation_id for a in live._activations.values() if a.kind == "child"
    )
    live.on_dispatched(child, "w1")
    live.on_succeeded(child)

    restart_emitter, _restart_exporter = emitter(TelemetryLevel.FULL)
    restarted = rehydrated(live, emitter=restart_emitter)

    # The restarted engine is usable: the region seals and the agent settles, which is
    # only reachable if rehydration completed rather than aborting mid-pass.
    restarted.route_boundary_event(
        "A",
        BoundaryEvent(
            kind=BoundaryEventKind.SPAWN_SEAL,
            call_correlation="s1",
            child_region_ref="worker",
        ),
    )
    assert restarted.region_closed("worker:spawn:join")
    restarted.on_dispatched("A", "w1")
    restarted.on_succeeded("A")
    assert restarted.resolve_output("out:A") is not None
