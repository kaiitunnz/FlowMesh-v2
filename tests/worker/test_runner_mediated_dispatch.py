"""Runner routing of a relayed mediated-op frame.

A permit relayed over the attachment routes to a held facade's rendezvous when one is
armed; otherwise a non-model permit drives the async sidecar while a waiterless model
permit is dropped as stale. A deny frame only ever wakes a held facade.
``_route_mediated_op`` does not use the rest of the Runner, so it is driven directly
with a lightweight stub self, mirroring ``_select_inference_executor_key``.
"""

import logging
from types import SimpleNamespace
from typing import Any, cast

from shared.tools.contract import MediatedOperationPermit
from shared.utils.ids import new_mediated_permit_id
from worker.model_turn import ModelTurnRendezvous, PermitDenied
from worker.runner import Runner

_AGENT = "tsk-agent"
_CALL = "t0"


class _StubSidecar:
    def __init__(self) -> None:
        self.submitted: list[MediatedOperationPermit] = []
        self.reaped: list[tuple[str, str]] = []

    def submit_permit(self, permit: MediatedOperationPermit) -> None:
        self.submitted.append(permit)

    def reap(self, agent_task_id: str, call_correlation: str) -> None:
        self.reaped.append((agent_task_id, call_correlation))


def _self(rv: ModelTurnRendezvous, sidecar: _StubSidecar) -> Any:
    return SimpleNamespace(
        _model_turn_rendezvous=rv,
        _ensure_mediated_sidecar=lambda: sidecar,
        logger=logging.getLogger("runner-dispatch-test"),
    )


def _permit(interface: str = "model") -> MediatedOperationPermit:
    return MediatedOperationPermit(
        permit_id=new_mediated_permit_id(),
        agent_task_id=_AGENT,
        call_correlation=_CALL,
        interface=interface,
        subject=interface,
        invocation_id="inv-1",
        idempotency_key="idm-1",
        request_digest="d",
        target_id="wkr-1",
        target_generation=3,
        deadline_epoch=2_000_000_000.0,
        max_results=1,
        timeout_sec=10.0,
        result_char_cap=4000,
    )


def _route(fake_self: Any, kind: str, frame: dict[str, Any]) -> None:
    Runner._route_mediated_op(cast(Runner, fake_self), kind, frame)


def test_permit_with_an_armed_waiter_wakes_the_facade_not_the_sidecar() -> None:
    rv, sidecar = ModelTurnRendezvous(), _StubSidecar()
    with rv.register(_AGENT, _CALL) as waiter:
        _route(_self(rv, sidecar), "permit", _permit().model_dump(mode="json"))
        got = waiter.await_permit(timeout=1.0)
    assert isinstance(got, MediatedOperationPermit)
    assert sidecar.submitted == []  # a held permit never drives the async lane


def test_a_stale_held_model_permit_is_dropped_not_egressed() -> None:
    # A held-turn model permit whose waiter has already left is stale; egressing it on
    # the async lane would duplicate the call and reap a concurrent retry's request, so
    # it is dropped and recovery re-proposes under a fresh permit.
    rv, sidecar = ModelTurnRendezvous(), _StubSidecar()
    with rv.register(_AGENT, _CALL):
        pass  # the held waiter registered and left (its turn timed out)
    _route(_self(rv, sidecar), "permit", _permit("model").model_dump(mode="json"))
    assert sidecar.submitted == []


def test_a_durable_yield_model_permit_never_held_drives_the_sidecar() -> None:
    # A durable-yield model permit never armed a waiter, so it drives the async lane.
    rv, sidecar = ModelTurnRendezvous(), _StubSidecar()
    _route(_self(rv, sidecar), "permit", _permit("model").model_dump(mode="json"))
    assert len(sidecar.submitted) == 1


def test_a_waiterless_search_permit_drives_the_async_sidecar() -> None:
    rv, sidecar = ModelTurnRendezvous(), _StubSidecar()
    _route(_self(rv, sidecar), "permit", _permit("search/v1").model_dump(mode="json"))
    assert len(sidecar.submitted) == 1


def test_deny_frame_wakes_a_held_facade() -> None:
    rv, sidecar = ModelTurnRendezvous(), _StubSidecar()
    with rv.register(_AGENT, _CALL) as waiter:
        _route(
            _self(rv, sidecar),
            "deny",
            {"agent_task_id": _AGENT, "call_correlation": _CALL, "reason": "nope"},
        )
        got = waiter.await_permit(timeout=1.0)
    assert isinstance(got, PermitDenied) and got.reason == "nope"
    assert sidecar.submitted == []


def test_reap_frame_reaps_on_the_sidecar() -> None:
    rv, sidecar = ModelTurnRendezvous(), _StubSidecar()
    _route(
        _self(rv, sidecar),
        "reap",
        {"agent_task_id": _AGENT, "call_correlation": _CALL},
    )
    assert sidecar.reaped == [(_AGENT, _CALL)]
