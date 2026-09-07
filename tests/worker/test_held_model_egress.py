"""The held model turn's authorize-before-egress round trip."""

from typing import Any

from shared.tools.contract import AgentModelTurnProposal, MediatedOperationPermit
from shared.tools.model.schema import (
    MODEL_INTERFACE,
    ModelCompletion,
    ModelRequest,
    model_request_digest,
)
from shared.utils.ids import new_mediated_permit_id
from worker.egress import HeldEgressReject, PendingEgressRequestStore
from worker.model_turn import HeldModelEgress, ModelTurnRendezvous

_AGENT = "tsk-agent"
_CALL = "t0"
_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_REQUEST = ModelRequest(interface=MODEL_INTERFACE, url="http://up/v1", body=_BODY)


def _permit() -> MediatedOperationPermit:
    return MediatedOperationPermit(
        permit_id=new_mediated_permit_id(),
        agent_task_id=_AGENT,
        call_correlation=_CALL,
        interface=MODEL_INTERFACE,
        subject=MODEL_INTERFACE,
        invocation_id="inv-1",
        idempotency_key="idm-1",
        request_digest=model_request_digest(MODEL_INTERFACE, "http://up/v1", _BODY),
        target_id="wkr-1",
        target_generation=3,
        deadline_epoch=2_000_000_000.0,
        max_results=1,
        timeout_sec=10.0,
        result_char_cap=1_000_000,
    )


class _StubSidecar:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.seen: MediatedOperationPermit | None = None

    def egress_now(self, permit: MediatedOperationPermit) -> Any:
        self.seen = permit
        return self._result


def _egress(rv: ModelTurnRendezvous, propose: Any, sidecar: Any) -> HeldModelEgress:
    return HeldModelEgress(
        rendezvous=rv,
        pending=PendingEgressRequestStore(),
        propose=propose,
        sidecar=sidecar,
        timeout_sec=1.0,
    )


def test_success_returns_the_completion_and_reaps_custody() -> None:
    rv = ModelTurnRendezvous()
    completion = ModelCompletion(content="a reply")
    sidecar = _StubSidecar(completion)
    seen: dict[str, Any] = {}

    def propose(p: AgentModelTurnProposal) -> None:
        # The waiter is armed before the propose is emitted (the race is closed), and
        # the propose carries the request digest.
        seen["armed"] = rv.has_waiter(_AGENT, _CALL)
        seen["digest"] = p.request_digest
        rv.deliver_permit(_permit())

    egress = _egress(rv, propose, sidecar)
    result = egress.run(_AGENT, _CALL, _REQUEST)
    assert result is completion
    assert seen["armed"] is True
    expected_digest = model_request_digest(MODEL_INTERFACE, "http://up/v1", _BODY)
    assert seen["digest"] == expected_digest
    assert sidecar.seen is not None
    # Custody is dropped once the attempt resolves, and the waiter cleared.
    assert not rv.has_waiter(_AGENT, _CALL)


def test_denial_is_terminal() -> None:
    rv = ModelTurnRendezvous()
    sidecar = _StubSidecar(ModelCompletion(content="unused"))

    def propose(p: AgentModelTurnProposal) -> None:
        rv.deliver_deny(_AGENT, _CALL, "model turn egress denied")

    result = _egress(rv, propose, sidecar).run(_AGENT, _CALL, _REQUEST)
    assert isinstance(result, HeldEgressReject) and "denied" in result.reason
    assert sidecar.seen is None  # a denial never egresses


def test_permit_never_arrives_is_terminal() -> None:
    rv = ModelTurnRendezvous()
    sidecar = _StubSidecar(ModelCompletion(content="unused"))
    result = _egress(rv, lambda _p: None, sidecar).run(_AGENT, _CALL, _REQUEST)
    assert isinstance(result, HeldEgressReject) and "never came" in result.reason
    assert not rv.has_waiter(_AGENT, _CALL)


def test_propose_fault_is_terminal() -> None:
    rv = ModelTurnRendezvous()
    sidecar = _StubSidecar(ModelCompletion(content="unused"))

    def propose(p: AgentModelTurnProposal) -> None:
        raise RuntimeError("event stream not ready")

    result = _egress(rv, propose, sidecar).run(_AGENT, _CALL, _REQUEST)
    assert isinstance(result, HeldEgressReject)
    assert sidecar.seen is None
    assert not rv.has_waiter(_AGENT, _CALL)


def test_sidecar_fence_reject_propagates() -> None:
    rv = ModelTurnRendezvous()
    sidecar = _StubSidecar(HeldEgressReject(reason="permit fence rejected: digest"))

    def propose(p: AgentModelTurnProposal) -> None:
        rv.deliver_permit(_permit())

    result = _egress(rv, propose, sidecar).run(_AGENT, _CALL, _REQUEST)
    assert isinstance(result, HeldEgressReject) and "fence" in result.reason
