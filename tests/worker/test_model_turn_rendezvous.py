"""The worker-local held-model-turn permit rendezvous."""

import threading

import pytest

from shared.tools.contract import MediatedOperationPermit
from shared.utils.ids import new_mediated_permit_id
from worker.model_turn import ModelTurnRendezvous, PermitDenied
from worker.model_turn import rendezvous as rendezvous_module

_AGENT = "tsk-agent"
_CALL = "t0"


def _permit(agent: str = _AGENT, call: str = _CALL) -> MediatedOperationPermit:
    return MediatedOperationPermit(
        permit_id=new_mediated_permit_id(),
        agent_task_id=agent,
        call_correlation=call,
        interface="model",
        subject="model",
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


def test_permit_delivered_to_an_armed_waiter() -> None:
    rv = ModelTurnRendezvous()
    with rv.register(_AGENT, _CALL) as waiter:
        assert rv.deliver_permit(_permit())
        got = waiter.await_permit(timeout=1.0)
    assert isinstance(got, MediatedOperationPermit)
    # The waiter cleared on context exit, so a late permit finds nothing armed.
    assert not rv.deliver_permit(_permit())


def test_deny_delivered_to_an_armed_waiter() -> None:
    rv = ModelTurnRendezvous()
    with rv.register(_AGENT, _CALL) as waiter:
        assert rv.deliver_deny(_AGENT, _CALL, "denied")
        got = waiter.await_permit(timeout=1.0)
    assert isinstance(got, PermitDenied) and got.reason == "denied"


def test_delivery_without_a_waiter_is_refused() -> None:
    rv = ModelTurnRendezvous()
    # Nothing armed: the caller falls back to the async sidecar lane.
    assert not rv.deliver_permit(_permit())
    assert not rv.has_waiter(_AGENT, _CALL)


def test_await_times_out_without_a_delivery() -> None:
    rv = ModelTurnRendezvous()
    with rv.register(_AGENT, _CALL) as waiter:
        assert waiter.await_permit(timeout=0.05) is None


def test_register_before_deliver_closes_the_race() -> None:
    """A permit relayed the instant after arming still wakes the waiter."""
    rv = ModelTurnRendezvous()
    with rv.register(_AGENT, _CALL) as waiter:
        started = threading.Event()

        def relay() -> None:
            started.set()
            rv.deliver_permit(_permit())

        t = threading.Thread(target=relay)
        t.start()
        started.wait()
        got = waiter.await_permit(timeout=1.0)
        t.join()
    assert isinstance(got, MediatedOperationPermit)


def test_a_refused_episode_arms_refused_waiters_until_it_reopens() -> None:
    rv = ModelTurnRendezvous()
    rv.refuse(_AGENT)
    with rv.register(_AGENT, _CALL) as waiter:
        assert waiter.refused
        assert waiter.await_permit(timeout=0.05) is None
    rv.reopen(_AGENT)
    with rv.register(_AGENT, _CALL) as waiter:
        assert not waiter.refused


def test_a_released_episode_arms_denied_waiters_until_it_reopens() -> None:
    rv = ModelTurnRendezvous()
    rv.release(_AGENT, "the model turn was cancelled")
    with rv.register(_AGENT, _CALL) as waiter:
        assert waiter.refused
        assert waiter.await_permit(timeout=0) == PermitDenied(
            reason="the model turn was cancelled"
        )
    rv.reopen(_AGENT)
    with rv.register(_AGENT, _CALL) as waiter:
        assert not waiter.refused


def test_only_the_latest_given_up_episodes_stay_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rendezvous_module, "_MAX_REFUSED_EPISODES", 2)
    rv = ModelTurnRendezvous()
    for agent in ("tsk-1", "tsk-2", "tsk-3"):
        rv.release(agent, "the model turn was cancelled")

    assert list(rv._refused) == ["tsk-2", "tsk-3"]


def test_a_permit_for_a_refused_episode_is_turned_away_until_it_reopens() -> None:
    rv = ModelTurnRendezvous()
    with rv.register(_AGENT, _CALL) as waiter:
        rv.refuse(_AGENT)
        assert not rv.deliver_permit(_permit())
        assert rv.was_held(_AGENT, _CALL)
        rv.release(_AGENT, "the model turn was cancelled")
        assert waiter.await_permit(timeout=1.0) == PermitDenied(
            reason="the model turn was cancelled"
        )
    rv.reopen(_AGENT)
    with rv.register(_AGENT, _CALL) as waiter:
        assert rv.deliver_permit(_permit())
        assert isinstance(waiter.await_permit(timeout=1.0), MediatedOperationPermit)


def test_an_older_waiter_exiting_late_leaves_a_newer_one_armed() -> None:
    rv = ModelTurnRendezvous()
    older = rv.register(_AGENT, _CALL)
    with rv.register(_AGENT, _CALL) as newer:
        older.__exit__(None, None, None)
        assert rv.deliver_permit(_permit())
        assert isinstance(newer.await_permit(timeout=1.0), MediatedOperationPermit)
    assert not rv.has_waiter(_AGENT, _CALL)
