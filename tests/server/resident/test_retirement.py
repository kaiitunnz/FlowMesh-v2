"""A consumed invocation leaves the working set for its durable record, and nothing a
late report, a re-drive or a restart does afterwards re-runs it or moves its credit.

Each scenario drives the real resident control over in-process durability that keeps
the stored layout as the store does.
"""

import asyncio
import json
from collections.abc import Sequence

import pytest

from server.resident import ClaimState, ClaimTerminalReason, ResidentStores
from server.resident.claim import ClaimTransitionError
from server.resident.durability import MemoryResidentDurability, decode_retired
from server.resident.state import (
    InvocationSubject,
    InvocationSubjectKind,
    RetiredInvocation,
)
from shared.content import reference_for
from shared.outcome import OutcomeManifest
from shared.resident.reports import ResidentBootstrapOutcome, ResidentStreamStatus
from tests.server.resident._helpers import PROFILE
from tests.server.resident.test_service import _ack, _build, _env, _outcome

_SUBJECT = InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-1")


def _served(durability: MemoryResidentDurability | None = None):
    """A workflow invocation admitted, acknowledged and completed by its worker."""
    svc, stores, settled, delivery = _build(
        durability=durability or MemoryResidentDurability()
    )
    asyncio.run(svc._originate(_env()))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))
    manifest = OutcomeManifest(
        content=reference_for("s", b"ok", media_type="text/plain")
    )
    asyncio.run(
        svc._on_outcome(_outcome(svc, ResidentStreamStatus.SUCCESS, manifest=manifest))
    )
    return svc, stores, settled, delivery


def _record(stores: ResidentStores, invocation_id: str = "inv-1") -> RetiredInvocation:
    [value] = stores.durability.retired_invocations([invocation_id])
    assert value is not None
    return decode_retired(invocation_id, value)


def _durability(stores: ResidentStores) -> MemoryResidentDurability:
    assert isinstance(stores.durability, MemoryResidentDurability)
    return stores.durability


def test_a_consumed_invocation_retires_into_its_record() -> None:
    svc, stores, _settled, delivery = _served()
    [claim] = stores.claims.by_invocation("inv-1")

    assert svc.on_invocation_terminal("inv-1").result() is True

    assert claim.state is ClaimState.TERMINAL
    assert not stores.live("inv-1")
    assert claim.replica_id is not None
    assert stores.credit_ledger.held(claim.replica_id) == 0
    assert stores.demand.get(claim.claim_id) is None
    stored = _durability(stores).live
    assert f"claim:{claim.claim_id}" not in stored and "request:inv-1" not in stored
    record = _record(stores)
    assert record.epoch_high == 0
    assert record.terminal_reason is ClaimTerminalReason.COMPLETED
    assert record.subject == _SUBJECT
    assert "resident_sidecar_reap" in delivery.kinds()


def test_a_successor_after_retirement_takes_the_next_epoch_across_a_restart() -> None:
    svc, stores, _settled, _delivery = _served()
    svc.on_invocation_terminal("inv-1").result()

    restarted = ResidentStores(stores.durability)
    restarted.load_snapshot(
        asyncio.run(stores.durability.load_snapshot_async())  # type: ignore[arg-type]
    )
    _fresh, _s, _st, _d = _build(durability=stores.durability)
    successor = _fresh._admission.raise_claim(
        invocation_id="inv-1", subject=_SUBJECT, family="m|chat", profile=PROFILE
    )
    assert successor.admission_epoch == 1

    _fresh._admission.settle_invocation_terminal("inv-1", ClaimTerminalReason.FAILED)
    assert _fresh._admission.retire("inv-1")
    record = _record(_fresh._stores)
    assert record.epoch_high == 1
    assert record.terminal_reason is ClaimTerminalReason.FAILED


def test_a_successor_under_another_owner_is_refused() -> None:
    svc, stores, _settled, _delivery = _served()
    svc.on_invocation_terminal("inv-1").result()

    with pytest.raises(ClaimTransitionError, match="another owner"):
        svc._admission.raise_claim(
            invocation_id="inv-1",
            subject=InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-2"),
            family="m|chat",
            profile=PROFILE,
        )


def test_a_terminal_before_consumption_records_it_and_reaps_without_a_release() -> None:
    svc, stores, _settled, delivery = _build(durability=MemoryResidentDurability())
    asyncio.run(svc._originate(_env()))
    [claim] = stores.claims.by_invocation("inv-1")
    svc._attempts.clear()  # a restart lost the attempt
    svc._admission.on_enqueue_failed(claim)
    delivery.relays.clear()

    assert svc.on_invocation_terminal("inv-1", failed=True).result() is False

    assert [(w, k) for w, k, _ in delivery.relays] == [
        ("wkr-origin", "resident_reap"),
        ("wkr-replica", "resident_sidecar_reap"),
    ]
    assert _record(stores).terminal_reason is ClaimTerminalReason.FAILED
    assert claim.terminal_reason is ClaimTerminalReason.ENQUEUE_FAILED
    assert not stores.live("inv-1")


def test_late_reports_after_retirement_change_nothing() -> None:
    svc, stores, _settled, delivery = _served()
    ack = _ack(svc, ResidentBootstrapOutcome.ACKED)
    outcome = _outcome(svc, ResidentStreamStatus.UNCERTAIN)
    svc.on_invocation_terminal("inv-1").result()
    durable = dict(_durability(stores).live), dict(_durability(stores).retired)
    delivery.relays.clear()

    asyncio.run(svc._on_ack(ack))
    asyncio.run(svc._on_outcome(outcome))
    assert svc.on_invocation_terminal("inv-1").result() is False

    assert delivery.relays == []
    assert svc._redispatched == []  # type: ignore[attr-defined]
    assert not stores.live("inv-1")
    assert (dict(_durability(stores).live), dict(_durability(stores).retired)) == (
        durable
    )


def test_a_predecessors_late_report_never_reaches_its_successor() -> None:
    svc, stores, _settled, _delivery = _build(durability=MemoryResidentDurability())
    asyncio.run(svc._originate(_env()))
    stale = _ack(svc, ResidentBootstrapOutcome.ACKED)
    # A definite rejection ends the first claim; the still-pending boundary re-drives.
    asyncio.run(
        svc._on_ack(_ack(svc, ResidentBootstrapOutcome.REJECTED, rejection="stale"))
    )
    asyncio.run(svc._originate(_env()))
    first, successor = stores.claims.by_invocation("inv-1")
    assert (first.admission_epoch, successor.admission_epoch) == (0, 1)

    asyncio.run(svc._on_ack(stale))

    assert successor.state is ClaimState.RESERVED


class _LostRetirement(MemoryResidentDurability):
    """Applies a retirement, then loses its reply, while ``lose`` is set."""

    lose = True

    def retire_invocation(
        self, record: RetiredInvocation, fields: Sequence[str]
    ) -> None:
        super().retire_invocation(record, fields)
        if self.lose:
            raise ConnectionError("reply lost")


def test_a_retirement_whose_reply_is_lost_stays_live_until_it_is_confirmed() -> None:
    durability = _LostRetirement()
    svc, stores, _settled, _delivery = _served(durability)

    svc.on_invocation_terminal("inv-1").result()

    assert stores.live("inv-1")  # published, but not confirmed
    assert not svc.invocations_settled(["inv-1"], 1.0)
    durability.lose = False
    assert svc._admission.retire("inv-1")
    assert not stores.live("inv-1")
    assert svc.invocations_settled(["inv-1"], 1.0)


class _Crash(MemoryResidentDurability):
    """Fails every retirement before it writes anything, as a crash would."""

    def retire_invocation(
        self, record: RetiredInvocation, fields: Sequence[str]
    ) -> None:
        raise ConnectionError("root crashed")


def test_a_restart_finishes_the_reap_and_retirement_a_crash_interrupted() -> None:
    crashed = _Crash()
    svc, stores, _settled, _delivery = _served(crashed)
    svc.on_invocation_terminal("inv-1").result()
    assert stores.live("inv-1")
    assert json.loads(crashed.live["request:inv-1"])["consumed"] == "completed"

    durability = MemoryResidentDurability()
    durability.live = dict(crashed.live)
    fresh, fresh_stores, _s, delivery = _build(durability=durability)
    fresh.rehydrate(asyncio.run(durability.load_snapshot_async()))
    fresh.reattach_replicas(frozenset({"tsk-serve-1"}))
    fresh.recover_retirements()

    assert [(w, k) for w, k, _ in delivery.relays if k == "resident_sidecar_reap"] == [
        ("wkr-replica", "resident_sidecar_reap")
    ]
    assert not fresh_stores.live("inv-1")
    assert _record(fresh_stores).terminal_reason is ClaimTerminalReason.COMPLETED


def test_a_refused_retirement_writes_nothing() -> None:
    durability = MemoryResidentDurability()
    svc, stores, _settled, _delivery = _build(durability=durability)
    asyncio.run(svc._originate(_env()))
    [claim] = stores.claims.by_invocation("inv-1")
    svc._admission.settle_invocation_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    # The stored claim is still open, as after a write lost before the terminal landed.
    durability.live[f"claim:{claim.claim_id}"] = json.dumps(
        {**json.loads(durability.live[f"claim:{claim.claim_id}"]), "state": "reserved"}
    )
    before = dict(durability.live)

    with pytest.raises(Exception, match="not terminal"):
        svc._admission.retire("inv-1")

    assert durability.live == before and durability.retired == {}
    assert stores.live("inv-1")


def test_a_cancel_during_the_attachment_wait_raises_no_claim() -> None:
    settleable = [True]
    svc, stores, settled, delivery = _build(
        boundary_settleable=lambda _task, _call: settleable[0]
    )

    async def run() -> None:
        svc._replicas_attached.clear()
        origination = asyncio.create_task(svc._originate(_env()))
        await asyncio.sleep(0)
        settleable[0] = False  # the boundary's cancel lands while it waits
        svc._replicas_attached.set()
        await origination

    asyncio.run(run())

    assert stores.claims.all() == []
    assert "resident_handoff" not in delivery.kinds()
    assert "resident_sidecar_bind" not in delivery.kinds()
    assert delivery.kinds() == ["resident_reap"]
    assert settled == []
