"""What a store failure, a lost reply or a restart leaves behind lands without a
further restart: no credit strands, no client waits on the store, no inert replica or
spent family stays live, and the listings answer as before.

Each scenario drives the real resident control over in-process durability that keeps
the stored layout as the store does.
"""

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence

import pytest

from server.resident import (
    AdmissionController,
    ClaimState,
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
    ReplicaState,
    RetiredInvocation,
    ServiceFamily,
    StoredReplica,
)
from server.resident.durability import (
    CLAIM,
    FAMILY,
    REPLICA,
    REQUEST,
    MemoryResidentDurability,
    live_field,
)
from server.resident.store_retry import SETTLEMENT
from shared.content import reference_for
from shared.outcome import OutcomeManifest
from shared.resident.reports import ResidentBootstrapOutcome, ResidentStreamStatus
from tests.server.resident._helpers import PROFILE, stored_replica, warm_stores
from tests.server.resident.test_serve_origination import (
    _FAMILY,
    _SERVE_TASK,
)
from tests.server.resident.test_serve_origination import _ack as _serve_ack
from tests.server.resident.test_serve_origination import (
    _adopt,
)
from tests.server.resident.test_serve_origination import _build as _build_serve
from tests.server.resident.test_serve_origination import (
    _origination,
)
from tests.server.resident.test_serve_origination import _outcome as _serve_outcome
from tests.server.resident.test_serve_origination import (
    _ServeDelivery,
)
from tests.server.resident.test_service import _ack, _build, _env, _outcome

_SUBJECT = InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-1")
_MANIFEST = OutcomeManifest(content=reference_for("s", b"ok", media_type="text/plain"))


def _accepted(durability: MemoryResidentDurability | None = None):
    svc, stores, settled, delivery = _build(
        durability=durability or MemoryResidentDurability()
    )
    asyncio.run(svc._originate(_env()))
    asyncio.run(svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED)))
    return svc, stores, settled, delivery


def _served(durability: MemoryResidentDurability | None = None):
    svc, stores, settled, delivery = _accepted(durability)
    asyncio.run(
        svc._on_outcome(_outcome(svc, ResidentStreamStatus.SUCCESS, manifest=_MANIFEST))
    )
    return svc, stores, settled, delivery


class _FailingListenerWrite(MemoryResidentDurability):
    """Refuses the first write that would make a replica's listener fence durable."""

    fail = True

    def write(self, changes: Mapping[str, str | None]) -> None:
        if self.fail and any(
            v is not None and '"listener":{' in v for v in changes.values()
        ):
            self.fail = False
            raise ConnectionError("store unavailable")
        super().write(changes)


def test_a_serve_origination_failing_after_admission_releases_its_credit() -> None:
    svc, stores, _settled, _deps = _build_serve()
    stores.durability = _FailingListenerWrite()
    _adopt(svc)
    delivery = _ServeDelivery()

    asyncio.run(svc._originate_serve(_origination(delivery)))

    assert all(
        claim.state is ClaimState.TERMINAL
        for claim in stores.claims.by_invocation("inv-1")
    )
    assert delivery.terminals == [(ClaimTerminalReason.FAILED, delivery.failed)]
    assert delivery.failed is not None
    [replica] = stores.directory.by_family(_FAMILY)
    assert stores.credit_ledger.held(replica.replica_id) == 0
    svc.drain_serve_replica(_SERVE_TASK)
    assert stores.directory.live_by_family(_FAMILY) == []


class _LostOnce(MemoryResidentDurability):
    """Applies the first retirement and loses its reply."""

    lose = True

    def retire_invocation(
        self, record: RetiredInvocation, fields: Sequence[str]
    ) -> None:
        super().retire_invocation(record, fields)
        if self.lose:
            self.lose = False
            raise ConnectionError("reply lost")


def test_a_retirement_whose_reply_was_lost_retires_on_its_retry() -> None:
    svc, stores, _settled, _delivery = _served(_LostOnce())

    svc.on_invocation_terminal("inv-1").result()
    assert stores.live("inv-1")
    assert svc._store_retry.pending == [(REQUEST, "inv-1")]
    assert not svc.invocations_settled(["inv-1"], 1.0)

    svc.retry_store_failures()

    assert not stores.live("inv-1")
    assert svc._store_retry.pending == []
    assert svc.invocations_settled(["inv-1"], 1.0)


class _ArchiveFailsOnce(MemoryResidentDurability):
    fail = True

    def archive_replica(self, stored: StoredReplica, fields: Sequence[str]) -> None:
        if self.fail:
            self.fail = False
            raise ConnectionError("store unavailable")
        super().archive_replica(stored, fields)


def test_a_replica_whose_archive_failed_moves_to_history_on_its_retry() -> None:
    svc, stores, _settled, _delivery = _accepted(_ArchiveFailsOnce())
    [claim] = stores.claims.by_invocation("inv-1")
    assert claim.replica_id is not None
    svc._lifecycle.drain(claim.replica_id)
    asyncio.run(
        svc._on_outcome(_outcome(svc, ResidentStreamStatus.SUCCESS, manifest=_MANIFEST))
    )
    svc.on_invocation_terminal("inv-1").result()
    assert svc._store_retry.pending == [(REPLICA, claim.replica_id)]

    svc.retry_store_failures()

    assert stores.directory.get(claim.replica_id) is None
    assert stored_replica(stores, claim.replica_id).state is ReplicaState.STOPPED


class _FamilyArchiveFailsOnce(MemoryResidentDurability):
    fail = True

    def archive_family(self, family: ServiceFamily, field: str) -> None:
        if self.fail:
            self.fail = False
            raise ConnectionError("store unavailable")
        super().archive_family(family, field)


def test_a_family_whose_archive_failed_moves_to_history_on_its_retry() -> None:
    svc, stores, _settled, _deps = _build_serve()
    stores.durability = durability = _FamilyArchiveFailsOnce()
    _adopt(svc)

    svc.drain_serve_replica(_SERVE_TASK)
    assert stores.directory.by_family(_FAMILY) == []
    assert stores.families.get(_FAMILY) is not None
    assert svc._store_retry.pending == [(FAMILY, _FAMILY)]

    svc.retry_store_failures()

    assert stores.families.get(_FAMILY) is None
    assert _FAMILY in durability.families
    assert svc._store_retry.pending == []


def test_a_drained_replica_moves_to_history_once_its_last_invocation_retires() -> None:
    svc, stores, _settled, _delivery = _accepted()
    [claim] = stores.claims.by_invocation("inv-1")
    replica_id = claim.replica_id
    assert replica_id is not None
    svc._lifecycle.drain(replica_id)
    svc._lifecycle.stop(replica_id)
    replica = stores.directory.get(replica_id)
    assert replica is not None and replica.state is ReplicaState.DRAINING

    asyncio.run(
        svc._on_outcome(_outcome(svc, ResidentStreamStatus.SUCCESS, manifest=_MANIFEST))
    )
    svc.on_invocation_terminal("inv-1").result()

    assert not stores.live("inv-1")
    assert stores.directory.get(replica_id) is None
    assert stored_replica(stores, replica_id).state is ReplicaState.STOPPED


def test_a_preempted_replica_moves_to_history_once_its_last_invocation_retires() -> (
    None
):
    svc, stores, _settled, _delivery = _accepted()
    [claim] = stores.claims.by_invocation("inv-1")
    replica_id = claim.replica_id
    assert replica_id is not None
    svc._admission.on_route_loss(claim)
    replica = stores.directory.get(replica_id)
    assert replica is not None
    replica.standing = False
    svc._lifecycle.on_preempt(replica_id)

    svc.on_invocation_terminal("inv-1", failed=True).result()

    assert stores.directory.get(replica_id) is None
    assert stored_replica(stores, replica_id).state is ReplicaState.PREEMPTED


def test_a_retired_invocation_reads_settled_after_a_restart_with_a_live_replica() -> (
    None
):
    durability = MemoryResidentDurability()
    svc, _stores, _s, _d = _served(durability)
    svc.on_invocation_terminal("inv-1").result()

    fresh, fresh_stores, _s2, _d2 = _build(durability=durability)
    fresh.rehydrate(asyncio.run(durability.load_snapshot_async()))
    fresh.reattach_replicas(frozenset({"tsk-serve-1"}))
    fresh.recover_retirements()

    assert fresh_stores.directory.all()
    assert fresh.invocations_retired(["inv-1"])


def test_a_serve_request_pending_at_a_crash_retires_on_the_restart() -> None:
    durability = MemoryResidentDurability()
    svc, _stores, _s, _d = _build(durability=durability)
    svc._admission.raise_claim(
        invocation_id="inv-serve",
        subject=InvocationSubject(kind=InvocationSubjectKind.EXTERNAL, id="key-1"),
        family="m|chat",
        profile=PROFILE,
    )

    fresh, fresh_stores, _s2, _d2 = _build(durability=durability)
    fresh.rehydrate(asyncio.run(durability.load_snapshot_async()))
    fresh.reattach_replicas(frozenset())
    fresh.recover_retirements()

    assert not fresh_stores.live("inv-serve")
    [record] = durability.retired_invocations(["inv-serve"])
    assert record is not None
    assert record.terminal_reason is ClaimTerminalReason.FAILED


def test_a_re_adopted_serve_family_keeps_its_stored_definition() -> None:
    svc, stores, _settled, _deps = _build_serve()
    _adopt(svc)
    original = stores.families.get(_FAMILY)
    assert original is not None
    svc.drain_serve_replica(_SERVE_TASK)
    assert stores.families.get(_FAMILY) is None
    durability = stores.durability
    assert isinstance(durability, MemoryResidentDurability)
    assert _FAMILY in durability.families

    _adopt(svc)

    restored = stores.families.get(_FAMILY)
    assert restored is not None
    assert restored.created_at == original.created_at
    [listed] = [
        family
        for family in asyncio.run(svc.list_service_families_async())
        if family.family == _FAMILY
    ]
    assert listed.created_at == original.created_at
    assert json.loads(listed.model_dump_json()) == json.loads(
        restored.model_dump_json()
    )


class _Outage(MemoryResidentDurability):
    """A store that goes down at the first write making a replica's listener fence
    durable, unless ``trip`` is unset, and stays down until restored."""

    trip = True
    down = False

    def write(self, changes: Mapping[str, str | None]) -> None:
        if self.trip and any(
            v is not None and '"listener":{' in v for v in changes.values()
        ):
            self.trip = False
            self.down = True
        if self.down:
            raise ConnectionError("store unavailable")
        super().write(changes)


class _StoredTerminalDelivery(_ServeDelivery):
    """Records its terminal through the store, as the gated edge does."""

    def __init__(self, store: _Outage) -> None:
        super().__init__()
        self.store = store

    def record_terminal(self, reason: ClaimTerminalReason, detail: str | None) -> None:
        if self.store.down:
            raise ConnectionError("serve snapshot write failed")
        super().record_terminal(reason, detail)


def test_a_serve_settlement_through_an_outage_lands_once_the_store_returns() -> None:
    svc, stores, _settled, _deps = _build_serve()
    _adopt(svc)
    stores.durability = outage = _Outage()
    [replica] = stores.directory.by_family(_FAMILY)
    delivery = _StoredTerminalDelivery(outage)

    asyncio.run(svc._originate_serve(_origination(delivery)))
    svc.retry_store_failures()

    assert delivery.failed is not None and not delivery.completed
    assert stores.credit_ledger.held(replica.replica_id) == 1
    assert svc._store_retry.pending == [(SETTLEMENT, "inv-1")]

    outage.down = False
    svc.retry_store_failures()

    assert stores.credit_ledger.held(replica.replica_id) == 0
    assert not stores.live("inv-1")
    assert svc._store_retry.pending == []
    assert delivery.terminals == [(ClaimTerminalReason.FAILED, delivery.failed)]


def test_a_serve_outcome_answers_its_client_while_the_store_is_down() -> None:
    svc, stores, _settled, _deps = _build_serve()
    stores.durability = outage = _Outage()
    outage.trip = False
    _adopt(svc)
    [replica] = stores.directory.by_family(_FAMILY)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    asyncio.run(svc._on_ack(_serve_ack(svc, ResidentBootstrapOutcome.ACKED)))
    outage.down = True

    asyncio.run(svc._on_outcome(_serve_outcome(svc, ResidentStreamStatus.SUCCESS)))

    assert delivery.completed and delivery.failed is None
    assert stores.live("inv-1")
    assert svc._store_retry.pending == [(SETTLEMENT, "inv-1")]

    asyncio.run(svc._on_outcome(_serve_outcome(svc, ResidentStreamStatus.SUCCESS)))

    assert delivery.completed and delivery.failed is None
    assert svc._store_retry.pending == [(SETTLEMENT, "inv-1")]

    # No request gets a handoff while the release is not durable.
    later = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(later, "inv-2")))
    assert later.opened == [] and later.failed is not None

    outage.down = False
    svc.retry_store_failures()

    assert not stores.live("inv-1") and not stores.live("inv-2")
    assert stores.credit_ledger.held(replica.replica_id) == 0
    assert svc._store_retry.pending == []
    [record] = outage.retired_invocations(["inv-1"])
    assert record is not None
    assert record.terminal_reason is ClaimTerminalReason.COMPLETED


class _HistoryReadFailsOnce(MemoryResidentDurability):
    fail = True

    def stored_family(self, family: str) -> ServiceFamily | None:
        if self.fail:
            self.fail = False
            raise ConnectionError("store unavailable")
        return super().stored_family(family)


def _unreadable_history() -> MemoryResidentDurability:
    durability = MemoryResidentDurability()
    durability.families[_FAMILY] = "not json"
    return durability


@pytest.mark.parametrize(
    "durability",
    [_HistoryReadFailsOnce, _unreadable_history],
    ids=["read-fails", "unreadable"],
)
def test_a_serve_task_adopts_when_its_family_history_cannot_be_read(
    durability: Callable[[], MemoryResidentDurability],
) -> None:
    svc, stores, _settled, _deps = _build_serve()
    stores.durability = durability()

    _adopt(svc)

    assert stores.families.get(_FAMILY) is not None
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    assert delivery.opened and delivery.failed is None


def test_a_replica_archived_on_its_retry_takes_its_unused_family_with_it() -> None:
    svc, stores, _settled, _deps = _build_serve()
    stores.durability = durability = _ArchiveFailsOnce()
    _adopt(svc)

    svc.drain_serve_replica(_SERVE_TASK)
    assert stores.families.get(_FAMILY) is not None
    svc.retry_store_failures()

    assert stores.directory.by_family(_FAMILY) == []
    assert stores.families.get(_FAMILY) is None
    assert _FAMILY in durability.families


def test_a_family_a_pending_request_held_retires_with_that_request() -> None:
    svc, stores, _settled, _deps = _build_serve()
    _adopt(svc)
    svc._admission.raise_claim(
        invocation_id="inv-pending",
        subject=InvocationSubject(kind=InvocationSubjectKind.EXTERNAL, id="key-1"),
        family=_FAMILY,
        profile=PROFILE,
    )
    svc.drain_serve_replica(_SERVE_TASK)
    assert stores.directory.by_family(_FAMILY) == []
    assert stores.families.get(_FAMILY) is not None

    svc.fail_serve("inv-pending", _ServeDelivery(), "serve task stopped")

    assert not stores.live("inv-pending")
    assert stores.families.get(_FAMILY) is None


def test_a_family_left_live_by_a_crash_after_its_replica_moved_retires_on_restart() -> (
    None
):
    svc, stores, _settled, _deps = _build_serve()
    stores.durability = durability = _FamilyArchiveFailsOnce()
    _adopt(svc)
    svc.drain_serve_replica(_SERVE_TASK)
    assert live_field(FAMILY, _FAMILY) in durability.live

    fresh, fresh_stores, _s, _d = _build(durability=durability)
    fresh.rehydrate(asyncio.run(durability.load_snapshot_async()))
    fresh.reattach_replicas(frozenset())
    fresh.recover_retirements()

    assert fresh_stores.families.get(_FAMILY) is None
    assert live_field(FAMILY, _FAMILY) not in durability.live
    assert _FAMILY in durability.families


class _FamilyReplyLost(MemoryResidentDurability):
    lose = True

    def archive_family(self, family: ServiceFamily, field: str) -> None:
        super().archive_family(family, field)
        if self.lose:
            self.lose = False
            raise ConnectionError("reply lost")


def test_a_family_re_adopted_after_a_lost_archive_reply_is_stored_live_again() -> None:
    svc, stores, _settled, _deps = _build_serve()
    stores.durability = durability = _FamilyReplyLost()
    _adopt(svc)
    svc.drain_serve_replica(_SERVE_TASK)
    assert svc._store_retry.pending == [(FAMILY, _FAMILY)]

    _adopt(svc)
    svc.retry_store_failures()

    assert svc._store_retry.pending == []
    snapshot = asyncio.run(durability.load_snapshot_async())
    assert snapshot is not None
    assert _FAMILY in {family.family for family in snapshot.families}
    fresh, fresh_stores, _s, _d = _build(durability=durability)
    fresh.rehydrate(snapshot)
    assert fresh_stores.families.get(_FAMILY) is not None
    assert fresh_stores.directory.by_family(_FAMILY)
    listed = asyncio.run(fresh.list_service_families_async())
    assert [family.family for family in listed].count(_FAMILY) == 1


def test_an_unwritten_replica_change_leaves_a_retired_invocation_retired() -> None:
    svc, stores, _settled, _delivery = _served()
    svc.on_invocation_terminal("inv-1").result()
    [replica] = stores.directory.all()
    replica.report_epoch += 1
    assert not stores.durable

    assert svc.invocations_retired(["inv-1"])


def test_an_unwritten_request_change_reads_nothing_retired() -> None:
    svc, stores, _settled, _delivery = _served()
    svc.on_invocation_terminal("inv-1").result()
    request = stores.invocations.get("inv-1")
    assert request is None
    svc._admission.raise_claim(
        invocation_id="inv-2", subject=_SUBJECT, family="m|chat", profile=PROFILE
    )
    request = stores.invocations.get("inv-2")
    assert request is not None
    stores.invocations.replace(request.model_copy(update={"replayable": False}))

    assert not svc.invocations_retired(["inv-1"])


def test_a_batch_claim_releases_its_whole_credit_at_once() -> None:
    stores = warm_stores(slots=4, durability=MemoryResidentDurability())
    admission = AdmissionController(stores)
    profile = PROFILE.model_copy(update={"batch_size": 3})
    claim = admission.raise_claim(
        invocation_id="inv-1", subject=_SUBJECT, family="fam", profile=profile
    )
    admission.admit(claim, profile, idempotency_key=None)
    assert stores.credit_ledger.held("rpl-1") == 3

    admission.settle_invocation_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    assert stores.credit_ledger.held("rpl-1") == 0
    assert admission.retire("inv-1")
    assert stores.credit_ledger.held("rpl-1") == 0


def _serve_accepted(outage: _Outage):
    svc, stores, _settled, _deps = _build_serve()
    stores.durability = outage
    outage.trip = False
    _adopt(svc)
    delivery = _ServeDelivery()
    asyncio.run(svc._originate_serve(_origination(delivery)))
    [replica] = stores.directory.by_family(_FAMILY)
    return svc, stores, delivery, replica


def test_a_serve_hold_during_an_outage_re_drives_and_settles_once_it_ends() -> None:
    outage = _Outage()
    svc, stores, delivery, replica = _serve_accepted(outage)
    asyncio.run(svc._on_ack(_serve_ack(svc, ResidentBootstrapOutcome.ACKED)))
    outage.down = True

    asyncio.run(
        svc._on_outcome(
            _serve_outcome(svc, ResidentStreamStatus.UNCERTAIN, error="lost")
        )
    )
    assert delivery.redrives == 1
    [claim] = stores.claims.by_invocation("inv-1")
    asyncio.run(svc._originate_serve(_origination(delivery)))

    assert delivery.failed is not None and not delivery.completed
    stored = json.loads(outage.live[live_field(CLAIM, claim.claim_id)])
    assert stored["state"] != ClaimState.TERMINAL
    assert svc._store_retry.pending == [(SETTLEMENT, "inv-1")]
    outage.down = False
    svc.retry_store_failures()

    assert stores.credit_ledger.held(replica.replica_id) == 0
    assert not stores.live("inv-1")
    assert svc._store_retry.pending == []


def test_a_serve_rejection_during_an_outage_releases_its_replica_once_it_ends() -> None:
    outage = _Outage()
    svc, stores, delivery, replica = _serve_accepted(outage)
    released: list[str] = []
    release = svc._admission._on_release

    def heard(replica_id: str) -> None:
        released.append(replica_id)
        release(replica_id)

    svc._admission.set_release_hook(heard)
    svc._lifecycle.drain(replica.replica_id)
    outage.down = True

    asyncio.run(
        svc._on_ack(
            _serve_ack(svc, ResidentBootstrapOutcome.REJECTED, rejection="fence")
        )
    )

    assert delivery.failed is not None and not delivery.completed
    assert released == []
    outage.down = False
    svc.retry_store_failures()

    assert released == [replica.replica_id]
    assert stores.directory.get(replica.replica_id) is None
    assert stored_replica(stores, replica.replica_id).state is ReplicaState.STOPPED
    assert not stores.live("inv-1")


def test_a_workflow_hold_during_an_outage_re_drives_its_boundary() -> None:
    outage = _Outage()
    outage.trip = False
    svc, _stores, _settled, _delivery = _accepted(outage)
    outage.down = True

    asyncio.run(svc._on_outcome(_outcome(svc, ResidentStreamStatus.UNCERTAIN)))

    assert svc._redispatched  # type: ignore[attr-defined]


def test_a_workflow_hold_that_preempts_during_an_outage_re_drives_its_boundary() -> (
    None
):
    outage = _Outage()
    outage.trip = False
    svc, stores, _settled, _delivery = _accepted(outage)
    svc._max_transient_redrives = 1
    [claim] = stores.claims.by_invocation("inv-1")
    outage.down = True

    asyncio.run(svc._on_outcome(_outcome(svc, ResidentStreamStatus.UNCERTAIN)))

    replica = stores.directory.get(claim.replica_id or "")
    assert replica is not None and replica.state is ReplicaState.PREEMPTED
    assert svc._redispatched  # type: ignore[attr-defined]


class _StreamStartFails(MemoryResidentDurability):
    def write(self, changes: Mapping[str, str | None]) -> None:
        if any(v is not None and '"state":"streaming"' in v for v in changes.values()):
            raise ConnectionError("store unavailable")
        super().write(changes)


def test_an_acceptance_whose_stream_start_write_fails_is_still_authorized() -> None:
    svc, stores, _settled, delivery = _accepted(_StreamStartFails())

    assert delivery.frame("resident_authorization")
    [claim] = stores.claims.by_invocation("inv-1")
    assert claim.state is ClaimState.STREAMING


def test_a_family_re_adopted_after_a_lost_archive_reply_survives_a_crash() -> None:
    svc, stores, _settled, _deps = _build_serve()
    stores.durability = durability = _FamilyReplyLost()
    _adopt(svc)
    svc.drain_serve_replica(_SERVE_TASK)
    _adopt(svc)

    fresh, fresh_stores, _s, _d = _build_serve()
    fresh_stores.durability = durability
    fresh.rehydrate(asyncio.run(durability.load_snapshot_async()))
    fresh.reattach_replicas(frozenset({_SERVE_TASK}))
    fresh.recover_retirements()

    assert fresh_stores.families.get(_FAMILY) is not None
    delivery = _ServeDelivery()
    asyncio.run(fresh._originate_serve(_origination(delivery)))
    assert delivery.opened and delivery.failed is None
