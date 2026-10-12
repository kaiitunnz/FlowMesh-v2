"""Whether an invocation has settled, asked from any thread: only one retired into its
durable record, or one that never raised a claim, answers settled, and only from a
complete lookup."""

import asyncio
import threading
from collections.abc import Sequence

from server.resident import (
    AdmissionController,
    ClaimState,
    ClaimTerminalReason,
    InvocationSubject,
    InvocationSubjectKind,
    ResidentStores,
    RetiredInvocation,
    ServiceClaim,
)
from server.resident.durability import MemoryResidentDurability
from tests.server.resident._helpers import PROFILE, warm_stores
from tests.server.resident.test_observability import _control

_SUBJECT = InvocationSubject(kind=InvocationSubjectKind.WORKFLOW, id="wfl-1")


def _claim(claim_id: str, invocation_id: str, state: ClaimState) -> ServiceClaim:
    return ServiceClaim(
        claim_id=claim_id,
        invocation_id=invocation_id,
        family="fam",
        admission_epoch=0,
        state=state,
    )


def _raised(stores: ResidentStores, *invocation_ids: str) -> AdmissionController:
    admission = AdmissionController(stores)
    for invocation_id in invocation_ids:
        claim = admission.raise_claim(
            invocation_id=invocation_id, subject=_SUBJECT, family="fam", profile=PROFILE
        )
        admission.admit(claim, PROFILE, idempotency_key=None)
    stores.flush()
    return admission


def _retired(admission: AdmissionController, invocation_id: str) -> None:
    admission.settle_invocation_terminal(invocation_id, ClaimTerminalReason.COMPLETED)
    assert admission.retire(invocation_id)


def test_claims_are_found_by_the_invocation_that_raised_them() -> None:
    stores = ResidentStores()
    for claim in (
        _claim("scl-1", "inv-1", ClaimState.PENDING),
        _claim("scl-2", "inv-2", ClaimState.TERMINAL),
        _claim("scl-3", "inv-1", ClaimState.TERMINAL),
    ):
        stores.claims.add(claim)

    assert [c.claim_id for c in stores.claims.by_invocation("inv-1")] == [
        "scl-1",
        "scl-3",
    ]
    assert stores.claims.by_invocation("inv-9") == []


def test_an_invocation_settles_once_it_retires_and_never_while_it_is_live() -> None:
    stores = warm_stores()
    admission = _raised(stores, "inv-1")
    control = _control(stores)

    assert not control.invocations_settled(["inv-1"], 1.0)
    admission.settle_invocation_terminal("inv-1", ClaimTerminalReason.COMPLETED)
    # Terminal and consumed, but still in the working set.
    assert not control.invocations_settled(["inv-1"], 1.0)
    assert admission.retire("inv-1")
    assert control.invocations_settled(["inv-1"], 1.0)


def test_an_invocation_that_never_raised_a_claim_is_settled() -> None:
    stores = warm_stores()
    stores.flush()
    control = _control(stores)

    assert control.invocations_settled(["inv-9"], 1.0)
    assert control.invocations_settled([], 1.0)


def test_a_terminal_claim_whose_terminal_was_never_consumed_is_unsettled() -> None:
    stores = warm_stores()
    admission = _raised(stores, "inv-1")
    [claim] = stores.claims.by_invocation("inv-1")
    admission.on_enqueue_failed(claim)

    assert claim.state is ClaimState.TERMINAL
    assert not _control(stores).invocations_settled(["inv-1"], 1.0)


def test_an_incomplete_load_reads_nothing_as_settled() -> None:
    stores = warm_stores()
    stores.flush()
    stores.complete = False

    assert not _control(stores).invocations_settled(["inv-9"], 1.0)


def test_a_claim_change_not_yet_durable_reads_nothing_as_settled() -> None:
    stores = warm_stores()
    admission = _raised(stores, "inv-1", "inv-2")
    _retired(admission, "inv-1")
    [claim] = stores.claims.by_invocation("inv-2")
    claim.updated_at = "2026-01-01T00:00:00Z"

    assert not _control(stores).invocations_settled(["inv-1"], 1.0)


def test_a_replica_change_not_yet_durable_leaves_the_answer_alone() -> None:
    stores = warm_stores()
    _retired(_raised(stores, "inv-1"), "inv-1")
    replica = stores.directory.get("rpl-1")
    assert replica is not None
    replica.last_active_at = "2026-01-01T00:00:00Z"

    assert not stores.durable
    assert _control(stores).invocations_settled(["inv-1"], 1.0)


class _Unreadable(MemoryResidentDurability):
    def retired_invocations(
        self, invocation_ids: Sequence[str]
    ) -> list[RetiredInvocation | None]:
        raise ConnectionError("store unavailable")


def test_an_unreadable_store_reads_nothing_as_settled() -> None:
    stores = ResidentStores(_Unreadable())

    assert not _control(stores).invocations_settled(["inv-9"], 1.0)


def test_a_record_naming_another_invocation_is_not_a_settled_answer() -> None:
    durability = MemoryResidentDurability()
    durability.retired["inv-1"] = RetiredInvocation(
        invocation_id="inv-2",
        subject=_SUBJECT,
        family="fam",
        epoch_high=0,
        terminal_reason=ClaimTerminalReason.COMPLETED,
    ).model_dump_json()

    assert not _control(ResidentStores(durability)).invocations_settled(["inv-1"], 1.0)


def test_a_live_successor_wins_over_its_predecessors_record() -> None:
    stores = warm_stores()
    admission = _raised(stores, "inv-1")
    _retired(admission, "inv-1")
    successor = admission.raise_claim(
        invocation_id="inv-1", subject=_SUBJECT, family="fam", profile=PROFILE
    )

    assert successor.admission_epoch == 1
    assert not _control(stores).invocations_settled(["inv-1"], 1.0)


def test_the_answer_is_read_on_the_loop_that_owns_the_stores() -> None:
    stores = warm_stores()
    admission = _raised(stores, "inv-1")
    control = _control(stores)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        control.bind_loop(loop)
        assert not control.invocations_settled(["inv-1"], 1.0)
        loop.call_soon_threadsafe(lambda: _retired(admission, "inv-1"))
        assert control.invocations_settled(["inv-1"], 1.0)

        stalled = threading.Event()
        loop.call_soon_threadsafe(stalled.wait)
        assert not control.invocations_settled(["inv-1"], 0.05)
        stalled.set()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)
        loop.close()
