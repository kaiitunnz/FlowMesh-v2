"""Resident-capacity control admits an ingress subject through the same gate.

An authenticated external principal with no workflow activation funnels through the same
Admission controller, two-phase relay, and terminal fences as a workflow consumer. Its
outcome routes through the ingress delivery handle — teed frames, a fenced ingress
terminal that releases the credit, and an uncertain re-drive — and it fabricates no DS
workflow state. A workflow terminal and an ingress terminal, keyed by distinct
invocation ids, never overwrite one another.
"""

import asyncio
from typing import Any

from server.resident import AdmissionProfile, ClaimState, ClaimTerminalReason
from server.resident.service import IngressOrigination
from server.resident.state import InvocationSubject, InvocationSubjectKind
from shared.outcome import OutcomeManifest
from shared.resident.reports import (
    ResidentBootstrapOutcome,
    ResidentStreamChunk,
    ResidentStreamStatus,
)
from tests.server.resident.test_service import _ack, _build, _dependency, _env, _outcome


class _FakeDelivery:
    """Records the settle and re-drive routing an ingress attempt drives through."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.terminals: list[tuple[Any, str | None]] = []
        self.completed = False
        self.failures: list[str] = []
        self.redrives = 0

    def tee(self, payload: str) -> None:
        self.chunks.append(payload)

    def record_terminal(self, reason: Any, detail: str | None) -> None:
        self.terminals.append((reason, detail))

    def complete(self) -> None:
        self.completed = True

    def fail(self, detail: str) -> None:
        self.failures.append(detail)

    def redrive(self) -> None:
        self.redrives += 1


def _ingress_request(
    delivery: _FakeDelivery,
    *,
    invocation_id: str = "inv-ing",
    tenant: str = "acme",
    origin_worker: str = "wkr-origin",
) -> IngressOrigination:
    dependency = _dependency()
    return IngressOrigination(
        invocation_id=invocation_id,
        idempotency_key=f"idm-{invocation_id}",
        task_id=invocation_id,
        call_correlation="ingress",
        subject=InvocationSubject(
            kind=InvocationSubjectKind.INGRESS, id="prn-1", tenant=tenant
        ),
        dependency=dependency,
        profile=AdmissionProfile(
            engine_batch_key=dependency.engine_batch_key, tenant=tenant
        ),
        origin_worker=origin_worker,
        delivery=delivery,  # type: ignore[arg-type]
    )


def test_ingress_originates_with_an_ingress_subject_and_no_workflow_state():
    svc, stores, settled, delivery_relay = _build()
    delivery = _FakeDelivery()
    asyncio.run(svc._originate_ingress(_ingress_request(delivery)))

    # The same admission gate raised a claim, tagged with an external-principal subject.
    request = stores.invocations.get("inv-ing")
    assert request is not None
    assert request.subject.kind is InvocationSubjectKind.INGRESS
    assert request.subject.id == "prn-1" and request.subject.tenant == "acme"

    # The handoff relayed to the designated deputy, marked for the response tee, and
    # carrying the tenant subject the target claim gate enforces.
    handoff = delivery_relay.frame("resident_handoff")
    assert handoff["tee"] is True
    assert handoff["handoff"]["tenant"] == "acme"

    # No DS workflow state was fabricated: the workflow settle/re-drive callbacks the
    # engine drives are never invoked for an ingress subject.
    assert settled == []
    assert svc._redispatched == []  # type: ignore[attr-defined]


def test_success_records_the_terminal_releases_credit_and_completes():
    svc, stores, settled, _relay = _build()
    delivery = _FakeDelivery()
    asyncio.run(svc._originate_ingress(_ingress_request(delivery)))
    asyncio.run(
        svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED, invocation_id="inv-ing"))
    )
    manifest = OutcomeManifest(
        content_digest="sha", size_bytes=2, media_type="text/plain"
    )
    asyncio.run(
        svc._on_outcome(
            _outcome(
                svc,
                ResidentStreamStatus.SUCCESS,
                invocation_id="inv-ing",
                manifest=manifest,
            )
        )
    )

    claim = stores.claims.by_invocation("inv-ing")[0]
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(claim.replica_id) == 0
    assert delivery.terminals and delivery.completed is True
    # The ingress settles itself: the workflow DS settle path is never used.
    assert settled == []


def test_definite_failure_fails_the_client_and_releases():
    svc, stores, _settled, _relay = _build()
    delivery = _FakeDelivery()
    asyncio.run(svc._originate_ingress(_ingress_request(delivery)))
    asyncio.run(
        svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED, invocation_id="inv-ing"))
    )
    asyncio.run(
        svc._on_outcome(
            _outcome(
                svc,
                ResidentStreamStatus.DEFINITE_FAILURE,
                invocation_id="inv-ing",
                error="engine refused",
            )
        )
    )
    claim = stores.claims.by_invocation("inv-ing")[0]
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(claim.replica_id) == 0
    assert delivery.failures and "engine refused" in delivery.failures[-1]


def test_uncertain_outcome_redrives_the_ingress_delivery():
    svc, stores, _settled, _relay = _build()
    delivery = _FakeDelivery()
    asyncio.run(svc._originate_ingress(_ingress_request(delivery)))
    asyncio.run(
        svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED, invocation_id="inv-ing"))
    )
    asyncio.run(
        svc._on_outcome(
            _outcome(
                svc,
                ResidentStreamStatus.UNCERTAIN,
                invocation_id="inv-ing",
                error="stream lost",
            )
        )
    )
    claim = stores.claims.by_invocation("inv-ing")[0]
    assert claim.state is ClaimState.UNCERTAIN
    assert stores.credit_ledger.held(claim.replica_id) == 1  # held, not released
    # An ingress loss re-drives through the edge, not the workflow redispatch callback.
    assert delivery.redrives == 1
    assert svc._redispatched == []  # type: ignore[attr-defined]


def test_teed_chunk_routes_to_the_ingress_delivery():
    svc, _stores, _settled, _relay = _build()
    delivery = _FakeDelivery()
    asyncio.run(svc._originate_ingress(_ingress_request(delivery)))
    session_id = svc._attempts["inv-ing"].session_id

    svc._tee_chunk(
        ResidentStreamChunk(
            invocation_id="inv-ing", session_id=session_id, payload="tok"
        )
    )
    # A chunk for a stale session is ignored.
    svc._tee_chunk(
        ResidentStreamChunk(
            invocation_id="inv-ing", session_id="rly-stale", payload="drop"
        )
    )
    assert delivery.chunks == ["tok"]


def test_redrive_give_up_terminalizes_the_held_claim_and_releases_credit():
    # H1: when a re-drive can find no deputy, the give-up must release the held credit
    # through the fenced path — an ingress claim has no DS terminal to save it.
    svc, stores, _settled, _relay = _build()
    delivery = _FakeDelivery()
    asyncio.run(svc._originate_ingress(_ingress_request(delivery)))
    asyncio.run(
        svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED, invocation_id="inv-ing"))
    )
    asyncio.run(
        svc._on_outcome(
            _outcome(
                svc,
                ResidentStreamStatus.UNCERTAIN,
                invocation_id="inv-ing",
                error="stream lost",
            )
        )
    )
    claim = stores.claims.by_invocation("inv-ing")[0]
    assert claim.state is ClaimState.UNCERTAIN  # credit held, pending the re-drive
    assert stores.credit_ledger.held(claim.replica_id) == 1

    # The edge's re-drive gave up (no deputy): it terminalizes through the fenced path.
    svc.fail_ingress("inv-ing", delivery, "no worker to re-drive")  # type: ignore[arg-type]
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(claim.replica_id) == 0
    assert delivery.failures  # the client is failed after the credit releases


def test_reconcile_terminal_settles_a_rehydrated_uncertain_claim():
    # M1: a crash between the terminal-fact write and the claim release leaves the claim
    # UNCERTAIN; replaying the recorded terminal on startup settles it.
    svc, stores, _settled, _relay = _build()
    delivery = _FakeDelivery()
    asyncio.run(svc._originate_ingress(_ingress_request(delivery)))
    asyncio.run(
        svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED, invocation_id="inv-ing"))
    )
    asyncio.run(
        svc._on_outcome(
            _outcome(
                svc,
                ResidentStreamStatus.UNCERTAIN,
                invocation_id="inv-ing",
                error="stream lost",
            )
        )
    )
    claim = stores.claims.by_invocation("inv-ing")[0]
    assert claim.state is ClaimState.UNCERTAIN

    # Startup replays the recorded ingress terminal through the FSM (no live client).
    svc.reconcile_ingress_terminal("inv-ing", ClaimTerminalReason.COMPLETED)
    assert claim.state is ClaimState.TERMINAL
    assert stores.credit_ledger.held(claim.replica_id) == 0


def test_a_redrive_reaps_every_deputy_it_touched():
    # M2: a re-drive that moves to a fresh deputy injected the request into both; the
    # terminal reap must reach every deputy, not only the last.
    svc, stores, _settled, delivery_relay = _build(deliver=True)
    delivery = _FakeDelivery()
    asyncio.run(
        svc._originate_ingress(_ingress_request(delivery, origin_worker="wkr-a"))
    )
    # A re-drive of the same invocation lands on a fresh deputy under the held claim.
    asyncio.run(
        svc._originate_ingress(_ingress_request(delivery, origin_worker="wkr-b"))
    )
    asyncio.run(
        svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED, invocation_id="inv-ing"))
    )
    manifest = OutcomeManifest(
        content_digest="sha", size_bytes=2, media_type="text/plain"
    )
    asyncio.run(
        svc._on_outcome(
            _outcome(
                svc,
                ResidentStreamStatus.SUCCESS,
                invocation_id="inv-ing",
                manifest=manifest,
            )
        )
    )
    reaped = {
        worker
        for worker, kind, _payload in delivery_relay.relays
        if kind == "resident_reap"
    }
    assert reaped == {"wkr-a", "wkr-b"}


def test_workflow_and_ingress_terminals_do_not_overwrite():
    svc, stores, settled, _relay = _build()
    delivery = _FakeDelivery()
    # A workflow invocation and an ingress invocation admit independently.
    asyncio.run(svc._originate(_env(invocation_id="inv-wf")))
    asyncio.run(
        svc._originate_ingress(_ingress_request(delivery, invocation_id="inv-ing"))
    )

    wf_claim = stores.claims.by_invocation("inv-wf")[0]
    ing_claim = stores.claims.by_invocation("inv-ing")[0]

    # Settling the workflow terminal releases only its own claim; the ingress claim,
    # keyed by a distinct invocation id, is untouched.
    svc.on_invocation_terminal("inv-wf")
    assert wf_claim.state is ClaimState.TERMINAL
    assert ing_claim.state is not ClaimState.TERMINAL
    assert delivery.completed is False

    # The ingress terminal then settles only the ingress claim.
    asyncio.run(
        svc._on_ack(_ack(svc, ResidentBootstrapOutcome.ACKED, invocation_id="inv-ing"))
    )
    manifest = OutcomeManifest(
        content_digest="sha", size_bytes=2, media_type="text/plain"
    )
    asyncio.run(
        svc._on_outcome(
            _outcome(
                svc,
                ResidentStreamStatus.SUCCESS,
                invocation_id="inv-ing",
                manifest=manifest,
            )
        )
    )
    assert ing_claim.state is ClaimState.TERMINAL and delivery.completed is True
