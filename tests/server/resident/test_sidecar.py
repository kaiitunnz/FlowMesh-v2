"""The resident-facing claim gate validates fences and rejects mismatches.

A bootstrap opens a session only under a fence naming this replica incarnation and
listener generation; the authorized stream is admitted only under a route authorization
that continues that session's subject, claim, invocation, and request identity. Every
mismatch is a distinct authorization rejection, never a path failure.
"""

from typing import Any

from server.resident.sidecar import (
    GateRejection,
    SidecarClaimGate,
    TrafficClass,
)
from server.resident.state import AdmissionHandoff, RouteAuthorization


def _gate(clock="2026-01-01T00:00:00Z") -> SidecarClaimGate:
    return SidecarClaimGate(
        replica_id="rpl-1",
        incarnation=3,
        listener_generation=2,
        clock=lambda: clock,
    )


def _handoff(**overrides) -> AdmissionHandoff:
    base: dict[str, Any] = dict(
        token="hnd-x",
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family="fam",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=3,
        listener_generation=2,
        expires_at="2026-01-01T00:05:00Z",
    )
    base.update(overrides)
    return AdmissionHandoff(**base)


def _auth(**overrides) -> RouteAuthorization:
    base: dict[str, Any] = dict(
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family="fam",
        operation="inference",
        admission_epoch=0,
        route_auth_epoch=1,
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=3,
        listener_generation=2,
        expires_at="2026-01-01T00:05:00Z",
    )
    base.update(overrides)
    return RouteAuthorization(**base)


def test_bootstrap_admitted_on_matching_fence() -> None:
    assert _gate().check_bootstrap(_handoff()).admitted


def test_bootstrap_rejections() -> None:
    gate = _gate()
    assert gate.check_bootstrap(_handoff(replica_id="rpl-2")).rejection is (
        GateRejection.WRONG_REPLICA
    )
    assert gate.check_bootstrap(_handoff(incarnation=2)).rejection is (
        GateRejection.WRONG_INCARNATION
    )
    assert gate.check_bootstrap(_handoff(listener_generation=1)).rejection is (
        GateRejection.STALE_LISTENER
    )
    assert (
        gate.check_bootstrap(_handoff(expires_at="2025-12-31T23:59:59Z")).rejection
        is GateRejection.EXPIRED
    )
    # An unexpired fence with no explicit expiry is admitted.
    assert gate.check_bootstrap(_handoff(expires_at=None)).admitted


def test_stream_admitted_under_a_continuing_authorization() -> None:
    gate = _gate()
    handoff = _handoff()
    session = gate.session_for(handoff)
    assert gate.check_stream(_auth(), session).admitted


def test_stream_requires_a_session() -> None:
    assert _gate().check_stream(_auth(), None).rejection is GateRejection.NO_SESSION


def test_stream_rejects_a_fence_that_does_not_continue_the_session() -> None:
    gate = _gate()
    session = gate.session_for(_handoff())
    cases: list[tuple[dict[str, Any], GateRejection]] = [
        ({"incarnation": 2}, GateRejection.WRONG_INCARNATION),
        ({"listener_generation": 1}, GateRejection.STALE_LISTENER),
        ({"expires_at": "2025-12-31T23:59:59Z"}, GateRejection.EXPIRED),
        ({"claim_id": "scl-2"}, GateRejection.WRONG_CLAIM),
        ({"invocation_id": "inv-2"}, GateRejection.WRONG_INVOCATION),
        ({"idempotency_key": "idm-2"}, GateRejection.WRONG_REQUEST),
        ({"tenant": "t2"}, GateRejection.WRONG_SUBJECT),
        ({"origin_id": "rog-2"}, GateRejection.WRONG_SUBJECT),
    ]
    for overrides, expected in cases:
        assert gate.check_stream(_auth(**overrides), session).rejection is expected


def test_load_evidence_is_claim_tagged() -> None:
    ev = _gate().load_evidence(_handoff(), "request")
    assert ev.claim_id == "scl-1" and ev.invocation_id == "inv-1"
    assert ev.replica_id == "rpl-1" and ev.incarnation == 3
    assert ev.traffic_class is TrafficClass.SERVICE
