"""The ingress edge authenticates, admits, streams, and re-drives one request.

The edge resolves a published alias under the caller's tenant, quota-limits the
principal, injects the raw request into a designated deputy before originating, streams
the teed response to the client, and re-drives an uncertain loss onto a fresh deputy. A
pre-admission rejection raises and consumes no quota beyond its own release.
"""

import asyncio
from typing import Any

from lumid_hooks import PrincipalContext

from server.ingress import (
    AliasCatalog,
    InferenceIngress,
    IngressTerminalStore,
    PrincipalQuota,
)
from server.ingress.service import (
    AliasNotFound,
    IngressResult,
    NoDeputyAvailable,
    TenantNotAuthorized,
)
from server.resident.service import IngressOrigination

_CATALOG = AliasCatalog.from_json("""
    {"aliases": [
        {"alias": "open", "service_ref": "org/model"},
        {"alias": "acme-only", "service_ref": "org/model", "allowed_tenants": ["acme"]}
    ]}
    """)


def _principal(principal_id: str = "p1", org_id: str = "acme") -> PrincipalContext:
    return PrincipalContext(
        principal_id=principal_id,
        org_id=org_id,
        external_id=principal_id,
        principal_type="user",
        scopes=[],
    )


class _FakeControl:
    """Records the injects and originations and drives each attempt's delivery."""

    def __init__(self, behavior) -> None:
        self.injects: list[tuple[str, str, str, str]] = []
        self.originations: list[IngressOrigination] = []
        self._behavior = behavior

    def inject_ingress_request(
        self, worker_id: str, task_id: str, call_correlation: str, request: str
    ) -> bool:
        self.injects.append((worker_id, task_id, call_correlation, request))
        return True

    def originate_ingress(self, origination: IngressOrigination) -> None:
        self.originations.append(origination)
        self._behavior(self, origination)


def _ingress(control, *, quota: int = 4) -> InferenceIngress:
    return InferenceIngress(
        catalog=_CATALOG,
        quota=PrincipalQuota(quota),
        terminals=IngressTerminalStore(),
        control=control,  # type: ignore[arg-type]
        select_worker=lambda: "wkr-1",
    )


async def _chunks(result: IngressResult) -> list[str]:
    return [ev.payload async for ev in result.events() if ev.kind == "chunk"]


def test_unknown_alias_and_unauthorized_tenant_never_originate():
    control = _FakeControl(lambda *_: None)
    ingress = _ingress(control)
    try:
        ingress.submit(_principal(), "missing", "{}")
        raise AssertionError("expected AliasNotFound")
    except AliasNotFound:
        pass
    try:
        ingress.submit(_principal(org_id="other"), "acme-only", "{}")
        raise AssertionError("expected TenantNotAuthorized")
    except TenantNotAuthorized:
        pass
    assert control.originations == []
    # Neither rejection held a quota slot: a fresh authorized request still admits.
    ingress.submit(_principal(), "open", "{}")


def test_submit_injects_then_originates_and_streams_the_completion():
    def behavior(_control, origination: IngressOrigination) -> None:
        origination.delivery.tee("he")
        origination.delivery.tee("llo")
        origination.delivery.complete()

    control = _FakeControl(behavior)
    ingress = _ingress(control)

    async def run() -> None:
        result = ingress.submit(_principal(), "open", '{"messages": []}')
        assert await _chunks(result) == ["he", "llo"]

    asyncio.run(run())
    # The raw request is injected into the deputy before the origination relays.
    assert control.injects and control.injects[0][0] == "wkr-1"
    assert control.injects[0][3] == '{"messages": []}'
    assert control.originations[0].subject.tenant == "acme"
    assert control.originations[0].origin_worker == "wkr-1"


def test_no_deputy_available_releases_the_quota():
    control = _FakeControl(lambda *_: None)
    ingress = InferenceIngress(
        catalog=_CATALOG,
        quota=PrincipalQuota(1),
        terminals=IngressTerminalStore(),
        control=control,  # type: ignore[arg-type]
        select_worker=lambda: None,
    )
    for _ in range(2):
        try:
            ingress.submit(_principal(), "open", "{}")
            raise AssertionError("expected NoDeputyAvailable")
        except NoDeputyAvailable:
            pass  # a released quota lets the retry reach deputy selection again


def test_quota_bounds_in_flight_requests():
    # A behavior that never terminates leaves the request in flight, holding its slot.
    control = _FakeControl(lambda *_: None)
    ingress = _ingress(control, quota=1)
    from server.ingress import QuotaExceeded

    ingress.submit(_principal(), "open", "{}")
    try:
        ingress.submit(_principal(), "open", "{}")
        raise AssertionError("expected QuotaExceeded")
    except QuotaExceeded:
        pass


def test_uncertain_loss_redrives_onto_a_fresh_deputy():
    state = {"attempts": 0}

    def behavior(_control, origination: IngressOrigination) -> None:
        state["attempts"] += 1
        if state["attempts"] == 1:
            origination.delivery.redrive()
        else:
            origination.delivery.tee("ok")
            origination.delivery.complete()

    control = _FakeControl(behavior)
    workers = iter(["wkr-1", "wkr-2"])
    ingress = InferenceIngress(
        catalog=_CATALOG,
        quota=PrincipalQuota(2),
        terminals=IngressTerminalStore(),
        control=control,  # type: ignore[arg-type]
        select_worker=lambda: next(workers, "wkr-2"),
    )

    async def run() -> None:
        result = ingress.submit(_principal(), "open", "{}")
        assert await _chunks(result) == ["ok"]

    asyncio.run(run())
    # The re-drive re-injected the retained request onto the second deputy under the
    # same invocation identity.
    assert [i[0] for i in control.injects] == ["wkr-1", "wkr-2"]
    assert control.injects[1][3] == "{}"
    assert (
        control.originations[0].invocation_id == control.originations[1].invocation_id
    )


def test_post_flush_loss_fails_the_client_without_duplicating_the_prefix():
    # Attempt one flushes a partial frame, then loses; the re-drive must not re-stream
    # onto the same connection (which would duplicate the delivered prefix).
    state = {"attempts": 0}

    def behavior(_control, origination: IngressOrigination) -> None:
        state["attempts"] += 1
        if state["attempts"] == 1:
            origination.delivery.tee("partial")
            origination.delivery.redrive()
        else:
            origination.delivery.tee("RESTARTED")
            origination.delivery.complete()

    control = _FakeControl(behavior)
    ingress = _ingress(control)

    async def run() -> list[Any]:
        result = ingress.submit(_principal(), "open", "{}")
        return [ev async for ev in result.events()]

    events = asyncio.run(run())
    assert (events[0].kind, events[0].payload) == ("chunk", "partial")
    assert events[-1].kind == "error"
    # The re-driven attempt's frames never reach this connection.
    assert all(ev.payload != "RESTARTED" for ev in events)
