"""Whether an invocation's resident claims are all terminal, asked from any thread."""

import asyncio
import threading

import pytest

from server.resident import ClaimState, ResidentStores, ServiceClaim
from tests.server.resident.test_observability import _control


def _claim(claim_id: str, invocation_id: str, state: ClaimState) -> ServiceClaim:
    return ServiceClaim(
        claim_id=claim_id,
        invocation_id=invocation_id,
        family="fam",
        admission_epoch=0,
        state=state,
    )


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


@pytest.mark.parametrize(
    ("state", "settled"),
    [
        (ClaimState.TERMINAL, True),
        (ClaimState.PENDING, False),
        (ClaimState.ACCEPTED, False),
        (ClaimState.UNCERTAIN, False),
    ],
)
def test_an_invocation_is_settled_only_once_every_claim_of_it_is_terminal(
    state: ClaimState, settled: bool
) -> None:
    stores = ResidentStores()
    stores.claims.add(_claim("scl-1", "inv-1", ClaimState.TERMINAL))
    stores.claims.add(_claim("scl-2", "inv-1", state))
    control = _control(stores)

    assert control.invocations_settled(["inv-1", "inv-2"], 1.0) is settled
    assert control.invocations_settled(["inv-2"], 1.0)


def test_the_claims_are_read_on_the_loop_that_owns_them() -> None:
    stores = ResidentStores()
    stores.claims.add(_claim("scl-1", "inv-1", ClaimState.ACCEPTED))
    control = _control(stores)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        control.bind_loop(loop)
        assert not control.invocations_settled(["inv-1"], 1.0)
        loop.call_soon_threadsafe(
            lambda: setattr(stores.claims.get("scl-1"), "state", ClaimState.TERMINAL)
        )
        assert control.invocations_settled(["inv-1"], 1.0)

        stalled = threading.Event()
        loop.call_soon_threadsafe(stalled.wait)
        assert not control.invocations_settled(["inv-1"], 0.05)
        stalled.set()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)
        loop.close()
