"""The holder's grant gate: only a pre-delivered, exact, unconsumed grant admits."""

import time

import pytest

from shared.content import (
    ContentHydrationGrant,
    ContentOperation,
    GrantRejection,
    HolderGrantGate,
    reference_for,
)
from shared.utils.ids import PREFIX_HYDRATION_GRANT, new_hydration_grant_id


def _grant(**overrides) -> ContentHydrationGrant:
    fields = {
        "grant_id": new_hydration_grant_id(),
        "reference": reference_for("local", b"body", media_type="application/json"),
        "requester_subject": "wkr-2",
        "requester_origin_id": "rog-2",
        "holder_id": "wkr-1",
        "holder_generation": 3,
        "transfer_session_id": "rly-1",
        "expires_at_epoch": time.time() + 30,
    }
    return ContentHydrationGrant(**{**fields, **overrides})


def _gate() -> HolderGrantGate:
    return HolderGrantGate(holder_id="wkr-1", generation=3)


def test_a_grant_id_is_unguessable_and_prefixed() -> None:
    first, second = new_hydration_grant_id(), new_hydration_grant_id()
    assert first.startswith(f"{PREFIX_HYDRATION_GRANT}-") and first != second


def test_a_pre_delivered_grant_admits_once() -> None:
    gate, grant = _gate(), _grant()
    gate.accept(grant)
    assert gate.admit(grant) is None
    assert gate.admit(grant) is GrantRejection.CONSUMED


def test_a_grant_the_holder_was_never_handed_is_refused() -> None:
    assert _gate().admit(_grant()) is GrantRejection.UNKNOWN_GRANT


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference", reference_for("local", b"other", media_type="application/json")),
        ("requester_subject", "wkr-9"),
        ("holder_generation", 4),
        ("transfer_session_id", "rly-9"),
        ("expires_at_epoch", time.time() + 3000),
    ],
)
def test_a_grant_altered_after_minting_is_refused(field, value) -> None:
    gate, grant = _gate(), _grant()
    gate.accept(grant)
    assert (
        gate.admit(grant.model_copy(update={field: value}))
        is GrantRejection.ALTERED_GRANT
    )


def test_a_grant_for_another_object_does_not_open_this_one() -> None:
    gate = _gate()
    granted = _grant()
    gate.accept(granted)
    other = _grant(reference=reference_for("local", b"secret"))
    assert gate.admit(other) is GrantRejection.UNKNOWN_GRANT


def test_an_expired_grant_is_refused() -> None:
    gate = _gate()
    grant = _grant(expires_at_epoch=time.time() - 1)
    gate.accept(grant)
    assert gate.admit(grant) is GrantRejection.EXPIRED


def test_a_grant_for_a_superseded_incarnation_is_refused() -> None:
    gate = HolderGrantGate(holder_id="wkr-1", generation=4)
    grant = _grant()
    gate.accept(grant)
    assert gate.admit(grant) is GrantRejection.STALE_GENERATION


def test_a_grant_for_another_holder_is_refused() -> None:
    gate = HolderGrantGate(holder_id="wkr-7", generation=3)
    grant = _grant()
    gate.accept(grant)
    assert gate.admit(grant) is GrantRejection.WRONG_HOLDER


def test_an_unreadable_grant_version_is_refused() -> None:
    gate, grant = _gate(), _grant(version=99)
    gate.accept(grant)
    assert gate.admit(grant) is GrantRejection.UNSUPPORTED_VERSION


def test_a_grant_carries_no_location_or_credential() -> None:
    fields = set(ContentHydrationGrant.model_fields)
    assert not fields & {"url", "endpoint", "path", "token", "credential", "api_key"}
    assert ContentOperation.HYDRATE is _grant().operation
