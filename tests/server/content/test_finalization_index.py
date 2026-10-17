"""Which content an outcome finalization is bound to, which drive decides it, and
what its binding protects."""

import fakeredis
import pytest

from server.clients.redis import (
    RECLAIM_ROOT_EPOCH_KEY,
    reclaim_deleting_key,
    reclaim_history_key,
    reclaim_pins_key,
    reclaim_protected_key,
)
from server.content import FinalizationIndex
from server.content.finalizations import (
    FinalizationRefused,
    StaleRootEpoch,
)
from shared.content import ContentReference, StoreRefusal, reference_for
from tests.server.redis_helpers import fake_redis_client

_SCOPE = "tenant-a"


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def raw(server: fakeredis.FakeServer) -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(server=server, decode_responses=True)


def _index(server: fakeredis.FakeServer, **kwargs) -> FinalizationIndex:
    return FinalizationIndex(fake_redis_client(server), **kwargs)


def test_a_finalization_binds_and_protects_the_content_it_materialized(
    server, raw
) -> None:
    reference = reference_for(_SCOPE, b"the outcome", media_type="application/json")

    manifest = _index(server).record(_SCOPE, "idm-1", reference)

    assert manifest.content == reference
    assert manifest.idempotency_key == "idm-1"
    assert raw.sismember(reclaim_protected_key(_SCOPE), reference.content_digest)
    assert not raw.exists(reclaim_pins_key(_SCOPE))


def test_a_re_drive_resolves_the_first_materialization_and_protects_nothing(
    server, raw
) -> None:
    index = _index(server)
    first = reference_for(_SCOPE, b"sampled once", media_type="application/json")
    index.record(_SCOPE, "idm-1", first)
    loser = reference_for(_SCOPE, b"sampled twice", media_type="application/json")

    again = index.record(_SCOPE, "idm-1", loser)

    assert again.content == first
    settled = index.find(_SCOPE, "idm-1")
    assert settled is not None and settled.content == first
    assert not raw.sismember(reclaim_protected_key(_SCOPE), loser.content_digest)


def test_a_scope_does_not_see_another_scopes_finalization(server) -> None:
    index = _index(server)
    index.record(_SCOPE, "idm-1", reference_for(_SCOPE, b"mine"))

    assert index.find("tenant-b", "idm-1") is None


def test_content_being_retired_is_refused_and_nothing_is_bound(server, raw) -> None:
    reference = reference_for(_SCOPE, b"doomed")
    raw.hset(reclaim_deleting_key(_SCOPE), reference.content_digest, "{}")
    with pytest.raises(FinalizationRefused) as refused:
        _index(server).record(_SCOPE, "idm-1", reference)
    assert refused.value.refusal is StoreRefusal.RETRY_RETIREMENT
    assert _index(server).find(_SCOPE, "idm-1") is None


@pytest.mark.parametrize(
    ("present", "refusal"),
    [
        (False, StoreRefusal.REMATERIALIZE),
        (None, StoreRefusal.RETRY),
    ],
)
def test_a_retired_identity_binds_only_on_a_verified_backing(
    server, raw, present: bool | None, refusal: StoreRefusal
) -> None:
    reference = reference_for(_SCOPE, b"retired once")
    raw.hset(reclaim_history_key(_SCOPE), reference.content_digest, "4")
    with pytest.raises(FinalizationRefused) as refused:
        _index(server, verify=lambda _: present).record(_SCOPE, "idm-1", reference)
    assert refused.value.refusal is refusal
    assert _index(server).find(_SCOPE, "idm-1") is None


def test_a_verified_backing_binds_a_retired_identity(server, raw) -> None:
    reference = reference_for(_SCOPE, b"retired once")
    raw.hset(reclaim_history_key(_SCOPE), reference.content_digest, "4")
    seen: list[ContentReference] = []

    def verify(ref: ContentReference) -> bool:
        seen.append(ref)
        return True

    manifest = _index(server, verify=verify).record(_SCOPE, "idm-1", reference)
    assert manifest.content == reference and seen == [reference]
    assert raw.sismember(reclaim_protected_key(_SCOPE), reference.content_digest)


def test_a_root_past_its_epoch_records_nothing(server, raw) -> None:
    raw.set(RECLAIM_ROOT_EPOCH_KEY, "2")
    with pytest.raises(StaleRootEpoch):
        _index(server, epoch=lambda: 1).record(
            _SCOPE, "idm-1", reference_for(_SCOPE, b"late")
        )
    assert _index(server).find(_SCOPE, "idm-1") is None
