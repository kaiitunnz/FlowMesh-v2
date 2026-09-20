"""The control plane's decision to authorize one hydration, and what it refuses."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.content import (
    ContentHolderDirectory,
    ContentHydrationAuthority,
    ContentTransferSessions,
    HydrationDenial,
)
from shared.content import ContentHydrationGrant, reference_for

_REFERENCE = reference_for("local", b"prepared", media_type="application/json")
_HELD = (_REFERENCE.authorization_scope, _REFERENCE.content_digest)


class _FakeRedis:
    """Only the hash and expire surface the directory and session record use."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, Any]] = {}

    def hash_set(self, key: str, mapping: dict[str, Any]) -> None:
        self.hashes.setdefault(key, {}).update(mapping)

    def hash_getall(self, key: str) -> dict[str, Any]:
        return dict(self.hashes.get(key, {}))

    def hash_delete(self, key: str, *fields: str) -> None:
        for field in fields:
            self.hashes.get(key, {}).pop(field, None)

    def expire(self, key: str, ttl_sec: int) -> bool:
        return True


class _Workers:
    """Live workers by id, and the control frames the authority relayed to them."""

    def __init__(self, **incarnations: int) -> None:
        self._live = incarnations
        self.frames: list[tuple[str, str, dict[str, Any]]] = []

    def get_worker(self, worker_id: str) -> Any:
        if worker_id not in self._live:
            return None
        return SimpleNamespace(
            id=worker_id,
            node_id=f"nde-{worker_id[-1]}",
            incarnation=self._live[worker_id],
        )

    def retire(self, worker_id: str) -> None:
        self._live.pop(worker_id, None)

    def restart(self, worker_id: str) -> None:
        self._live[worker_id] = self._live[worker_id] + 1

    def publish_mediated_op(self, worker: Any, payload: Any) -> int:
        self.frames.append((worker.id, payload.frame_kind, payload.payload))
        return 0

    def kinds(self) -> list[str]:
        return [kind for _, kind, _ in self.frames]


def _authority(
    workers: _Workers, *, authorizes: bool = True, redis: Any = None
) -> ContentHydrationAuthority:
    fake = redis if redis is not None else _FakeRedis()
    client = cast(Any, SimpleNamespace(sync=fake))
    return ContentHydrationAuthority(
        ContentHolderDirectory(client, record_ttl_sec=300.0),
        cast(Any, workers),
        authorizes=lambda task_id, worker_id, reference: authorizes,
        grant_ttl_sec=60.0,
        sessions=ContentTransferSessions(client, ttl_sec=600.0),
    )


def _granted(workers: _Workers) -> ContentHydrationGrant:
    serve = next(p for _, kind, p in workers.frames if kind == "content_serve_grant")
    return ContentHydrationGrant.model_validate(serve["grant"])


def _denial(workers: _Workers) -> str:
    denied = next(p for _, kind, p in workers.frames if kind == "content_grant_denied")
    return str(denied["reason"])


def test_a_bound_request_grants_both_ends_the_same_grant() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    authority.record_holding("wkr-1", [_HELD])

    authority.authorize("wkr-2", "tsk-1", _REFERENCE)

    assert workers.kinds() == ["content_serve_grant", "content_grant"]
    holder_frame, requester_frame = (payload for _, _, payload in workers.frames)
    assert holder_frame == requester_frame
    grant = _granted(workers)
    assert grant.reference == _REFERENCE
    assert (grant.holder_id, grant.holder_generation) == ("wkr-1", 3)
    assert grant.requester_subject == "wkr-2"


def test_the_transfer_record_routes_between_the_two_ends() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    redis = _FakeRedis()
    authority = _authority(workers, redis=redis)
    authority.record_holding("wkr-1", [_HELD])

    authority.authorize("wkr-2", "tsk-1", _REFERENCE)

    grant = _granted(workers)
    record = redis.hashes[f"ct:sess:{grant.transfer_session_id}"]
    assert record["origin_worker"] == "wkr-2"
    assert record["target_worker"] == "wkr-1"
    assert _REFERENCE.content_digest not in str(record)


def test_a_request_no_binding_authorizes_is_refused() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers, authorizes=False)
    authority.record_holding("wkr-1", [_HELD])

    authority.authorize("wkr-2", "tsk-1", _REFERENCE)

    assert workers.kinds() == ["content_grant_denied"]
    assert _denial(workers) == HydrationDenial.NO_BINDING


def test_an_object_no_holder_reported_is_refused_as_untracked() -> None:
    workers = _Workers(**{"wkr-2": 5})
    authority = _authority(workers)

    authority.authorize("wkr-2", "tsk-1", _REFERENCE)

    assert _denial(workers) == HydrationDenial.NOT_TRACKED


def test_a_holder_that_is_gone_is_refused_rather_than_granted() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    authority.record_holding("wkr-1", [_HELD])
    workers.retire("wkr-1")

    authority.authorize("wkr-2", "tsk-1", _REFERENCE)

    assert _denial(workers) == HydrationDenial.HOLDER_UNAVAILABLE


def test_a_holder_that_came_back_as_another_incarnation_is_refused() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    authority.record_holding("wkr-1", [_HELD])
    workers.restart("wkr-1")

    authority.authorize("wkr-2", "tsk-1", _REFERENCE)

    assert _denial(workers) == HydrationDenial.HOLDER_UNAVAILABLE


def test_a_worker_is_never_sent_to_itself_for_an_object() -> None:
    workers = _Workers(**{"wkr-1": 3})
    authority = _authority(workers)
    authority.record_holding("wkr-1", [_HELD])

    authority.authorize("wkr-1", "tsk-1", _REFERENCE)

    assert _denial(workers) == HydrationDenial.HOLDER_UNAVAILABLE


def test_a_scope_is_part_of_what_a_holder_reported() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    authority.record_holding("wkr-1", [_HELD])

    elsewhere = _REFERENCE.model_copy(update={"authorization_scope": "other"})
    authority.authorize("wkr-2", "tsk-1", elsewhere)

    assert _denial(workers) == HydrationDenial.NOT_TRACKED


def test_an_unknown_requester_is_answered_with_nothing() -> None:
    workers = _Workers(**{"wkr-1": 3})
    authority = _authority(workers)
    authority.record_holding("wkr-1", [_HELD])

    authority.authorize("wkr-9", "tsk-1", _REFERENCE)

    assert workers.frames == []


@pytest.mark.parametrize("field", ["grant_id", "transfer_session_id"])
def test_every_grant_is_freshly_minted(field: str) -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    authority.record_holding("wkr-1", [_HELD])

    authority.authorize("wkr-2", "tsk-1", _REFERENCE)
    first = _granted(workers)
    workers.frames.clear()
    authority.authorize("wkr-2", "tsk-1", _REFERENCE)
    second = _granted(workers)

    assert getattr(first, field) != getattr(second, field)


def test_a_grant_carries_no_service_admission_identity() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    authority.record_holding("wkr-1", [_HELD])

    authority.authorize("wkr-2", "tsk-1", _REFERENCE)

    fields = set(ContentHydrationGrant.model_fields)
    assert not fields & {"idempotency_key", "invocation_id", "claim_id"}
    assert "idm-" not in str(_granted(workers).model_dump())
