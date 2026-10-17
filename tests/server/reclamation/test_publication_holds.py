"""What a publication names, and how its objects are held, refused and released."""

from collections.abc import Iterator, Sequence
from typing import Any

import fakeredis
import pytest
import redis.exceptions

from server.clients.redis import (
    RedisClient,
    reclaim_deleting_key,
    reclaim_history_key,
    reclaim_pins_key,
    reclaim_unexpanded_key,
)
from server.reclamation.pins import CHUNK, Acquisition, Nonces, PinStore
from server.reclamation.publications import (
    VERIFY_ROUNDS,
    Assertion,
    PublicationHolds,
    assertion_of,
)
from server.reclamation.trees import TreeExpander, TreeStatus
from shared.content import ContentReference, StoreRefusal, reference_for
from shared.content.tree import TREE_MEDIA_TYPE, TreeManifest, TreeMember
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from tests.server.redis_helpers import fake_redis_client

_SCOPE = "org-a"


def _ref(data: bytes, scope: str = _SCOPE) -> ContentReference:
    return reference_for(scope, data)


def _file(name: str, reference: ContentReference) -> dict[str, Any]:
    return ArtifactBinding(
        name=name,
        kind=ArtifactKind.FILE,
        reference=reference,
        size_bytes=reference.size_bytes,
    ).model_dump(mode="json")


def _tree(name: str, manifest: ContentReference, revision: str = "final") -> Any:
    return ArtifactBinding(
        name=name,
        revision=revision,
        kind=ArtifactKind.TREE,
        reference=manifest,
        size_bytes=10,
        files=2,
    ).model_dump(mode="json")


RESULT, CHILD, FILE, MEMBER, PREPARED = (
    _ref(b"result"),
    _ref(b"child"),
    _ref(b"file"),
    _ref(b"member"),
    _ref(b"prepared"),
)
_MANIFEST_BYTES = TreeManifest.of(
    [TreeMember("weights.bin", MEMBER.content_digest, MEMBER.size_bytes)]
).encode()
MANIFEST = _ref(_MANIFEST_BYTES).model_copy(update={"media_type": TREE_MEDIA_TYPE})
_STORED = {MANIFEST.content_digest: _MANIFEST_BYTES}


def _open(reference: ContentReference) -> Iterator[bytes]:
    yield _STORED[reference.content_digest]


def _stored_size(reference: ContentReference) -> int | None:
    data = _STORED.get(reference.content_digest)
    return None if data is None else len(data)


def test_a_success_names_its_result_children_files_and_tree_manifests() -> None:
    payload = {
        "result_reference": RESULT.model_dump(mode="json"),
        "child_result_references": {"tsk-2": CHILD.model_dump(mode="json")},
        "artifacts": [_file("a.bin", FILE), _tree("model", MANIFEST)],
    }
    assertion = assertion_of("TASK_SUCCEEDED", payload)
    assert assertion.bind[_SCOPE] == {
        RESULT.content_digest,
        CHILD.content_digest,
        FILE.content_digest,
        MANIFEST.content_digest,
    }
    assert not assertion.pending
    assert assertion.manifests[_SCOPE] == {MANIFEST.content_digest}


def test_a_staged_file_is_held_pending_and_its_tree_manifest_bound() -> None:
    payload = {
        "result_reference": RESULT.model_dump(mode="json"),
        "artifacts": [_file("a.bin", FILE), _tree("model", MANIFEST)],
        "artifacts_pending": True,
    }
    assertion = assertion_of("TASK_SUCCEEDED", payload)
    assert assertion.pending[_SCOPE] == {FILE.content_digest}
    assert assertion.bind[_SCOPE] == {RESULT.content_digest, MANIFEST.content_digest}


def test_a_checkpoint_and_a_prepared_request_name_their_one_object() -> None:
    checkpoint = assertion_of(
        "TASK_UPDATE", {"artifact_published": [_tree("ckpt", MANIFEST, "step-4")]}
    )
    assert checkpoint.bind[_SCOPE] == {MANIFEST.content_digest}
    prepared = assertion_of(
        "TASK_SUCCEEDED",
        {
            "input_materialization": {"reference": PREPARED.model_dump(mode="json")},
            "result_reference": RESULT.model_dump(mode="json"),
        },
    )
    assert prepared.bind[_SCOPE] == {PREPARED.content_digest}


class _Backed:
    def __init__(self, answer: bool | None) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def __call__(self, reference: ContentReference) -> bool | None:
        self.asked.append(reference.content_digest)
        return self.answer


@pytest.fixture
def raw() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)


def _client(raw: fakeredis.FakeRedis) -> RedisClient:
    return fake_redis_client(raw.connection_pool.connection_kwargs["server"])


def _holds(
    raw: fakeredis.FakeRedis,
    backed: _Backed | None = None,
    pins: PinStore | None = None,
) -> PublicationHolds:
    client = _client(raw)
    nonces = Nonces(7)
    return PublicationHolds(
        pins or PinStore(client),
        lambda: nonces,
        backed or _Backed(True),
        TreeExpander(client, _open, _stored_size),
    )


_LOCATOR = {"task": "tsk-1", "dispatch": "dsp-1", "kind": "result", "key": "tsk-1"}
_SUCCESS = {
    "result_reference": RESULT.model_dump(mode="json"),
    "artifacts": [_file("a.bin", FILE), _tree("model", MANIFEST)],
}


def test_holding_a_success_holds_everything_it_names_until_released(raw) -> None:
    holds = _holds(raw)
    preparation = holds.acquire(assertion_of("TASK_SUCCEEDED", _SUCCESS), _LOCATOR)
    assert preparation.refusal is None and preparation.holds is not None
    pinned = raw.hgetall(reclaim_pins_key(_SCOPE))
    assert set(pinned) == {
        RESULT.content_digest,
        FILE.content_digest,
        MANIFEST.content_digest,
        MEMBER.content_digest,
    }
    assert raw.smembers(reclaim_unexpanded_key(_SCOPE)) == {MANIFEST.content_digest}
    holds.release(preparation.holds)
    assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}


def test_an_object_being_retired_refuses_and_holds_nothing(raw) -> None:
    raw.hset(reclaim_deleting_key(_SCOPE), FILE.content_digest, "{}")
    preparation = _holds(raw).acquire(
        assertion_of("TASK_SUCCEEDED", _SUCCESS), _LOCATOR
    )
    assert preparation.refusal is StoreRefusal.RETRY_RETIREMENT
    assert preparation.digests == (FILE.content_digest,)
    assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}


def test_a_staged_file_being_retired_is_still_accepted(raw) -> None:
    raw.hset(reclaim_deleting_key(_SCOPE), FILE.content_digest, "{}")
    preparation = _holds(raw).acquire(
        assertion_of("TASK_SUCCEEDED", {**_SUCCESS, "artifacts_pending": True}),
        _LOCATOR,
    )
    assert preparation.refusal is None


@pytest.mark.parametrize(
    ("backed", "refusal"),
    [
        (True, None),
        (False, StoreRefusal.REMATERIALIZE),
        (None, StoreRefusal.RETRY),
    ],
)
def test_an_object_retired_before_is_verified_in_the_store(
    raw, backed: bool | None, refusal: StoreRefusal | None
) -> None:
    raw.hset(reclaim_history_key(_SCOPE), RESULT.content_digest, "3")
    check = _Backed(backed)
    preparation = _holds(raw, check).acquire(
        assertion_of("TASK_SUCCEEDED", _SUCCESS), _LOCATOR
    )
    assert check.asked == [RESULT.content_digest]
    assert preparation.refusal is refusal
    if refusal is None:
        assert RESULT.content_digest in raw.hgetall(reclaim_pins_key(_SCOPE))
    else:
        assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}
    if refusal is StoreRefusal.REMATERIALIZE:
        assert preparation.digests == (RESULT.content_digest,)


def test_a_refusal_in_one_scope_releases_the_holds_of_another(raw) -> None:
    other = _ref(b"other", "org-b")
    raw.hset(reclaim_deleting_key("org-b"), other.content_digest, "{}")
    payload = {
        "result_reference": RESULT.model_dump(mode="json"),
        "child_result_references": {"tsk-2": other.model_dump(mode="json")},
    }
    preparation = _holds(raw).acquire(assertion_of("TASK_SUCCEEDED", payload), _LOCATOR)
    assert preparation.refusal is StoreRefusal.RETRY_RETIREMENT
    assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}


def test_a_root_without_an_epoch_asks_for_a_retry(raw) -> None:
    client = _client(raw)
    holds = PublicationHolds(
        PinStore(client),
        lambda: None,
        _Backed(True),
        TreeExpander(client, _open, _stored_size),
    )
    preparation = holds.acquire(assertion_of("TASK_SUCCEEDED", _SUCCESS), _LOCATOR)
    assert preparation.refusal is StoreRefusal.RETRY


def _children(count: int) -> dict[str, Any]:
    return {
        f"tsk-{i}": _ref(f"child-{i}".encode()).model_dump(mode="json")
        for i in range(count)
    }


def test_objects_retired_before_across_many_chunks_are_all_verified(raw) -> None:
    children = _children(3 * CHUNK + 1)
    digests = sorted(ContentReference(**c).content_digest for c in children.values())
    marked = digests[::CHUNK]
    for digest in marked:
        raw.hset(reclaim_history_key(_SCOPE), digest, "1")
    check = _Backed(True)
    preparation = _holds(raw, check).acquire(
        assertion_of("TASK_SUCCEEDED", {"child_result_references": children}),
        _LOCATOR,
    )
    assert preparation.refusal is None
    assert sorted(check.asked) == marked


class _Moving(_Backed):
    """Confirms each object, and moves its retirement on as it is asked."""

    def __init__(self, raw: fakeredis.FakeRedis) -> None:
        super().__init__(True)
        self.raw = raw

    def __call__(self, reference: ContentReference) -> bool | None:
        self.raw.hincrby(reclaim_history_key(_SCOPE), reference.content_digest, 1)
        return super().__call__(reference)


def test_a_retirement_that_keeps_moving_bounds_the_verification(raw) -> None:
    raw.hset(reclaim_history_key(_SCOPE), RESULT.content_digest, "1")
    check = _Moving(raw)
    preparation = _holds(raw, check).acquire(
        assertion_of(
            "TASK_SUCCEEDED", {"result_reference": _SUCCESS["result_reference"]}
        ),
        _LOCATOR,
    )
    assert preparation.refusal is StoreRefusal.RETRY
    assert len(check.asked) == VERIFY_ROUNDS + 1
    assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}


def test_a_staged_success_writes_its_checkpoint_trees_through() -> None:
    checkpoint = _ref(b"checkpoint manifest")
    assertion = assertion_of(
        "TASK_SUCCEEDED",
        {
            "artifacts": [
                _tree("model", MANIFEST),
                _tree("ckpt", checkpoint, "step-4"),
            ],
            "artifacts_pending": True,
        },
    )
    assert assertion.tree_references(True) == [checkpoint]
    assert assertion.tree_references(False) == [MANIFEST]


class _FlakyPins(PinStore):
    """Pin store whose releases raise ``release_error``, and whose acquisitions raise
    ``acquire_error``, while each is set."""

    def __init__(self, client: RedisClient) -> None:
        super().__init__(client)
        self.release_error: Exception | None = None
        self.acquire_error: Exception | None = None

    def release(self, scope: str, digests: Sequence[str], nonce: str) -> int:
        if self.release_error is not None and digests:
            raise self.release_error
        return super().release(scope, digests, nonce)

    def acquire(self, *args: Any, **kwargs: Any) -> Acquisition:
        if self.acquire_error is not None:
            raise self.acquire_error
        return super().acquire(*args, **kwargs)


def _refused_in_another_scope(raw: fakeredis.FakeRedis) -> Assertion:
    other = _ref(b"other", "org-b")
    raw.hset(reclaim_deleting_key("org-b"), other.content_digest, "{}")
    return assertion_of(
        "TASK_SUCCEEDED",
        {
            "result_reference": RESULT.model_dump(mode="json"),
            "child_result_references": {"tsk-2": other.model_dump(mode="json")},
        },
    )


_DOWN = redis.exceptions.ConnectionError("control store down")


def test_a_release_the_control_store_drops_stays_owed_until_repaid(raw) -> None:
    pins = _FlakyPins(_client(raw))
    holds = _holds(raw, pins=pins)
    pins.release_error = _DOWN
    preparation = holds.acquire(_refused_in_another_scope(raw), _LOCATOR)
    assert preparation.refusal is StoreRefusal.RETRY_RETIREMENT
    assert RESULT.content_digest in raw.hgetall(reclaim_pins_key(_SCOPE))
    assert holds.repay() == 1
    pins.release_error = None
    assert holds.repay() == 0
    assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}


def _owe(
    raw: fakeredis.FakeRedis, holds: PublicationHolds, pins: _FlakyPins, scope: str
) -> ContentReference:
    """Leave the hold of a result in ``scope`` owed, as a refused publication whose
    release the control store dropped does."""
    result = _ref(b"result", scope)
    other = _ref(b"other", "zz-retiring")
    raw.hset(reclaim_deleting_key("zz-retiring"), other.content_digest, "{}")
    payload = {
        "result_reference": result.model_dump(mode="json"),
        "child_result_references": {"tsk-2": other.model_dump(mode="json")},
    }
    pins.release_error = _DOWN
    holds.acquire(assertion_of("TASK_SUCCEEDED", payload), _LOCATOR)
    pins.release_error = None
    return result


def test_an_owed_release_that_fails_does_not_block_the_others(raw) -> None:
    pins = _FlakyPins(_client(raw))
    holds = _holds(raw, pins=pins)
    _owe(raw, holds, pins, "org-poison")
    _owe(raw, holds, pins, _SCOPE)
    raw.delete(reclaim_pins_key("org-poison"))
    raw.set(reclaim_pins_key("org-poison"), "not a hash")

    preparation = holds.acquire(
        assertion_of(
            "TASK_SUCCEEDED",
            {"result_reference": _ref(b"c", "org-c").model_dump(mode="json")},
        ),
        _LOCATOR,
    )
    assert preparation.holds is not None
    assert holds.finish(preparation.holds, frozenset(), _LOCATOR) == []

    assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}
    assert raw.hgetall(reclaim_pins_key("org-c")) == {}
    assert holds.repay() == 1
    raw.delete(reclaim_pins_key("org-poison"))
    assert holds.repay() == 0


def test_a_repayment_the_control_store_drops_keeps_what_it_did_not_release(
    raw,
) -> None:
    pins = _FlakyPins(_client(raw))
    holds = _holds(raw, pins=pins)
    for scope in (_SCOPE, "org-d"):
        _owe(raw, holds, pins, scope)
    pins.release_error = _DOWN
    assert holds.repay() == 2
    pins.release_error = None
    assert holds.repay() == 0
    assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}
    assert raw.hgetall(reclaim_pins_key("org-d")) == {}


def test_an_expansion_the_control_store_drops_is_unavailable(raw) -> None:
    pins = _FlakyPins(_client(raw))
    pins.acquire_error = _DOWN
    outcome = _holds(raw, pins=pins).expand(MANIFEST, _LOCATOR)
    assert outcome.status is TreeStatus.UNAVAILABLE


def test_an_acquisition_that_fails_midway_releases_what_it_held(
    raw, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(raw)
    real = client.sync.eval
    acquisitions: list[int] = []

    def eval(script: str, numkeys: int, *keys_and_args: str) -> str:
        if numkeys > 1:
            acquisitions.append(numkeys)
            if len(acquisitions) == 2:
                raise redis.exceptions.ConnectionError("control store went away")
        return real(script, numkeys, *keys_and_args)

    monkeypatch.setattr(client.sync, "eval", eval)
    preparation = _holds(raw, pins=PinStore(client)).acquire(
        assertion_of(
            "TASK_SUCCEEDED", {"child_result_references": _children(CHUNK + 1)}
        ),
        _LOCATOR,
    )
    assert preparation.refusal is StoreRefusal.RETRY
    assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}
