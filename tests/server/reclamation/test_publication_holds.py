"""What a publication names, and how its objects are held, refused and released."""

from typing import Any

import fakeredis
import pytest

from server.clients.redis import (
    reclaim_deleting_key,
    reclaim_history_key,
    reclaim_pins_key,
    reclaim_unexpanded_key,
)
from server.reclamation.pins import Nonces, PinStore
from server.reclamation.publications import (
    PublicationHolds,
    assertion_of,
)
from shared.content import ContentReference, reference_for
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from shared.tasks.publication import PublicationDisposition
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


RESULT, CHILD, FILE, MANIFEST, PREPARED = (
    _ref(b"result"),
    _ref(b"child"),
    _ref(b"file"),
    _ref(b"manifest"),
    _ref(b"prepared"),
)


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


def _holds(raw: fakeredis.FakeRedis, backed: _Backed | None = None) -> PublicationHolds:
    client = fake_redis_client(raw.connection_pool.connection_kwargs["server"])
    return PublicationHolds(
        PinStore(client), lambda: Nonces(7), backed or _Backed(True)
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
    }
    assert raw.smembers(reclaim_unexpanded_key(_SCOPE)) == {MANIFEST.content_digest}
    holds.release(preparation.holds)
    assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}


def test_an_object_being_retired_refuses_and_holds_nothing(raw) -> None:
    raw.hset(reclaim_deleting_key(_SCOPE), FILE.content_digest, "{}")
    preparation = _holds(raw).acquire(
        assertion_of("TASK_SUCCEEDED", _SUCCESS), _LOCATOR
    )
    assert preparation.refusal is PublicationDisposition.RETRY_RETIREMENT
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
        (False, PublicationDisposition.REMATERIALIZE),
        (None, PublicationDisposition.RETRY),
    ],
)
def test_an_object_retired_before_is_verified_in_the_store(
    raw, backed: bool | None, refusal: PublicationDisposition | None
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
    if refusal is PublicationDisposition.REMATERIALIZE:
        assert preparation.digests == (RESULT.content_digest,)


def test_a_refusal_in_one_scope_releases_the_holds_of_another(raw) -> None:
    other = _ref(b"other", "org-b")
    raw.hset(reclaim_deleting_key("org-b"), other.content_digest, "{}")
    payload = {
        "result_reference": RESULT.model_dump(mode="json"),
        "child_result_references": {"tsk-2": other.model_dump(mode="json")},
    }
    preparation = _holds(raw).acquire(assertion_of("TASK_SUCCEEDED", payload), _LOCATOR)
    assert preparation.refusal is PublicationDisposition.RETRY_RETIREMENT
    assert raw.hgetall(reclaim_pins_key(_SCOPE)) == {}


def test_a_root_without_an_epoch_asks_for_a_retry(raw) -> None:
    client = fake_redis_client(raw.connection_pool.connection_kwargs["server"])
    holds = PublicationHolds(PinStore(client), lambda: None, _Backed(True))
    preparation = holds.acquire(assertion_of("TASK_SUCCEEDED", _SUCCESS), _LOCATOR)
    assert preparation.refusal is PublicationDisposition.RETRY
