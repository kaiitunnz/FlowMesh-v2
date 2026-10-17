"""Tree members: held before a write-through tree binds, protected only after its
manifest verifies, and expanded after the commit for a staged or backfilled tree."""

import hashlib
from collections.abc import Iterator
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import (
    RECLAIM_SCOPES_KEY,
    reclaim_deleting_key,
    reclaim_pins_key,
    reclaim_protected_key,
    reclaim_unexpanded_key,
)
from server.reclamation.pins import Nonces, PinStore
from server.reclamation.publications import PublicationHolds, assertion_of
from server.reclamation.trees import TreeExpander, TreeStatus
from server.task.results import ResultUnavailable, ResultUnreadable
from shared.content import ContentReference, reference_for
from shared.content.tree import TREE_MEDIA_TYPE, TreeManifest, TreeMember
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from shared.tasks.publication import PublicationDisposition
from tests.server.redis_helpers import fake_redis_client

_SCOPE = "org-a"
_LOCATOR = {"task": "tsk-1", "kind": "result"}


class _Store:
    """Stored objects by digest, read in small windows and verified at the end."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.reads = 0
        self.down = False
        self.corrupt: set[str] = set()

    def put(self, data: bytes, media_type: str = TREE_MEDIA_TYPE) -> ContentReference:
        reference = reference_for(_SCOPE, data).model_copy(
            update={"media_type": media_type}
        )
        self.objects[reference.content_digest] = data
        return reference

    def open(self, reference: ContentReference) -> Iterator[bytes]:
        self.reads += 1
        if self.down:
            raise ResultUnavailable("store down")
        data = self.objects.get(reference.content_digest)
        if data is None:
            raise ResultUnreadable("missing")
        if reference.content_digest in self.corrupt:
            data = data.replace(b'"m/0', b'"m/9', 1)
        for start in range(0, len(data), 7):
            yield data[start : start + 7]
        if hashlib.sha256(data).hexdigest() != reference.content_digest:
            raise ResultUnreadable("digest mismatch")

    def stored_size(self, reference: ContentReference) -> int | None:
        data = self.objects.get(reference.content_digest)
        return None if data is None else len(data)


def _members(count: int, salt: str = "") -> list[TreeMember]:
    return [
        TreeMember(
            f"m/{i:04d}", hashlib.sha256(f"{salt}{i}".encode()).hexdigest(), i + 1
        )
        for i in range(count)
    ]


def _tree(store: _Store, members: list[TreeMember]) -> ContentReference:
    return store.put(TreeManifest.of(members).encode())


def _binding(manifest: ContentReference, name: str = "model") -> dict[str, Any]:
    return ArtifactBinding(
        name=name,
        kind=ArtifactKind.TREE,
        reference=manifest,
        size_bytes=1,
        files=1,
    ).model_dump(mode="json")


class _Rig:
    def __init__(self) -> None:
        server = fakeredis.FakeServer()
        self.raw = fakeredis.FakeRedis(server=server, decode_responses=True)
        client = fake_redis_client(server)
        self.store = _Store()
        self.trees = TreeExpander(client, self.store.open, self.store.stored_size)
        self.holds = PublicationHolds(
            PinStore(client), lambda: Nonces(9), lambda _r: True, trees=self.trees
        )

    def protected(self) -> set[str]:
        return cast(set[str], self.raw.smembers(reclaim_protected_key(_SCOPE)))

    def pinned(self) -> set[str]:
        return set(cast(list[str], self.raw.hkeys(reclaim_pins_key(_SCOPE))))

    def unexpanded(self) -> set[str]:
        return cast(set[str], self.raw.smembers(reclaim_unexpanded_key(_SCOPE)))


@pytest.fixture
def rig() -> _Rig:
    return _Rig()


def _success(manifest: ContentReference, staged: bool = False) -> Any:
    payload: dict[str, Any] = {"artifacts": [_binding(manifest)]}
    if staged:
        payload["artifacts_pending"] = True
    return assertion_of("TASK_SUCCEEDED", payload)


def test_a_write_through_tree_holds_its_members_before_the_commit(rig) -> None:
    members = _members(2500)
    manifest = _tree(rig.store, members)
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.holds is not None
    assert rig.pinned() == {m.digest for m in members} | {manifest.content_digest}
    assert rig.protected() == set()
    assert rig.unexpanded() == {manifest.content_digest}

    # The commit bound the manifest; its members are protected, the holds end.
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    assert (
        rig.holds.finish(
            preparation.holds, frozenset({manifest.content_digest}), _LOCATOR
        )
        == []
    )
    assert rig.protected() == {m.digest for m in members} | {manifest.content_digest}
    assert rig.pinned() == set()
    assert rig.unexpanded() == set()


def test_a_tree_the_commit_did_not_bind_protects_nothing(rig) -> None:
    members = _members(3)
    manifest = _tree(rig.store, members)
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.holds is not None
    assert rig.holds.finish(preparation.holds, frozenset(), _LOCATOR) == []
    assert rig.protected() == set()
    assert rig.pinned() == set()


def test_a_member_being_retired_refuses_the_tree_and_holds_nothing(rig) -> None:
    members = _members(1500)
    manifest = _tree(rig.store, members)
    rig.raw.hset(reclaim_deleting_key(_SCOPE), members[1200].digest, "{}")
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.refusal is PublicationDisposition.RETRY_RETIREMENT
    assert preparation.digests == (members[1200].digest,)
    assert rig.pinned() == set()


def test_a_malformed_manifest_is_rejected_before_acceptance(rig) -> None:
    manifest = rig.store.put(b'{"schema":1,"members":[["../x","00",1]]}')
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.refusal is PublicationDisposition.REJECTED
    assert preparation.malformed
    assert rig.pinned() == set()


def test_a_store_outage_asks_for_a_retry(rig) -> None:
    manifest = _tree(rig.store, _members(3))
    rig.store.down = True
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.refusal is PublicationDisposition.RETRY
    assert not preparation.malformed


def test_a_tree_already_expanded_is_not_read_again(rig) -> None:
    manifest = _tree(rig.store, _members(3))
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.holds is not None and preparation.holds.trees == []
    assert rig.store.reads == 0


def test_a_staged_tree_is_expanded_once_its_binding_is_durable(rig) -> None:
    members = _members(4)
    manifest = _tree(rig.store, members)
    rig.raw.hset(reclaim_deleting_key(_SCOPE), members[0].digest, "{}")
    preparation = rig.holds.acquire(_success(manifest, staged=True), _LOCATOR)
    assert preparation.holds is not None
    # Members still to be uploaded are not held or read before the commit.
    assert rig.store.reads == 0
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    assert (
        rig.holds.finish(
            preparation.holds, frozenset({manifest.content_digest}), _LOCATOR
        )
        == []
    )
    assert {m.digest for m in members} <= rig.protected()
    assert rig.pinned() == set()
    assert rig.unexpanded() == set()


def test_a_corrupt_read_never_protects_a_member(rig) -> None:
    members = _members(5)
    manifest = _tree(rig.store, members)
    rig.store.corrupt.add(manifest.content_digest)
    outcome = rig.holds.expand(manifest, _LOCATOR)
    assert outcome.status is TreeStatus.MALFORMED
    assert rig.protected() == set()
    assert rig.pinned() == set()


def test_reconciling_expands_bound_trees_and_drops_unbound_ones(rig) -> None:
    bound, unbound, held = (
        _tree(rig.store, _members(3, salt)) for salt in ("a", "b", "c")
    )
    rig.raw.sadd(RECLAIM_SCOPES_KEY, _SCOPE)
    rig.raw.sadd(
        reclaim_unexpanded_key(_SCOPE),
        bound.content_digest,
        unbound.content_digest,
        held.content_digest,
    )
    rig.raw.sadd(reclaim_protected_key(_SCOPE), bound.content_digest)
    rig.raw.hset(reclaim_pins_key(_SCOPE), held.content_digest, '[{"n":"9:1"}]')
    waiting = rig.trees.reconcile(
        rig.trees.scopes(),
        lambda manifest: rig.holds.expand(manifest, _LOCATOR),
        lambda: False,
    )
    assert waiting == 1
    assert rig.unexpanded() == {held.content_digest}
    assert {m.digest for m in _members(3, "a")} <= rig.protected()
