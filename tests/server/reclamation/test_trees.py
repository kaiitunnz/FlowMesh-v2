"""Tree members: held before a write-through tree binds, protected only after its
manifest verifies, and expanded after the commit for a staged or backfilled tree."""

import hashlib
from collections.abc import Iterator
from typing import Any, cast

import fakeredis
import pytest
import redis.exceptions

from server.clients.redis import (
    RECLAIM_HOOKS_ON_KEY,
    RECLAIM_ROOT_EPOCH_KEY,
    RECLAIM_SCOPES_KEY,
    reclaim_cert_key,
    reclaim_deleting_key,
    reclaim_expanded_key,
    reclaim_history_key,
    reclaim_pins_key,
    reclaim_protected_key,
    reclaim_uncertified_key,
    reclaim_unexpanded_key,
)
from server.reclamation.pins import Nonces, PinStore
from server.reclamation.publications import PublicationHolds, assertion_of
from server.reclamation.trees import _EXPANDED, TreeExpander, TreeStatus
from server.task.results import ResultUnavailable, ResultUnreadable
from shared.content import ContentReference, StoreRefusal, reference_for
from shared.content.tree import TREE_MEDIA_TYPE, TreeManifest, TreeMember
from shared.tasks.artifact_binding import ArtifactBinding, ArtifactKind
from tests.server.redis_helpers import fake_redis_client

_SCOPE = "org-a"
_LOCATOR = {"task": "tsk-1", "kind": "result"}


class _Store:
    """Stored objects by digest, read in small windows and verified at the end."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.reads = 0
        self.down = False
        self.fail_after: list[int] = []
        """For each coming read in turn, fail it after this many windows."""
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
            # Still a well-formed tree, naming another member: only the digest
            # check at the end of the stream catches it.
            first = TreeManifest.decode(data).members[0].digest.encode()
            data = data.replace(first, b"f" * 64, 1)
        fail_after = self.fail_after.pop(0) if self.fail_after else None
        for index, start in enumerate(range(0, len(data), 7)):
            if fail_after is not None and index >= fail_after:
                raise ResultUnavailable("store went away")
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
        self.raw.set(RECLAIM_ROOT_EPOCH_KEY, "9")
        self.client = client = fake_redis_client(server)
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
    assert preparation.refusal is StoreRefusal.RETRY_RETIREMENT
    assert preparation.digests == (members[1200].digest,)
    assert rig.pinned() == set()


def test_a_malformed_manifest_is_rejected_before_acceptance(rig) -> None:
    manifest = rig.store.put(b'{"schema":1,"members":[["../x","00",1]]}')
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.refusal is None
    assert preparation.malformed
    assert rig.pinned() == set()


def test_a_store_outage_asks_for_a_retry(rig) -> None:
    manifest = _tree(rig.store, _members(3))
    rig.store.down = True
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.refusal is StoreRefusal.RETRY
    assert not preparation.malformed


def test_a_tree_already_expanded_is_not_read_again(rig) -> None:
    manifest = _tree(rig.store, _members(3))
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    rig.raw.sadd(reclaim_expanded_key(_SCOPE), manifest.content_digest)
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
    rig.raw.hdel(reclaim_deleting_key(_SCOPE), members[0].digest)
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
    # More than a page, so members are held before the stream's digest check.
    members = _members(2500)
    manifest = _tree(rig.store, members)
    rig.store.corrupt.add(manifest.content_digest)
    outcome = rig.holds.expand(manifest, _LOCATOR)
    assert outcome.status is TreeStatus.MALFORMED
    # One read held the members, a second released them.
    assert rig.store.reads == 2
    assert rig.protected() == set()
    assert rig.pinned() == set()


def test_write_through_tree_members_are_not_kept_in_memory(rig) -> None:
    members = _members(2500)
    manifest = _tree(rig.store, members)
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.holds is not None
    assert preparation.holds.held == {_SCOPE: (manifest.content_digest,)}
    rig.holds.release(preparation.holds)
    assert rig.pinned() == set()


def test_a_release_the_store_blocks_is_repaid_once_it_reads_again(rig) -> None:
    members = _members(2500)
    manifest = _tree(rig.store, members)
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    # The outage lands after more than a page of members was held, and lasts
    # through the release that follows.
    rig.store.fail_after = [len(TreeManifest.of(members).encode()) // 7 - 10, 0, 0]
    outcome = rig.holds.expand(manifest, _LOCATOR)
    assert outcome.status is TreeStatus.UNAVAILABLE
    assert rig.pinned()
    assert rig.protected() == {manifest.content_digest}
    assert rig.holds.repay() == 1

    assert rig.holds.repay() == 0
    assert rig.pinned() == set()
    assert rig.protected() == {manifest.content_digest}


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


def test_a_staged_tree_with_a_member_being_retired_waits_for_it(rig) -> None:
    members = _members(600)
    manifest = _tree(rig.store, members)
    preparation = rig.holds.acquire(_success(manifest, staged=True), _LOCATOR)
    assert preparation.holds is not None
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    # Past the first page, so a page was held before the expansion stopped.
    rig.raw.hset(reclaim_deleting_key(_SCOPE), members[400].digest, "{}")
    bound = frozenset({manifest.content_digest})

    ((tree, outcome),) = rig.holds.finish(preparation.holds, bound, _LOCATOR)
    assert tree == manifest and outcome.status is TreeStatus.RETIRING
    assert outcome.retiring == (members[400].digest,)
    assert rig.pinned() == set()
    assert rig.protected() == {manifest.content_digest}
    assert rig.raw.scard(reclaim_uncertified_key(_SCOPE)) == 0

    rig.raw.hdel(reclaim_deleting_key(_SCOPE), members[400].digest)
    assert rig.holds.finish(preparation.holds, bound, _LOCATOR) == []
    assert {m.digest for m in members} <= rig.protected()


def test_a_bound_tree_with_a_member_being_retired_uncertifies_its_scope(rig) -> None:
    members = _members(3)
    manifest = _tree(rig.store, members)
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    rig.raw.sadd(reclaim_unexpanded_key(_SCOPE), manifest.content_digest)
    rig.raw.hset(reclaim_deleting_key(_SCOPE), members[1].digest, "{}")

    outcome = rig.holds.expand(manifest, _LOCATOR, staged=False)
    assert outcome.status is TreeStatus.RETIRING
    assert rig.raw.smembers(reclaim_uncertified_key(_SCOPE)) == {
        f"tree:{manifest.content_digest}"
    }
    assert rig.unexpanded() == {manifest.content_digest}

    rig.raw.hdel(reclaim_deleting_key(_SCOPE), members[1].digest)
    assert rig.holds.expand(manifest, _LOCATOR, staged=False).status is TreeStatus.DONE
    assert rig.raw.scard(reclaim_uncertified_key(_SCOPE)) == 0
    assert rig.unexpanded() == set()


def test_an_expansion_reports_the_members_retired_before(rig) -> None:
    members = _members(3)
    manifest = _tree(rig.store, members)
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    rig.raw.hset(reclaim_history_key(_SCOPE), members[2].digest, "1")

    outcome = rig.holds.expand(manifest, _LOCATOR)

    assert outcome.status is TreeStatus.DONE
    assert outcome.retired == (members[2].reference(_SCOPE),)
    assert {m.digest for m in members} <= rig.protected()


def test_reconciling_leaves_a_tree_its_owed_upload_expands(rig) -> None:
    manifest = _tree(rig.store, _members(3))
    rig.raw.sadd(RECLAIM_SCOPES_KEY, _SCOPE)
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    rig.raw.sadd(reclaim_unexpanded_key(_SCOPE), manifest.content_digest)

    waiting = rig.trees.reconcile(
        rig.trees.scopes(),
        lambda m: rig.holds.expand(m, _LOCATOR, staged=False),
        lambda: False,
        {manifest.content_digest},
    )

    assert waiting == 1 and rig.store.reads == 0


def _flaky(rig: _Rig, method: str, fail_on: int, monkeypatch) -> list[int]:
    """Make the ``fail_on``-th call of the control store's ``method`` drop its
    connection; returns the running call count."""
    real = getattr(rig.client.sync, method)
    calls = [0]

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] == fail_on:
            raise redis.exceptions.ConnectionError("connection reset")
        return real(*args, **kwargs)

    monkeypatch.setattr(rig.client.sync, method, flaky)
    return calls


def test_a_promotion_the_control_store_drops_finishes_when_promoted_again(
    rig, monkeypatch
) -> None:
    members = _members(600)
    manifest = _tree(rig.store, members)
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.holds is not None
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    _flaky(rig, "eval", 2, monkeypatch)

    outcome = rig.trees.promote(manifest, preparation.holds.nonce)

    assert outcome.status is TreeStatus.UNAVAILABLE
    assert rig.unexpanded() == {manifest.content_digest}
    assert {m.digest for m in members[:256]} <= rig.protected()
    assert {m.digest for m in members[256:]} <= rig.pinned()
    assert rig.trees.promote(manifest, preparation.holds.nonce).status is (
        TreeStatus.DONE
    )
    assert {m.digest for m in members} <= rig.protected()
    assert rig.pinned() == {manifest.content_digest}
    assert rig.unexpanded() == set()


def test_a_hold_the_control_store_drops_is_unavailable(rig) -> None:
    manifest = _tree(rig.store, _members(3))

    def dropped(_scope: str, _members: list[TreeMember]) -> None:
        raise redis.exceptions.TimeoutError("timed out")

    assert rig.trees.hold(manifest, dropped).status is TreeStatus.UNAVAILABLE


def test_a_release_the_control_store_drops_is_unavailable(rig) -> None:
    manifest = _tree(rig.store, _members(3))

    def dropped(*_args: Any) -> int:
        raise redis.exceptions.ConnectionError("connection reset")

    outcome = rig.trees.release(manifest, "9:1", dropped)
    assert outcome.status is TreeStatus.UNAVAILABLE


def test_a_reconcile_the_control_store_drops_resumes_on_the_next_pass(
    rig, monkeypatch
) -> None:
    trees = [_tree(rig.store, _members(3, salt)) for salt in ("a", "b")]
    rig.raw.sadd(RECLAIM_SCOPES_KEY, _SCOPE)
    for manifest in trees:
        rig.raw.sadd(reclaim_unexpanded_key(_SCOPE), manifest.content_digest)
        rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    _flaky(rig, "sismember", 2, monkeypatch)

    def expand(manifest: ContentReference) -> Any:
        return rig.holds.expand(manifest, _LOCATOR, staged=False)

    assert rig.trees.reconcile(rig.trees.scopes(), expand, lambda: False) == 1
    assert len(rig.unexpanded()) == 1
    assert rig.trees.reconcile(rig.trees.scopes(), expand, lambda: False) == 0
    assert rig.unexpanded() == set()
    for salt in ("a", "b"):
        assert {m.digest for m in _members(3, salt)} <= rig.protected()


def test_a_member_whose_hold_ended_keeps_its_tree_unexpanded(rig) -> None:
    members = _members(3)
    manifest = _tree(rig.store, members)
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.holds is not None
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    rig.raw.hdel(reclaim_pins_key(_SCOPE), members[1].digest)
    bound = frozenset({manifest.content_digest})

    ((tree, outcome),) = rig.holds.finish(preparation.holds, bound, _LOCATOR)

    assert tree == manifest and outcome.status is TreeStatus.INCOMPLETE
    assert members[1].digest not in rig.protected()
    assert rig.unexpanded() == {manifest.content_digest}
    # A later expansion under a fresh hold finishes it.
    waiting = rig.trees.reconcile(
        rig.trees.scopes(),
        lambda m: rig.holds.expand(m, _LOCATOR, staged=False),
        lambda: False,
    )
    assert waiting == 0
    assert {m.digest for m in members} <= rig.protected()
    assert rig.unexpanded() == set()


def test_an_expansion_a_later_root_fenced_protects_nothing(rig) -> None:
    members = _members(3)
    manifest = _tree(rig.store, members)
    rig.raw.sadd(RECLAIM_SCOPES_KEY, _SCOPE)
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    rig.raw.sadd(reclaim_unexpanded_key(_SCOPE), manifest.content_digest)
    rig.raw.set(RECLAIM_ROOT_EPOCH_KEY, "10")

    outcome = rig.holds.expand(manifest, _LOCATOR, staged=False)

    assert outcome.status is TreeStatus.FENCED
    assert rig.protected() == {manifest.content_digest}
    assert rig.unexpanded() == {manifest.content_digest}
    assert rig.pinned() == set()

    other = _tree(rig.store, _members(3, "b"))
    rig.raw.sadd(reclaim_protected_key(_SCOPE), other.content_digest)
    rig.raw.sadd(reclaim_unexpanded_key(_SCOPE), other.content_digest)
    expanded: list[ContentReference] = []

    def expand(m: ContentReference) -> Any:
        expanded.append(m)
        return rig.holds.expand(m, _LOCATOR, staged=False)

    # The pass stops at the first fenced expansion.
    assert rig.trees.reconcile(rig.trees.scopes(), expand, lambda: False) == 1
    assert len(expanded) == 1


def test_a_write_through_promotion_a_later_root_fenced_leaves_the_tree(rig) -> None:
    members = _members(3)
    manifest = _tree(rig.store, members)
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.holds is not None
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    rig.raw.set(RECLAIM_ROOT_EPOCH_KEY, "10")

    assert (
        rig.trees.promote(manifest, preparation.holds.nonce).status is TreeStatus.FENCED
    )
    assert rig.protected() == {manifest.content_digest}
    assert rig.unexpanded() == {manifest.content_digest}


def test_settling_an_unbound_tree_never_drops_one_a_publication_holds(
    rig, monkeypatch
) -> None:
    manifest = _tree(rig.store, _members(3))
    digest = manifest.content_digest
    rig.raw.sadd(reclaim_unexpanded_key(_SCOPE), digest)
    pins = PinStore(rig.client)
    raced: list[str] = []
    real_pipeline, real_eval = rig.client.sync.control_pipeline, rig.client.sync.eval

    def race() -> None:
        # A publication naming the tree holds it, entering the unexpanded set with
        # its hold, between the settle's first read and its write.
        if not raced:
            raced.append(digest)
            pins.acquire(_SCOPE, [digest], "9:7", {}, manifests=[digest])

    class _Pipeline:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __enter__(self) -> Any:
            pipe = self._inner.__enter__()
            execute = pipe.execute

            def raced_execute() -> Any:
                result = execute()
                race()
                return result

            pipe.execute = raced_execute
            return pipe

        def __exit__(self, *exc: Any) -> Any:
            return self._inner.__exit__(*exc)

    def raced_eval(*args: Any) -> Any:
        result = real_eval(*args)
        race()
        return result

    monkeypatch.setattr(
        rig.client.sync,
        "control_pipeline",
        lambda *a, **k: _Pipeline(real_pipeline(*a, **k)),
    )
    monkeypatch.setattr(rig.client.sync, "eval", raced_eval)

    rig.trees.settle_unbound(_SCOPE, digest)

    assert raced
    assert digest in rig.pinned()
    assert rig.unexpanded() == {digest}


def _file_success(reference: ContentReference) -> Any:
    binding = ArtifactBinding(
        name="weights.bin",
        kind=ArtifactKind.FILE,
        reference=reference,
        size_bytes=reference.size_bytes,
        files=1,
    )
    return assertion_of("TASK_SUCCEEDED", {"artifacts": [binding.model_dump()]})


def test_a_file_with_a_manifests_bytes_never_marks_its_tree_expanded(rig) -> None:
    members = _members(3)
    manifest = _tree(rig.store, members)
    as_file = manifest.model_copy(update={"media_type": "application/octet-stream"})
    rig.raw.set(RECLAIM_HOOKS_ON_KEY, "9")
    rig.raw.set(reclaim_cert_key(_SCOPE), "9")
    published = rig.holds.acquire(_file_success(as_file), _LOCATOR)
    assert published.holds is not None
    rig.raw.sadd(reclaim_protected_key(_SCOPE), as_file.content_digest)
    assert rig.holds.finish(published.holds, frozenset(), _LOCATOR) == []
    assert rig.protected() == {manifest.content_digest}
    assert rig.unexpanded() == set()

    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)

    assert preparation.holds is not None
    assert preparation.holds.trees == [manifest]
    assert {m.digest for m in members} <= rig.pinned()
    assert rig.trees.expanded([manifest]) == set()
    assert (
        rig.holds.finish(
            preparation.holds, frozenset({manifest.content_digest}), _LOCATOR
        )
        == []
    )
    assert {m.digest for m in members} <= rig.protected()
    assert rig.trees.expanded([manifest]) == {manifest.content_digest}


@pytest.mark.parametrize("staged", [False, True], ids=["write-through", "staged"])
def test_a_fully_promoted_tree_is_expanded(rig, staged: bool) -> None:
    manifest = _tree(rig.store, _members(600))
    preparation = rig.holds.acquire(_success(manifest, staged), _LOCATOR)
    assert preparation.holds is not None
    assert rig.trees.expanded([manifest]) == set()
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    bound = frozenset({manifest.content_digest})

    assert rig.holds.finish(preparation.holds, bound, _LOCATOR) == []

    assert rig.trees.expanded([manifest]) == {manifest.content_digest}
    reads = rig.store.reads
    again = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert again.holds is not None and again.holds.trees == []
    assert rig.store.reads == reads


def test_an_incomplete_promotion_leaves_the_tree_unexpanded(rig) -> None:
    members = _members(3)
    manifest = _tree(rig.store, members)
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.holds is not None
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    rig.raw.hdel(reclaim_pins_key(_SCOPE), members[1].digest)

    outcome = rig.trees.promote(manifest, preparation.holds.nonce)

    assert outcome.status is TreeStatus.INCOMPLETE
    assert rig.trees.expanded([manifest]) == set()
    assert not rig.raw.exists(reclaim_expanded_key(_SCOPE))


def test_a_fenced_promotion_leaves_the_tree_unexpanded(rig, monkeypatch) -> None:
    manifest = _tree(rig.store, _members(3))
    preparation = rig.holds.acquire(_success(manifest), _LOCATOR)
    assert preparation.holds is not None
    rig.raw.sadd(reclaim_protected_key(_SCOPE), manifest.content_digest)
    real_eval = rig.client.sync.eval

    def fence_before_expanded(script: str, *args: Any) -> Any:
        # The members promote under this root; a later root takes the epoch just
        # before the manifest is marked expanded.
        if script == _EXPANDED:
            rig.raw.set(RECLAIM_ROOT_EPOCH_KEY, "10")
        return real_eval(script, *args)

    monkeypatch.setattr(rig.client.sync, "eval", fence_before_expanded)

    outcome = rig.trees.promote(manifest, preparation.holds.nonce)

    assert outcome.status is TreeStatus.FENCED
    assert rig.trees.expanded([manifest]) == set()
    assert rig.unexpanded() == {manifest.content_digest}
