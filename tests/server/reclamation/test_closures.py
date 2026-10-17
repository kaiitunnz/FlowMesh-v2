"""Tree closures: a bound tree's members are marked in its scope only once its manifest
verifies, and a tree whose read settled is never read again."""

import hashlib
import logging
import threading
from collections.abc import Callable, Iterator

from server.reclamation.closures import Expansion, TreeClosures
from server.reclamation.pacing import Pacer
from server.task.results import ResultUnavailable, ResultUnreadable
from shared.content import ContentReference, reference_for
from shared.content.tree import TREE_MEDIA_TYPE, TreeManifest, TreeMember

_A = "org-a"
_B = "org-b"
_LOGGER = logging.getLogger("test.closures")

type Opener = Callable[[ContentReference], Iterator[bytes]]


def _members(count: int) -> list[TreeMember]:
    return [
        TreeMember(f"f{i:04d}", hashlib.sha256(f"m{i}".encode()).hexdigest(), i + 1)
        for i in range(count)
    ]


def _tree(manifest: bytes, scope: str = _A) -> ContentReference:
    return reference_for(scope, manifest, media_type=TREE_MEDIA_TYPE)


def _chunks(data: bytes, size: int = 64) -> Iterator[bytes]:
    for start in range(0, len(data), size):
        yield data[start : start + size]


class _Counted:
    def __init__(self, open_content: Opener) -> None:
        self.open_content = open_content
        self.reads = 0

    def __call__(self, reference: ContentReference) -> Iterator[bytes]:
        self.reads += 1
        return self.open_content(reference)


def _serving(data: bytes) -> _Counted:
    return _Counted(lambda _: _chunks(data))


def _pacer(stop: threading.Event | None = None, every: int = 1000) -> Pacer:
    return Pacer(stop or threading.Event(), every, duty=1.0, min_pause_sec=0)


def test_a_verified_tree_marks_its_members_in_its_own_scope() -> None:
    members = _members(5)
    manifest = TreeManifest.of(members).encode()
    closures = TreeClosures(_serving(manifest), _LOGGER)

    assert closures.expand(_tree(manifest), _pacer()) is Expansion.EXPANDED

    assert all(closures.mark.binds(_A, m.digest) for m in members)
    assert not any(closures.mark.binds(_B, m.digest) for m in members)
    assert closures.known(_tree(manifest)) is Expansion.EXPANDED
    assert closures.known(_tree(manifest, _B)) is None


def test_an_expanded_tree_is_never_read_again() -> None:
    manifest = TreeManifest.of(_members(3)).encode()
    opener = _serving(manifest)
    closures = TreeClosures(opener, _LOGGER)

    for _ in range(3):
        assert closures.expand(_tree(manifest), _pacer()) is Expansion.EXPANDED
        closures.compact(_pacer())

    assert opener.reads == 1


def test_a_manifest_failing_verification_marks_no_member() -> None:
    members = _members(50)
    manifest = TreeManifest.of(members).encode()
    opener = _Counted(lambda _: _corrupt(manifest))
    closures = TreeClosures(opener, _LOGGER)
    tree = _tree(manifest)

    assert closures.expand(tree, _pacer()) is Expansion.MALFORMED
    closures.compact(_pacer())
    assert closures.expand(tree, _pacer()) is Expansion.MALFORMED

    assert not any(closures.mark.binds(_A, m.digest) for m in members)
    assert opener.reads == 1
    assert closures.known(tree) is Expansion.MALFORMED
    assert closures.known(_tree(manifest, _B)) is None


def _corrupt(manifest: bytes) -> Iterator[bytes]:
    yield from _chunks(manifest)
    raise ResultUnreadable("digest mismatch")


def test_a_malformed_manifest_marks_no_member_and_is_not_read_again() -> None:
    members = _members(4)
    manifest = TreeManifest.of(members).encode()
    broken = manifest[:-2] + b',["../escape","' + b"0" * 64 + b'",1]]}'
    opener = _serving(broken)
    closures = TreeClosures(opener, _LOGGER)
    tree = _tree(broken)

    assert closures.expand(tree, _pacer()) is Expansion.MALFORMED
    assert closures.expand(tree, _pacer()) is Expansion.MALFORMED

    assert not any(closures.mark.binds(_A, m.digest) for m in members)
    assert opener.reads == 1


def test_an_unavailable_manifest_is_read_again_at_the_next_call() -> None:
    members = _members(3)
    manifest = TreeManifest.of(members).encode()

    def flaky(_: ContentReference) -> Iterator[bytes]:
        if opener.reads == 1:
            raise ResultUnavailable("store down")
        yield from _chunks(manifest)

    opener = _Counted(flaky)
    closures = TreeClosures(opener, _LOGGER)
    tree = _tree(manifest)

    assert closures.expand(tree, _pacer()) is Expansion.UNAVAILABLE
    assert closures.known(tree) is None
    assert not any(closures.mark.binds(_A, m.digest) for m in members)

    assert closures.expand(tree, _pacer()) is Expansion.EXPANDED
    assert all(closures.mark.binds(_A, m.digest) for m in members)
    assert opener.reads == 2


def test_a_stop_mid_tree_marks_nothing_and_leaves_it_unread() -> None:
    members = _members(10)
    manifest = TreeManifest.of(members).encode()
    closures = TreeClosures(_serving(manifest), _LOGGER)
    stop = threading.Event()
    stop.set()
    tree = _tree(manifest)

    assert closures.expand(tree, _pacer(stop, every=3)) is Expansion.STOPPED

    assert not any(closures.mark.binds(_A, m.digest) for m in members)
    assert closures.known(tree) is None
    assert closures.expand(tree, _pacer()) is Expansion.EXPANDED


def test_the_closures_count_the_trees_they_settle_in_their_size() -> None:
    manifests = [TreeManifest.of(_members(i + 1)).encode() for i in range(20)]
    closures = TreeClosures(
        _Counted(lambda ref: _chunks(by[ref.content_digest])), _LOGGER
    )
    by = {_tree(m).content_digest: m for m in manifests}
    for manifest in manifests:
        closures.expand(_tree(manifest), _pacer())
    closures.compact(_pacer())

    assert closures.nbytes == closures.mark.nbytes + 20 * 32
