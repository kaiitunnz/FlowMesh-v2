"""Where the worker's content plane reads an object from, and what it keeps of it."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from shared.content import (
    STREAM_CHUNK_BYTES,
    ContentHydrationError,
    ContentReference,
    ContentUnavailable,
    FabricObjectStore,
    SharedFilesystemObjectStore,
    reference_for,
)
from worker.content import TaskContentStore, WorkerContentCache, WorkerContentPlane
from worker.content.access import ContentAccessDenied

_LARGE = 64


class _Lane:
    """A cache lane holding nothing locally, whose peer has whatever it is given."""

    def __init__(self, peer: dict[str, bytes] | None = None) -> None:
        # What the peer holds, by digest.
        self.peer = peer or {}
        self.peer_reads: list[ContentReference] = []
        self.kept: list[ContentReference] = []

    def read_local(self, reference: ContentReference) -> bytes:
        raise ContentHydrationError("not held")

    def copy_local(self, reference: ContentReference, dest: Path) -> None:
        raise ContentHydrationError("not held")

    def read_peer(self, reference: ContentReference, task_id: str) -> bytes:
        self.peer_reads.append(reference)
        if reference.content_digest not in self.peer:
            raise ContentHydrationError("no holder")
        return self.peer[reference.content_digest]

    def copy_peer(self, reference: ContentReference, task_id: str, dest: Path) -> None:
        dest.write_bytes(self.read_peer(reference, task_id))

    def keep(self, scope: str, data: bytes, *, media_type: str = "") -> Any:
        reference = reference_for(scope, data, media_type=media_type)
        self.kept.append(reference)
        return reference

    def held(self, reference: ContentReference) -> ContentReference:
        return reference

    def keep_file(self, reference: ContentReference, path: Path) -> Any:
        assert reference_for(reference.authorization_scope, path.read_bytes()) == (
            reference.model_copy(update={"media_type": "application/octet-stream"})
        )
        self.kept.append(reference)
        return reference


class _DownStore(FabricObjectStore):
    def write(self, scope: str, data: bytes, *, media_type: str = "") -> Any:
        raise ContentUnavailable("store down")

    def fetch(self, reference: ContentReference) -> bytes:
        raise ContentUnavailable("store down")


class _Access:
    def __init__(self, store: FabricObjectStore) -> None:
        self._store = store

    def store_for(self, task_id: str, scope: str) -> FabricObjectStore:
        return self._store


def _plane(
    store: FabricObjectStore, lane: _Lane, announced: list[Sequence[Any]]
) -> WorkerContentPlane:
    return WorkerContentPlane(
        cast(Any, lane),
        cast(Any, _Access(store)),
        announce=announced.append,
        peer_max_bytes=_LARGE,
    )


def test_a_small_object_is_read_from_a_peer_before_the_store(tmp_path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    reference = store.write("org", b"small")
    lane = _Lane({reference.content_digest: b"small"})

    announced: list[Sequence[Any]] = []

    assert _plane(store, lane, announced).hydrate("tsk-1", reference) == b"small"
    assert lane.peer_reads == [reference]
    # The copy the peer read leaves here is announced for the next reader.
    assert announced == [[("org", reference.content_digest)]]


def test_a_large_object_is_read_from_the_store_not_relayed(tmp_path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    data = b"x" * _LARGE
    reference = store.write("org", data)
    lane = _Lane({reference.content_digest: data})
    announced: list[Sequence[Any]] = []

    assert _plane(store, lane, announced).hydrate("tsk-1", reference) == data
    assert lane.peer_reads == []
    # A store read leaves a copy here for the next reader, and says so.
    assert lane.kept == [reference]
    assert announced == [[("org", reference.content_digest)]]


def test_a_large_object_is_relayed_while_the_store_is_down() -> None:
    data = b"x" * _LARGE
    reference = reference_for("org", data)
    lane = _Lane({reference.content_digest: data})

    assert _plane(_DownStore(), lane, []).hydrate("tsk-1", reference) == data
    assert lane.peer_reads == [reference]


def test_a_large_object_no_peer_holds_fails_as_the_outage_it_is() -> None:
    reference = reference_for("org", b"x" * _LARGE)

    with pytest.raises(ContentUnavailable, match="store down"):
        _plane(_DownStore(), _Lane(), []).hydrate("tsk-1", reference)


def test_a_large_file_no_peer_holds_fails_as_the_outage_it_is(tmp_path) -> None:
    reference = reference_for("org", b"x" * _LARGE)
    dest = tmp_path / "object.bin"

    with pytest.raises(ContentUnavailable, match="store down"):
        _plane(_DownStore(), _Lane(), []).hydrate_to("tsk-1", reference, dest)
    assert not dest.exists()


def test_a_file_hydration_routes_as_a_read_does(tmp_path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    data = b"y" * (2 * _LARGE)
    reference = store.write("org", data)
    lane = _Lane()
    dest = tmp_path / "out" / "object.bin"

    _plane(store, lane, []).hydrate_to("tsk-1", reference, dest)

    assert dest.read_bytes() == data
    assert lane.peer_reads == []
    assert lane.kept == [reference]


def test_a_file_hydration_is_relayed_while_the_store_is_down(tmp_path) -> None:
    data = b"y" * (2 * _LARGE)
    reference = reference_for("org", data)
    lane = _Lane({reference.content_digest: data})
    dest = tmp_path / "object.bin"

    _plane(_DownStore(), lane, []).hydrate_to("tsk-1", reference, dest)

    assert dest.read_bytes() == data


def test_a_written_file_lands_in_the_store_and_a_copy_is_kept(tmp_path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    lane = _Lane()
    announced: list[Sequence[Any]] = []
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"produced bytes")

    reference = _plane(store, lane, announced).write_file("tsk-1", "org", source)

    assert store.hydrate(reference) == b"produced bytes"
    assert lane.kept == [reference]
    assert announced == [[("org", reference.content_digest)]]


class _NoAccess:
    def store_for(self, task_id: str, scope: str) -> FabricObjectStore:
        raise ContentAccessDenied(f"task {task_id} holds no access in scope {scope}")


class _CachedLane(_Lane):
    """A lane whose local copies are a real cache."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.store = WorkerContentCache(root)
        self.local_reads = 0

    def read_local(self, reference: ContentReference) -> bytes:
        self.local_reads += 1
        return self.store.hydrate(reference)

    def copy_local(self, reference: ContentReference, dest: Path) -> None:
        self.local_reads += 1
        self.store.hydrate_to(reference, dest)


def _read(store: TaskContentStore, reference: ContentReference, how: str, tmp_path):
    match how:
        case "hydrate":
            return store.hydrate(reference)
        case "hydrate_to":
            store.hydrate_to(reference, tmp_path / "out.bin")
            return (tmp_path / "out.bin").read_bytes()
        case _:
            return b"".join(store.open(reference))


_READS = ["hydrate", "hydrate_to", "open"]


@pytest.mark.parametrize("how", _READS)
def test_a_copy_here_serves_its_dispatched_scope_without_live_access(
    tmp_path, how: str
) -> None:
    lane = _CachedLane(tmp_path / "cache")
    reference = lane.store.write("org", b"this task's object")
    plane = WorkerContentPlane(cast(Any, lane), cast(Any, _NoAccess()))

    assert _read(plane.for_task("tsk-1", "org"), reference, how, tmp_path) == (
        b"this task's object"
    )


@pytest.mark.parametrize("how", _READS)
def test_a_copy_here_is_never_served_outside_the_dispatched_scope(
    tmp_path, how: str
) -> None:
    lane = _CachedLane(tmp_path / "cache")
    reference = lane.store.write("org", b"another task's object")
    plane = WorkerContentPlane(cast(Any, lane), cast(Any, _NoAccess()))

    with pytest.raises(ContentAccessDenied):
        _read(plane.for_task("tsk-1", "other"), reference, how, tmp_path)
    assert lane.local_reads == 0


@pytest.mark.parametrize("how", ["hydrate", "hydrate_to"])
def test_a_large_object_is_relayed_to_a_task_with_no_live_access(
    tmp_path, how: str
) -> None:
    data = b"x" * _LARGE
    reference = reference_for("org", data)
    lane = _Lane({reference.content_digest: data})
    plane = WorkerContentPlane(
        cast(Any, lane), cast(Any, _NoAccess()), peer_max_bytes=_LARGE
    )

    assert _read(plane.for_task("tsk-1", "org"), reference, how, tmp_path) == data
    assert lane.peer_reads == [reference]


def test_a_copy_that_fails_verification_mid_read_leaves_the_next_open_to_the_store(
    tmp_path,
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    data = os.urandom(4 * STREAM_CHUNK_BYTES)
    reference = store.write("org", data)
    lane = _CachedLane(tmp_path / "cache")
    lane.store.write("org", data)
    (copy,) = [p for p in (tmp_path / "cache").rglob("*") if p.is_file()]
    copy.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
    plane = _plane(store, lane, [])

    with pytest.raises(ContentHydrationError):
        b"".join(plane.open("tsk-1", reference))
    assert not lane.store.holds(reference)
    assert b"".join(plane.open("tsk-1", reference)) == data
