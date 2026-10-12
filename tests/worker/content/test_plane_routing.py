"""Where the worker's content plane reads an object from, and what it keeps of it."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from shared.content import (
    ContentHydrationError,
    ContentReference,
    ContentUnavailable,
    FabricObjectStore,
    SharedFilesystemObjectStore,
    reference_for,
)
from worker.content import WorkerContentPlane

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

    assert _plane(store, lane, []).hydrate("tsk-1", reference) == b"small"
    assert lane.peer_reads == [reference]


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

    with pytest.raises(ContentHydrationError):
        _plane(_DownStore(), _Lane(), []).hydrate("tsk-1", reference)


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
