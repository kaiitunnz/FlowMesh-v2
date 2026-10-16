"""A completed artifact file is stored and cached as the file itself, while a task's
writable input keeps a copy of its own."""

import os
import time
from pathlib import Path

import pytest

from shared.content import (
    BACKEND_FILESYSTEM,
    ContentOperationKind,
    ContentReference,
    ContentStoreAccess,
    ContentStoreAccessGrant,
    ObjectStoreConfig,
    ScopedContentCredential,
    reference_for,
)
from worker.content import (
    ContentAccessRegistry,
    ContentLaneHost,
    WorkerContentCache,
    WorkerContentPlane,
)
from worker.content import store as store_module

_SCOPE = "tenant-a"


def _plane(tmp_path: Path) -> tuple[WorkerContentPlane, WorkerContentCache]:
    registry = ContentAccessRegistry(
        ObjectStoreConfig(
            backend=BACKEND_FILESYSTEM, filesystem_root=tmp_path / "shared"
        ),
        arrival_wait_sec=0.1,
    )
    registry.accept(
        ContentStoreAccess(
            grant=ContentStoreAccessGrant(
                grant_id="csg-1",
                task_id="tsk-1",
                authorization_scope=_SCOPE,
                subject="wkr-1",
                subject_generation=1,
                operations=(ContentOperationKind.READ, ContentOperationKind.WRITE),
                backend_policy_version="test-1",
                expires_at_epoch=time.time() + 900.0,
            ),
            credential=ScopedContentCredential(material={"token": "opaque"}),
        )
    )
    cache = WorkerContentCache(tmp_path / "cache")
    lane = ContentLaneHost(
        store=cache,
        push_frame=lambda frame: None,
        request_grant=lambda reference, task_id, peer_only, request_id: None,
        worker_id="wkr-1",
        generation=1,
    )
    return WorkerContentPlane(lane, registry), cache


def _cached(cache: WorkerContentCache, reference: ContentReference) -> Path:
    return cache._objects.object_path(_SCOPE, reference.content_digest)


def _output(tmp_path: Path, data: bytes, name: str = "weights") -> Path:
    path = tmp_path / "out" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _write(plane: WorkerContentPlane, path: Path) -> ContentReference:
    with path.open("rb") as opened:
        return plane.for_task("tsk-1", _SCOPE).write_completed(_SCOPE, opened)


def test_a_completed_file_is_cached_as_itself(tmp_path: Path) -> None:
    plane, cache = _plane(tmp_path)
    output = _output(tmp_path, b"weights")

    reference = _write(plane, output)

    assert reference == reference_for(_SCOPE, b"weights")
    assert os.path.samestat(_cached(cache, reference).stat(), output.stat())


def test_a_completed_file_that_cannot_be_linked_is_cached_as_a_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_module, "link_opened", lambda *args: False)
    plane, cache = _plane(tmp_path)
    output = _output(tmp_path, b"weights")

    reference = _write(plane, output)

    cached = _cached(cache, reference)
    assert cached.read_bytes() == b"weights"
    assert not os.path.samestat(cached.stat(), output.stat())


def test_a_copy_already_cached_stays_for_the_same_bytes(tmp_path: Path) -> None:
    plane, cache = _plane(tmp_path)
    earlier = cache.write(_SCOPE, b"weights")
    inode = _cached(cache, earlier).stat().st_ino

    _write(plane, _output(tmp_path, b"weights"))

    assert _cached(cache, earlier).stat().st_ino == inode


def test_a_file_not_the_size_stored_is_not_cached_and_leaves_the_copy_there(
    tmp_path: Path,
) -> None:
    _, cache = _plane(tmp_path)
    reference = cache.write(_SCOPE, b"weights")
    inode = _cached(cache, reference).stat().st_ino
    output = _output(tmp_path, b"weights and more")

    with output.open("rb") as opened:
        assert cache.link_completed(reference, opened)

    assert _cached(cache, reference).stat().st_ino == inode
    assert _cached(cache, reference).read_bytes() == b"weights"
    assert list(cache.staging_dir(_SCOPE).iterdir()) == []


def test_a_writable_input_hydrated_from_the_store_keeps_its_own_copy(
    tmp_path: Path,
) -> None:
    plane, cache = _plane(tmp_path)
    reference = plane.write("tsk-1", _SCOPE, b"input")
    cache.drop(reference)
    dest = tmp_path / "input"

    plane.hydrate_to("tsk-1", reference, dest)
    dest.write_bytes(b"edited by its task")

    assert _cached(cache, reference).read_bytes() == b"input"
