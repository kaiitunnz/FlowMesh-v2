"""A worker puts the published artifacts its task's spec names on disk, and the spec
names their local paths."""

import os
import stat
from pathlib import Path
from typing import Any, cast

import pytest
import requests

from shared.content import (
    TREE_MEDIA_TYPE,
    ContentReference,
    ContentUnavailable,
    SharedFilesystemObjectStore,
    TreeManifest,
    TreeMember,
)
from shared.schemas.event import TaskFailureKind
from shared.tasks import TaskType
from shared.tasks.artifact_binding import ArchiveFacts, ArtifactBinding, ArtifactKind
from shared.tasks.result_binding import ResultBinding
from shared.tasks.specs import EchoSpecStrict, SSHSpecStrict
from tests.worker.content.test_plane_routing import _Access, _Lane
from tests.worker.factories import (
    DEFAULT_WORKER_CONFIG,
    FakeContentPlane,
    make_worker_task_message,
)
from worker.content import WorkerContentCache, WorkerContentPlane
from worker.content.artifact_inputs import ArtifactMaterializer
from worker.content.inputs import TaskInputHydrator
from worker.executors.base_executor import (
    ExecutionError,
    RunSignals,
    TaskCancelledError,
)
from worker.executors.ssh_session import SSHConfig
from worker.executors.ssh_session.inputs import resolve_inputs, stage_inputs_locally

_SCOPE = "local"
_FILES = {"config.json": b"{}", "weights/model.safetensors": b"w" * 100}


def _publish_tree(
    store: SharedFilesystemObjectStore, files: dict[str, bytes] = _FILES
) -> ArtifactBinding:
    members = []
    for path, data in files.items():
        reference = store.write(_SCOPE, data)
        members.append(TreeMember(path, reference.content_digest, len(data)))
    manifest = TreeManifest.of(members)
    return ArtifactBinding(
        name="final_model",
        kind=ArtifactKind.TREE,
        reference=store.write(_SCOPE, manifest.encode(), media_type=TREE_MEDIA_TYPE),
        size_bytes=manifest.size_bytes,
        files=len(members),
    )


def _publish_file(store: SharedFilesystemObjectStore) -> ArtifactBinding:
    reference = store.write(_SCOPE, b"notes")
    return ArtifactBinding(
        name="reports/notes.txt",
        kind=ArtifactKind.FILE,
        reference=reference,
        size_bytes=reference.size_bytes,
    )


def _message(items: list[str], inputs: dict[str, ArtifactBinding]) -> Any:
    return make_worker_task_message(
        EchoSpecStrict(taskType=TaskType.ECHO, data={"type": "list", "items": items}),
        task_type=TaskType.ECHO,
        content_scope=_SCOPE,
        artifact_inputs=inputs,
        dispatch_id="dsp-1",
    )


def test_a_tree_lands_whole_and_read_only_and_two_tokens_share_it(
    tmp_path: Path,
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree = _publish_tree(store)
    plane = FakeContentPlane(store)
    with ArtifactMaterializer(cast(Any, plane), "tsk-1", tmp_path / "in") as held:
        first = held.materialize(tree)
        reads = len(plane.reads)
        again = held.materialize(tree)

    assert first == again == tmp_path / "in" / tree.reference.content_digest
    assert len(plane.reads) == reads
    for path, data in _FILES.items():
        member = first / path
        assert member.read_bytes() == data
        assert not member.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP)
    assert sorted(p.name for p in (tmp_path / "in").iterdir()) == [first.name]


def test_a_file_lands_under_its_own_name(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    notes = _publish_file(store)
    with ArtifactMaterializer(
        cast(Any, FakeContentPlane(store)), "tsk-1", tmp_path / "in"
    ) as materializer:
        path = materializer.materialize(notes)

    assert path.name == "notes.txt" and path.read_bytes() == b"notes"


def test_the_spec_names_each_artifacts_local_path(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree, notes = _publish_tree(store), _publish_file(store)
    msg = _message(
        [
            "flowmesh-artifact:dsp-1.0",
            "flowmesh-artifact:dsp-1.0/config.json",
            "flowmesh-artifact:dsp-1.1",
        ],
        {"dsp-1.0": tree, "dsp-1.1": notes},
    )

    TaskInputHydrator(cast(Any, FakeContentPlane(store))).install_artifacts(
        msg, tmp_path / "in"
    )

    root = (tmp_path / "in").as_posix()
    assert msg.task.spec.data["items"] == [
        f"{root}/{tree.reference.content_digest}",
        f"{root}/{tree.reference.content_digest}/config.json",
        f"{root}/{notes.reference.content_digest}/notes.txt",
    ]


def test_text_that_only_looks_like_a_token_is_left_alone(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    notes = _publish_file(store)
    msg = _message(
        [
            "flowmesh-artifact:0",
            "flowmesh-artifact:dsp-9.0",
            "flowmesh-artifact:dsp-1.0",
        ],
        {"dsp-1.0": notes},
    )

    TaskInputHydrator(cast(Any, FakeContentPlane(store))).install_artifacts(
        msg, tmp_path / "in"
    )

    assert msg.task.spec.data["items"][:2] == [
        "flowmesh-artifact:0",
        "flowmesh-artifact:dsp-9.0",
    ]
    assert msg.task.spec.data["items"][2].endswith("/notes.txt")


def test_a_token_of_its_dispatch_naming_no_input_is_unreadable(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    notes = _publish_file(store)
    msg = _message(["flowmesh-artifact:dsp-1.7"], {"dsp-1.0": notes})

    with pytest.raises(ExecutionError, match="input_unreadable") as raised:
        TaskInputHydrator(cast(Any, FakeContentPlane(store))).install_artifacts(
            msg, tmp_path / "in"
        )
    assert not raised.value.retryable


def test_an_artifact_outside_the_tasks_scope_is_refused(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree = _publish_tree(store)
    elsewhere = tree.model_copy(
        update={
            "reference": tree.reference.model_copy(
                update={"authorization_scope": "other"}
            )
        }
    )
    msg = _message(["flowmesh-artifact:dsp-1.0"], {"dsp-1.0": elsewhere})

    with pytest.raises(ExecutionError, match="input_unreadable") as raised:
        TaskInputHydrator(cast(Any, FakeContentPlane(store))).install_artifacts(
            msg, tmp_path / "in"
        )
    assert not raised.value.retryable
    assert not (tmp_path / "in").exists()


def test_an_outage_mid_tree_names_its_input_and_leaves_no_partial_tree(
    tmp_path: Path,
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree = _publish_tree(store)
    plane = FakeContentPlane(store)

    def fail_after_manifest() -> None:
        if len(plane.reads) == 2:
            plane.error = ContentUnavailable("store down")

    plane.on_read = fail_after_manifest
    msg = _message(["flowmesh-artifact:dsp-1.0"], {"dsp-1.0": tree})

    with pytest.raises(ExecutionError) as raised:
        TaskInputHydrator(cast(Any, plane)).install_artifacts(msg, tmp_path / "in")
    assert raised.value.retryable
    assert raised.value.failure_kind is TaskFailureKind.INPUT_UNAVAILABLE
    assert raised.value.unavailable_inputs == (plane.reads[1],)
    assert plane.reads.count(plane.reads[1]) == 3
    assert not (tmp_path / "in" / tree.reference.content_digest).exists()

    plane.error, plane.on_read = None, None
    with ArtifactMaterializer(cast(Any, plane), "tsk-1", tmp_path / "in") as held:
        path = held.materialize(tree)
    assert (path / "config.json").read_bytes() == b"{}"


def test_a_tree_whose_manifest_is_not_one_is_unreadable(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    bogus = _publish_file(store).model_copy(update={"kind": ArtifactKind.TREE})
    msg = _message(["flowmesh-artifact:dsp-1.0"], {"dsp-1.0": bogus})

    with pytest.raises(ExecutionError, match="input_unreadable"):
        TaskInputHydrator(cast(Any, FakeContentPlane(store))).install_artifacts(
            msg, tmp_path / "in"
        )


class _CacheLane(_Lane):
    """A lane over a real cache, linking its copies as the worker's lane does."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.cache = WorkerContentCache(root)

    def link_local(self, reference: ContentReference, dest: Path) -> None:
        self.cache.link_to(reference, dest)

    def keep_file(self, reference: ContentReference, path: Path) -> Any:
        self.kept.append(reference)
        return self.cache.write_file(reference.authorization_scope, path)


def _inode(cache: WorkerContentCache, reference: ContentReference) -> int:
    path = cache._objects.object_path(
        reference.authorization_scope, reference.content_digest
    )
    return os.stat(path).st_ino


def test_with_a_cache_the_task_reads_the_cached_copy_by_link(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    held = store.write(_SCOPE, b"cached already")
    fetched = store.write(_SCOPE, b"fetched first")
    lane = _CacheLane(tmp_path / "cache")
    lane.cache.write(_SCOPE, b"cached already")
    plane = WorkerContentPlane(cast(Any, lane), cast(Any, _Access(store)))

    plane.materialize_to("tsk-1", held, tmp_path / "held.bin")
    plane.materialize_to("tsk-1", fetched, tmp_path / "fetched.bin")

    assert (tmp_path / "held.bin").stat().st_ino == _inode(lane.cache, held)
    assert (tmp_path / "fetched.bin").stat().st_ino == _inode(lane.cache, fetched)
    assert (tmp_path / "fetched.bin").read_bytes() == b"fetched first"
    assert lane.kept == [fetched]
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == [
        "fetched.bin",
        "held.bin",
    ]


def test_without_a_cache_the_task_holds_its_own_copy(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    reference = store.write(_SCOPE, b"no cache here")
    plane = WorkerContentPlane(None, cast(Any, _Access(store)))

    plane.materialize_to("tsk-1", reference, tmp_path / "copy.bin")

    assert (tmp_path / "copy.bin").read_bytes() == b"no cache here"


def _ssh_message(
    store: SharedFilesystemObjectStore, published: list[ArtifactBinding]
) -> Any:
    msg = make_worker_task_message(
        SSHSpecStrict.model_validate(
            {
                "taskType": "ssh",
                "accessMode": "direct",
                "inputs": [{"stage": "train"}],
            }
        ),
        task_type=TaskType.SSH,
        content_scope=_SCOPE,
    )
    msg.upstream_results = {"train": ResultBinding(task_id="tsk-train")}
    msg.upstream_artifacts = {"tsk-train": published}
    return msg


def test_an_ssh_input_stages_its_upstreams_artifacts_from_the_store_not_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree, notes = _publish_tree(store), _publish_file(store)
    archive = tree.model_copy(
        update={
            "name": "final_model.tar.gz",
            "kind": ArtifactKind.ARCHIVE,
            "of": tree.name,
            "archive": ArchiveFacts(sha256="ab" * 32, crc32=1, size_bytes=9, top="m"),
        }
    )
    msg = _ssh_message(store, [tree, notes, archive])

    def no_root(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a worker reads no artifact over HTTP")

    monkeypatch.setattr(requests, "get", no_root)
    lane = _CacheLane(tmp_path / "cache")
    plane = WorkerContentPlane(cast(Any, lane), cast(Any, _Access(store)))

    TaskInputHydrator(cast(Any, plane)).stage_ssh_inputs(msg, tmp_path / "in")

    staged = msg.staged_input("tsk-train")
    assert staged == (tmp_path / "in" / "tsk-train").as_posix()
    resolved = resolve_inputs(
        msg,
        SSHConfig.from_spec(msg.task.spec, DEFAULT_WORKER_CONFIG),
        tmp_path / "results",
    )
    staging = stage_inputs_locally(resolved, "ssn-12345678", RunSignals())
    root = staging / "tsk-train" / "artifacts"
    assert {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    } == {
        **{f"final_model/{path}": data for path, data in _FILES.items()},
        "reports/notes.txt": b"notes",
    }
    assert len(lane.peer_reads) == len(_FILES) + 2


def test_an_ssh_input_stages_from_the_plane_whatever_its_local_output_holds(
    tmp_path: Path,
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    msg = _ssh_message(store, [_publish_file(store)])
    stale = tmp_path / "results" / "tsk-train" / "artifacts"
    stale.mkdir(parents=True)
    (stale / "partial.bin").write_bytes(b"left by an earlier attempt")

    TaskInputHydrator(cast(Any, FakeContentPlane(store))).stage_ssh_inputs(
        msg, tmp_path / "in"
    )
    resolved = resolve_inputs(
        msg,
        SSHConfig.from_spec(msg.task.spec, DEFAULT_WORKER_CONFIG),
        tmp_path / "results",
    )
    staging = stage_inputs_locally(resolved, "ssn-12345678", RunSignals())

    root = staging / "tsk-train" / "artifacts"
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == [
        "reports",
        "reports/notes.txt",
    ]


def test_staged_ssh_inputs_are_writable_copies_of_the_cache(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    msg = _ssh_message(store, [_publish_tree(store)])
    lane = _CacheLane(tmp_path / "cache")
    plane = WorkerContentPlane(cast(Any, lane), cast(Any, _Access(store)))

    TaskInputHydrator(cast(Any, plane)).stage_ssh_inputs(msg, tmp_path / "in")

    staged = tmp_path / "in" / "tsk-train" / "artifacts" / "final_model" / "config.json"
    assert staged.stat().st_nlink == 1
    assert staged.stat().st_mode & stat.S_IWUSR


def test_a_task_that_ends_stops_reading_its_artifact_inputs(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree = _publish_tree(store)
    plane = FakeContentPlane(store)
    ended = False

    def end_after_manifest() -> None:
        nonlocal ended
        ended = True

    plane.on_read = end_after_manifest
    msg = _message(["flowmesh-artifact:dsp-1.0"], {"dsp-1.0": tree})

    with pytest.raises(TaskCancelledError):
        TaskInputHydrator(
            cast(Any, plane), cancelled=lambda task_id: ended
        ).install_artifacts(msg, tmp_path / "in")
    assert plane.reads == [tree.reference]


@pytest.mark.parametrize("planted", ["link", "foreign"])
def test_a_link_or_foreign_dir_in_place_of_the_inputs_root_is_refused(
    tmp_path: Path, planted: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree = _publish_tree(store)
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "out" / ".artifact-inputs"
    root.parent.mkdir()
    if planted == "link":
        root.symlink_to(outside)
    else:
        root.mkdir()
        monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 1)
    msg = _message(["flowmesh-artifact:dsp-1.0"], {"dsp-1.0": tree})

    with pytest.raises(ExecutionError) as raised:
        TaskInputHydrator(cast(Any, FakeContentPlane(store))).install_artifacts(
            msg, root
        )

    assert not raised.value.retryable
    assert list(outside.iterdir()) == []


def test_the_inputs_land_through_the_held_dir_when_its_name_is_swapped(
    tmp_path: Path,
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree = _publish_tree(store)
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "out" / ".artifact-inputs"
    root.parent.mkdir()
    plane = FakeContentPlane(store)

    def swap() -> None:
        if len(plane.reads) == 1:
            root.rename(tmp_path / "out" / "moved")
            root.symlink_to(outside)

    plane.on_read = swap
    with ArtifactMaterializer(cast(Any, plane), "tsk-1", root) as held:
        held.materialize(tree)

    assert list(outside.iterdir()) == []
    assert (tmp_path / "out" / "moved" / tree.reference.content_digest).is_dir()
