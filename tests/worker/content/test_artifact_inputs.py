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
from worker.executors.base_executor import ExecutionError, RunSignals
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
    )


def test_a_tree_lands_whole_and_read_only_and_two_tokens_share_it(
    tmp_path: Path,
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree = _publish_tree(store)
    plane = FakeContentPlane(store)
    materializer = ArtifactMaterializer(cast(Any, plane), "tsk-1", tmp_path / "in")

    first = materializer.materialize(tree)
    reads = len(plane.reads)
    again = materializer.materialize(tree)

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
    materializer = ArtifactMaterializer(
        cast(Any, FakeContentPlane(store)), "tsk-1", tmp_path / "in"
    )

    path = materializer.materialize(notes)

    assert path.name == "notes.txt" and path.read_bytes() == b"notes"


def test_the_spec_names_each_artifacts_local_path(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree, notes = _publish_tree(store), _publish_file(store)
    msg = _message(
        [
            "flowmesh-artifact:0",
            "flowmesh-artifact:0/config.json",
            "flowmesh-artifact:1",
        ],
        {"0": tree, "1": notes},
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
    msg = _message(["flowmesh-artifact:0"], {"0": elsewhere})

    with pytest.raises(ExecutionError, match="input_unreadable") as raised:
        TaskInputHydrator(cast(Any, FakeContentPlane(store))).install_artifacts(
            msg, tmp_path / "in"
        )
    assert not raised.value.retryable
    assert not (tmp_path / "in").exists()


def test_an_outage_mid_tree_is_retryable_and_leaves_no_partial_tree(
    tmp_path: Path,
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    tree = _publish_tree(store)
    plane = FakeContentPlane(store)

    def fail_after_manifest() -> None:
        if len(plane.reads) == 2:
            plane.error = ContentUnavailable("store down")

    plane.on_read = fail_after_manifest
    msg = _message(["flowmesh-artifact:0"], {"0": tree})

    with pytest.raises(ExecutionError) as raised:
        TaskInputHydrator(cast(Any, plane)).install_artifacts(msg, tmp_path / "in")
    assert raised.value.retryable
    assert not (tmp_path / "in" / tree.reference.content_digest).exists()

    plane.error, plane.on_read = None, None
    path = ArtifactMaterializer(cast(Any, plane), "tsk-1", tmp_path / "in").materialize(
        tree
    )
    assert (path / "config.json").read_bytes() == b"{}"


def test_a_tree_whose_manifest_is_not_one_is_unreadable(tmp_path: Path) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    bogus = _publish_file(store).model_copy(update={"kind": ArtifactKind.TREE})
    msg = _message(["flowmesh-artifact:0"], {"0": bogus})

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

    TaskInputHydrator(cast(Any, plane)).stage_ssh_inputs(
        msg, tmp_path / "in", tmp_path / "results"
    )

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


def test_an_ssh_input_whose_upstream_ran_here_is_not_staged_again(
    tmp_path: Path,
) -> None:
    store = SharedFilesystemObjectStore(tmp_path / "shared")
    msg = _ssh_message(store, [_publish_file(store)])
    (tmp_path / "results" / "tsk-train").mkdir(parents=True)
    plane = FakeContentPlane(store)

    TaskInputHydrator(cast(Any, plane)).stage_ssh_inputs(
        msg, tmp_path / "in", tmp_path / "results"
    )

    assert msg.staged_input("tsk-train") is None and plane.reads == []
