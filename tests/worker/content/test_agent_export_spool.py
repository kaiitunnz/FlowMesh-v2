"""A completing agent's exports are copied out of its workspace without following a
link, refused when anything could reach past the workspace or alias other bytes, and
land whole under the task's artifacts or not at all."""

import errno
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from shared.content import SharedFilesystemObjectStore, TreeManifest
from shared.tasks.artifact_binding import ArtifactKind
from shared.tasks.components.exports import AgentExport, ExportFailureKind
from shared.utils import nofollow
from worker.content import exports as exports_module
from worker.content.artifacts import ArtifactPublisher
from worker.content.exports import ExportFailed, spool_exports
from worker.executors.agent_episode_executor import _export


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "state" / "workspace"
    (root / "out" / "nested").mkdir(parents=True)
    (root / "out" / "report.md").write_text("# report\n")
    (root / "out" / "nested" / "table.csv").write_text("a,b\n1,2\n")
    (root / "notes.txt").write_text("notes\n")
    return root


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    return tmp_path / "task" / "artifacts"


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _refused(workspace: Path, artifacts: Path, *exports: AgentExport) -> ExportFailed:
    with pytest.raises(ExportFailed) as raised:
        spool_exports(workspace, exports, artifacts)
    return raised.value


def test_a_file_and_a_directory_copy_exactly_under_their_names(
    workspace: Path, artifacts: Path
) -> None:
    spool_exports(
        workspace,
        [AgentExport(path="out"), AgentExport(path="notes.txt", name="docs/notes.txt")],
        artifacts,
    )

    assert _files(artifacts) == {
        "out/report.md": b"# report\n",
        "out/nested/table.csv": b"a,b\n1,2\n",
        "docs/notes.txt": b"notes\n",
    }


def test_a_later_completion_replaces_an_earlier_copy(
    workspace: Path, artifacts: Path
) -> None:
    spool_exports(workspace, [AgentExport(path="out")], artifacts)
    (workspace / "out" / "nested" / "table.csv").unlink()

    spool_exports(workspace, [AgentExport(path="out")], artifacts)

    assert _files(artifacts) == {"out/report.md": b"# report\n"}


def test_a_re_driven_completion_publishes_exactly_what_it_exported(
    tmp_path: Path, workspace: Path, artifacts: Path
) -> None:
    spool_exports(workspace, [AgentExport(path="out")], artifacts)
    (workspace / "out" / "nested" / "table.csv").unlink()
    (artifacts / "earlier.txt").write_text("left by an earlier attempt")

    spool_exports(workspace, [AgentExport(path="out")], artifacts)
    store = SharedFilesystemObjectStore(tmp_path / "store")
    bindings = ArtifactPublisher(store, "org").publish_final(artifacts, {})

    assert [(b.name, b.kind) for b in bindings] == [("out", ArtifactKind.TREE)]
    manifest = TreeManifest.decode(store.fetch(bindings[0].reference))
    assert [member.path for member in manifest.members] == ["report.md"]


def test_a_missing_path_is_missing(workspace: Path, artifacts: Path) -> None:
    failure = _refused(workspace, artifacts, AgentExport(path="out/absent.md"))
    assert failure.failure.kind == ExportFailureKind.MISSING


def test_a_link_to_outside_the_workspace_is_refused(
    tmp_path: Path, workspace: Path, artifacts: Path
) -> None:
    secret = tmp_path / "secret"
    secret.write_text("key")
    (workspace / "out" / "leak").symlink_to(secret)

    failure = _refused(workspace, artifacts, AgentExport(path="out"))

    assert failure.failure.kind == ExportFailureKind.UNSAFE
    assert "out/leak" in failure.failure.message
    assert not artifacts.exists() or _files(artifacts) == {}


def test_a_declared_path_through_a_linked_directory_is_refused(
    tmp_path: Path, workspace: Path, artifacts: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.md").write_text("x")
    (workspace / "via").symlink_to(outside)

    failure = _refused(workspace, artifacts, AgentExport(path="via/x.md"))

    assert failure.failure.kind == ExportFailureKind.UNSAFE


def test_a_hard_linked_file_is_refused(workspace: Path, artifacts: Path) -> None:
    os.link(workspace / "notes.txt", workspace / "out" / "alias.txt")

    failure = _refused(workspace, artifacts, AgentExport(path="out"))

    assert failure.failure.kind == ExportFailureKind.UNSAFE
    assert "hard link" in failure.failure.message


@pytest.mark.parametrize("special", ["fifo", "socket"])
def test_a_special_file_is_refused(
    workspace: Path, artifacts: Path, special: str
) -> None:
    path = workspace / "out" / "special"
    held = None
    if special == "fifo":
        os.mkfifo(path)
    else:
        held = socket.socket(socket.AF_UNIX)
        held.bind(str(path))
    try:
        failure = _refused(workspace, artifacts, AgentExport(path="out"))
    finally:
        if held is not None:
            held.close()

    assert failure.failure.kind == ExportFailureKind.UNSAFE


def test_credential_material_inside_a_declared_directory_is_refused(
    workspace: Path, artifacts: Path
) -> None:
    (workspace / ".ssh").mkdir()
    (workspace / ".ssh" / "id_ed25519").write_text("key")

    failure = _refused(workspace, artifacts, AgentExport(path=".ssh/id_ed25519"))

    assert failure.failure.kind == ExportFailureKind.UNSAFE


def test_a_refusal_leaves_the_earlier_copy_in_place(
    workspace: Path, artifacts: Path
) -> None:
    spool_exports(workspace, [AgentExport(path="out")], artifacts)
    (workspace / "out" / "report.md").write_text("changed")
    (workspace / "out" / "loop").symlink_to("loop")

    _refused(workspace, artifacts, AgentExport(path="out"))

    assert _files(artifacts)["out/report.md"] == b"# report\n"
    assert not (artifacts.parent / ".export-staging").exists()


def test_a_planted_staging_link_neither_fails_nor_redirects_the_copy(
    workspace: Path, artifacts: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    artifacts.parent.mkdir(parents=True)
    (artifacts.parent / ".export-staging").symlink_to(outside)

    spool_exports(workspace, [AgentExport(path="notes.txt")], artifacts)

    assert _files(artifacts) == {"notes.txt": b"notes\n"}
    assert list(outside.iterdir()) == []


def test_an_artifacts_link_is_refused_and_nothing_lands_through_it(
    workspace: Path, artifacts: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    artifacts.parent.mkdir(parents=True)
    artifacts.symlink_to(outside)

    failure = _refused(workspace, artifacts, AgentExport(path="notes.txt"))

    assert failure.failure.kind == ExportFailureKind.UNSAFE
    assert list(outside.iterdir()) == []


def test_a_tree_deeper_than_a_walk_opens_is_refused(
    workspace: Path, artifacts: Path
) -> None:
    deep = workspace / "deep"
    for level in range(nofollow._MAX_WALK_DEPTH + 2):
        deep = deep / f"d{level}"
    deep.mkdir(parents=True)
    (deep / "lost.txt").write_text("past the bound")

    failure = _refused(workspace, artifacts, AgentExport(path="deep"))

    assert failure.failure.kind == ExportFailureKind.UNSAFE
    assert "deeper than" in failure.failure.message


def test_a_copy_the_disk_cannot_hold_fails_and_leaves_nothing_to_publish(
    tmp_path: Path,
    workspace: Path,
    artifacts: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spool_exports(workspace, [AgentExport(path="out")], artifacts)

    def full(*_args: Any) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(exports_module, "_copy_file", full)
    failure = _export(
        [AgentExport(path="notes.txt")],
        cast(Any, SimpleNamespace(workspace=workspace)),
        artifacts.parent,
    )

    assert failure is not None
    assert failure.kind == ExportFailureKind.UNAVAILABLE
    assert not artifacts.exists()
