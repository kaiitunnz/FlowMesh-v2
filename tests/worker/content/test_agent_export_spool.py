"""A completing agent's exports are copied out of its workspace without following a
link, refused when anything could reach past the workspace or alias other bytes, and
land whole under the task's artifacts or not at all."""

import os
import socket
from pathlib import Path

import pytest

from shared.tasks.components.exports import EXPORT_MISSING, EXPORT_UNSAFE, AgentExport
from worker.content.exports import ExportFailed, spool_exports


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


def test_a_missing_path_is_missing(workspace: Path, artifacts: Path) -> None:
    failure = _refused(workspace, artifacts, AgentExport(path="out/absent.md"))
    assert failure.failure.kind == EXPORT_MISSING


def test_a_link_to_outside_the_workspace_is_refused(
    tmp_path: Path, workspace: Path, artifacts: Path
) -> None:
    secret = tmp_path / "secret"
    secret.write_text("key")
    (workspace / "out" / "leak").symlink_to(secret)

    failure = _refused(workspace, artifacts, AgentExport(path="out"))

    assert failure.failure.kind == EXPORT_UNSAFE
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

    assert failure.failure.kind == EXPORT_UNSAFE


def test_a_hard_linked_file_is_refused(workspace: Path, artifacts: Path) -> None:
    os.link(workspace / "notes.txt", workspace / "out" / "alias.txt")

    failure = _refused(workspace, artifacts, AgentExport(path="out"))

    assert failure.failure.kind == EXPORT_UNSAFE
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

    assert failure.failure.kind == EXPORT_UNSAFE


def test_credential_material_inside_a_declared_directory_is_refused(
    workspace: Path, artifacts: Path
) -> None:
    (workspace / ".ssh").mkdir()
    (workspace / ".ssh" / "id_ed25519").write_text("key")

    failure = _refused(workspace, artifacts, AgentExport(path=".ssh/id_ed25519"))

    assert failure.failure.kind == EXPORT_UNSAFE


def test_a_refusal_leaves_the_earlier_copy_in_place(
    workspace: Path, artifacts: Path
) -> None:
    spool_exports(workspace, [AgentExport(path="out")], artifacts)
    (workspace / "out" / "report.md").write_text("changed")
    (workspace / "out" / "loop").symlink_to("loop")

    _refused(workspace, artifacts, AgentExport(path="out"))

    assert _files(artifacts)["out/report.md"] == b"# report\n"
    assert not (artifacts.parent / ".export-staging").exists()
