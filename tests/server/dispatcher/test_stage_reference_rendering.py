"""Stage-reference artifact rendering in the dispatcher.

A cross-stage reference like ``${lora-train.final_lora_archive}`` resolves to an
``ArtifactRef`` result field. The dispatcher renders it as a token naming the
producer's published artifact, which the task's worker hydrates and replaces with a
local path; a bare ``{"path": ...}`` dict renders the same way.
"""

import hashlib
from typing import Any

import pytest

from server.dispatcher.base import Dispatcher
from server.registries.artifacts import final_set
from shared.content import ContentReference
from shared.schemas.artifact import (
    ArtifactContext,
    ArtifactRef,
    artifact_download_path,
)
from shared.schemas.result import LoRAResult
from shared.tasks.artifact_binding import (
    ArchiveFacts,
    ArtifactBinding,
    ArtifactInputs,
    ArtifactKind,
)


def _ref(data: bytes) -> ContentReference:
    return ContentReference(
        authorization_scope="org",
        content_digest=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


_TREE = ArtifactBinding(
    name="final_lora",
    kind=ArtifactKind.TREE,
    reference=_ref(b"tree"),
    size_bytes=10,
    files=2,
)
_ARCHIVE = ArtifactBinding(
    name="final_lora.tar.gz",
    kind=ArtifactKind.ARCHIVE,
    reference=_ref(b"tree"),
    size_bytes=30,
    of="final_lora",
    archive=ArchiveFacts(sha256="ab" * 32, crc32=7, size_bytes=30, top="final_lora"),
)
_FILE = ArtifactBinding(
    name="x.bin", kind=ArtifactKind.FILE, reference=_ref(b"x"), size_bytes=1
)


_STEP = ArtifactBinding(
    name="checkpoints",
    revision="step-20",
    kind=ArtifactKind.TREE,
    reference=_ref(b"step-20"),
    size_bytes=10,
    files=1,
)


class _Runtime:
    def __init__(self, bindings: dict[str, list[ArtifactBinding]]) -> None:
        self._bindings = bindings

    def artifact_bindings(self, task_id: str) -> dict[tuple[str, str], Any]:
        return {binding.key: binding for binding in self._bindings.get(task_id, [])}

    def final_artifacts(self, task_id: str) -> dict[tuple[str, str], Any]:
        return final_set(
            {binding.key: binding for binding in self._bindings.get(task_id, [])}
        )


def _dispatcher() -> Dispatcher:
    dispatcher = object.__new__(Dispatcher)
    dispatcher._runtime = _Runtime(  # type: ignore[assignment]
        {"tsk-abc": [_TREE, _ARCHIVE, _FILE, _STEP]}
    )
    return dispatcher


def _result(**fields: Any) -> LoRAResult:
    return LoRAResult(_artifacts=ArtifactContext(task_id="tsk-abc"), **fields)


def test_an_archive_ref_renders_as_the_tree_it_archives() -> None:
    result = _result(final_lora_archive=ArtifactRef(path="final_lora.tar.gz"))
    inputs = ArtifactInputs()
    rendered = _dispatcher()._render_artifact_ref(
        result.final_lora_archive, result.artifacts_, inputs
    )
    assert rendered == "flowmesh-artifact:0"
    assert inputs.bindings == {"0": _TREE}


def test_a_path_inside_a_tree_renders_under_its_token() -> None:
    result = _result()
    inputs = ArtifactInputs()
    dispatcher = _dispatcher()
    whole = dispatcher._render_artifact_ref(
        {"path": "final_lora"}, result.artifacts_, inputs
    )
    member = dispatcher._render_artifact_ref(
        {"path": "final_lora/adapter_config.json"}, result.artifacts_, inputs
    )
    assert whole == "flowmesh-artifact:0"
    assert member == "flowmesh-artifact:0/adapter_config.json"
    assert list(inputs.bindings.values()) == [_TREE]


def test_a_file_artifact_renders_its_own_token() -> None:
    inputs = ArtifactInputs()
    rendered = _dispatcher()._render_artifact_ref(
        {"path": "x.bin"}, _result().artifacts_, inputs
    )
    assert rendered == "flowmesh-artifact:0"
    assert inputs.bindings == {"0": _FILE}


def test_a_ref_the_producer_never_published_fails_the_render() -> None:
    with pytest.raises(ValueError, match="published no artifact"):
        _dispatcher()._render_artifact_ref(
            {"path": "missing.bin"}, _result().artifacts_, ArtifactInputs()
        )


def test_a_non_artifact_value_or_context_renders_nothing() -> None:
    dispatcher = _dispatcher()
    assert (
        dispatcher._render_artifact_ref(
            "plain-text", _result().artifacts_, ArtifactInputs()
        )
        is None
    )
    assert (
        dispatcher._render_artifact_ref({"path": "x.bin"}, None, ArtifactInputs())
        is None
    )


def test_a_checkpoints_ref_renders_as_the_last_checkpoint_published() -> None:
    inputs = ArtifactInputs()
    rendered = _dispatcher()._render_artifact_ref(
        {"path": "checkpoints"},
        ArtifactContext(task_id="tsk-abc", checkpoints="step-20"),
        inputs,
    )
    assert rendered == "flowmesh-artifact:0"
    assert inputs.bindings == {"0": _STEP}


def test_a_checkpoints_ref_of_a_task_that_published_none_fails_the_render() -> None:
    with pytest.raises(ValueError, match="published no artifact"):
        _dispatcher()._render_artifact_ref(
            {"path": "checkpoints"}, _result().artifacts_, ArtifactInputs()
        )


def test_a_checkpoints_ref_reads_as_its_revisions_download_path() -> None:
    context = ArtifactContext(task_id="tsk-abc", checkpoints="step-20")
    assert (
        artifact_download_path(
            "tsk-abc", "checkpoints", context.revision_of("checkpoints")
        )
        == "/api/v1/results/tsk-abc/files/checkpoints?revision=step-20"
    )
    assert context.revision_of("final_lora") is None
