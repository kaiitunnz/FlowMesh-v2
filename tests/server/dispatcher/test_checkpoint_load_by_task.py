"""A checkpoint a task loads by another task's id and file name is that task's
published artifact, named by a token its worker hydrates, never a root URL."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from server.task.models import TaskRecord
from shared.content import TREE_MEDIA_TYPE, reference_for
from shared.tasks import TaskEnvelopeStrict
from shared.tasks.artifact_binding import (
    ARTIFACT_REF_SCHEME,
    ArchiveFacts,
    ArtifactBinding,
    ArtifactInputs,
    ArtifactKind,
)
from shared.tasks.specs import SFTSpecStrict
from tests.server.dispatcher.helpers import make_capturing_dispatcher

_ORG = "org"
_TREE = ArtifactBinding(
    name="final_model",
    kind=ArtifactKind.TREE,
    reference=reference_for(_ORG, b"manifest", media_type=TREE_MEDIA_TYPE),
    size_bytes=8,
)
_ARCHIVE = ArtifactBinding(
    name="final_model.tar.gz",
    kind=ArtifactKind.ARCHIVE,
    reference=_TREE.reference,
    size_bytes=30,
    of="final_model",
    archive=ArchiveFacts(sha256="ab" * 32, crc32=1, size_bytes=30, top="final_model"),
)


class _Runtime:
    def final_artifacts(self, task_id: str) -> Any:
        if task_id != "tsk-train":
            return {}
        return {b.key: b for b in (_TREE, _ARCHIVE)}


def _task(load: dict[str, Any]) -> TaskEnvelopeStrict:
    return TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "resume"},
            "spec": {
                "taskType": "sft",
                "model": {"source": {"type": "huggingface", "identifier": "m"}},
                "checkpoint": {"load": load},
            },
        }
    )


def _render(load: dict[str, Any], org: str = _ORG) -> tuple[Any, ArtifactInputs]:
    dispatcher = make_capturing_dispatcher(runtime=_Runtime())
    inputs = ArtifactInputs()
    rendered = dispatcher._render_checkpoint_load(
        cast(TaskRecord, SimpleNamespace(org_id=org)), _task(load), inputs
    )
    assert isinstance(rendered.spec, SFTSpecStrict) and rendered.spec.checkpoint
    return rendered.spec.checkpoint["load"], inputs


@pytest.mark.parametrize("filename", ["final_model", "final_model.tar.gz"])
def test_a_checkpoint_named_by_task_and_file_loads_the_published_tree(
    filename: str,
) -> None:
    load, inputs = _render(
        {"type": "http", "task_id": "tsk-train", "filename": filename, "base_url": "x"}
    )

    assert load == {"type": "http", "url": f"{ARTIFACT_REF_SCHEME}0"}
    assert list(inputs.bindings.values()) == [_TREE]


@pytest.mark.parametrize(
    ("load", "org"),
    [
        ({"type": "http", "task_id": "tsk-train", "filename": "missing"}, _ORG),
        ({"type": "http", "task_id": "tsk-other", "filename": "final_model"}, _ORG),
        (
            {"type": "http", "task_id": "tsk-train", "filename": "final_model"},
            "elsewhere",
        ),
    ],
)
def test_a_checkpoint_its_task_never_published_in_scope_fails(
    load: dict[str, Any], org: str
) -> None:
    with pytest.raises(ValueError, match="published no artifact named"):
        _render(load, org)


def test_a_checkpoint_with_a_url_is_left_as_it_is() -> None:
    load = {"type": "http", "url": "https://example.com/ckpt.tar.gz"}

    assert _render(load)[0] == load
