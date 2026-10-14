"""A model a task loads by another task's id is that task's published artifact, named
by a token its worker hydrates, never a local guess or a root URL."""

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
from shared.tasks.specs import InferenceSpecStrict, SFTSpecStrict
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


_LORA = ArtifactBinding(
    name="final_lora",
    kind=ArtifactKind.TREE,
    reference=reference_for(_ORG, b"lora", media_type=TREE_MEDIA_TYPE),
    size_bytes=4,
)


class _Runtime:
    def final_artifacts(self, task_id: str) -> Any:
        published = {
            "tsk-train": (_TREE, _ARCHIVE),
            "tsk-lora": (_LORA,),
            "tsk-both": (_TREE, _LORA),
        }.get(task_id, ())
        return {b.key: b for b in published}


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
@pytest.mark.parametrize("key", ["task_id", "taskId"])
@pytest.mark.parametrize("source", [{"type": "http"}, {"type": "local"}, {}])
def test_a_checkpoint_named_by_task_and_file_loads_the_published_tree(
    filename: str, key: str, source: dict[str, str]
) -> None:
    load, inputs = _render({**source, key: "tsk-train", "filename": filename})

    assert load == {"url": f"{ARTIFACT_REF_SCHEME}0"}
    assert list(inputs.bindings.values()) == [_TREE]


@pytest.mark.parametrize(
    ("producer", "model"), [("tsk-train", _TREE), ("tsk-lora", _LORA)]
)
def test_a_checkpoint_named_by_task_alone_loads_its_model(
    producer: str, model: ArtifactBinding
) -> None:
    load, inputs = _render({"taskId": producer, "subdir": "x"})

    assert load == {"url": f"{ARTIFACT_REF_SCHEME}0", "subdir": "x"}
    assert list(inputs.bindings.values()) == [model]


@pytest.mark.parametrize("producer", ["tsk-both", "tsk-none"])
def test_a_task_without_one_model_must_name_its_file(producer: str) -> None:
    with pytest.raises(ValueError, match="name the artifact to load by filename"):
        _render({"task_id": producer})


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
    with pytest.raises(ValueError, match="published no artifact"):
        _render(load, org)


@pytest.mark.parametrize(
    "load",
    [
        {"type": "http", "url": "https://example.com/ckpt.tar.gz"},
        {"task_id": "tsk-train", "filename": "final_model", "base_url": "https://b"},
    ],
)
def test_a_checkpoint_fetched_from_a_url_is_left_as_it_is(load: dict[str, Any]) -> None:
    assert _render(load)[0] == load


def test_a_lora_adapter_named_by_task_loads_its_published_adapter() -> None:
    dispatcher = make_capturing_dispatcher(runtime=_Runtime())
    inputs = ArtifactInputs()
    task = TaskEnvelopeStrict.model_validate(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "Task",
            "metadata": {"name": "serve"},
            "spec": {
                "taskType": "inference",
                "model": {
                    "source": {"type": "huggingface", "identifier": "m"},
                    "adapters": [
                        {"type": "lora", "name": "a", "task_id": "tsk-lora"},
                        {"type": "lora", "name": "b", "url": "https://e/b.tar.gz"},
                    ],
                },
                "data": {"type": "list", "items": ["hi"]},
            },
        }
    )

    rendered = dispatcher._render_checkpoint_load(
        cast(TaskRecord, SimpleNamespace(org_id=_ORG)), task, inputs
    )

    assert isinstance(rendered.spec, InferenceSpecStrict)
    adapters = rendered.spec.adapters
    assert adapters is not None
    assert [(a.url, a.task_id) for a in adapters] == [
        (f"{ARTIFACT_REF_SCHEME}0", None),
        ("https://e/b.tar.gz", None),
    ]
    assert list(inputs.bindings.values()) == [_LORA]
