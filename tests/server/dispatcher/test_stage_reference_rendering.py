"""Stage-reference artifact rendering in the dispatcher.

A cross-stage reference like ``${lora-train.final_lora_archive}`` resolves to a
result field that the typed-result schema now models as an ``ArtifactRef``. The
dispatcher must render it to a URL just as it did for the legacy ``{"path": ...}``
dict, or the downstream task's spec fails validation with the raw object.
"""

from server.dispatcher.base import Dispatcher
from shared.schemas.artifact import ArtifactContext, ArtifactRef
from shared.schemas.result import LoRAResult, ResultEnvelope


def _envelope(result: LoRAResult) -> ResultEnvelope:
    return ResultEnvelope(task_id="tsk-abc", result=result)


def test_render_typed_artifact_ref_to_url() -> None:
    result = LoRAResult(
        final_lora_archive=ArtifactRef(path="final_lora.tar"),
        _artifacts=ArtifactContext(
            base_dir="/data/results/tsk-abc", base_url="http://srv:8000"
        ),
    )
    rendered = Dispatcher._render_artifact_ref(
        result.final_lora_archive, _envelope(result)
    )
    assert rendered == "http://srv:8000/api/v1/results/tsk-abc/files/final_lora.tar"


def test_render_typed_artifact_ref_to_filesystem_path() -> None:
    result = LoRAResult(
        final_lora_archive=ArtifactRef(path="final_lora.tar"),
        _artifacts=ArtifactContext(base_dir="/data/results/tsk-abc"),
    )
    rendered = Dispatcher._render_artifact_ref(
        result.final_lora_archive, _envelope(result)
    )
    assert rendered == "/data/results/tsk-abc/artifacts/final_lora.tar"


def test_render_legacy_dict_ref_still_supported() -> None:
    result = LoRAResult(
        _artifacts=ArtifactContext(
            base_dir="/data/results/tsk-abc", base_url="http://srv:8000"
        ),
    )
    rendered = Dispatcher._render_artifact_ref({"path": "x.bin"}, _envelope(result))
    assert rendered == "http://srv:8000/api/v1/results/tsk-abc/files/x.bin"


def test_render_non_artifact_value_returns_none() -> None:
    result = LoRAResult(
        _artifacts=ArtifactContext(base_dir="/data/results/tsk-abc"),
    )
    assert Dispatcher._render_artifact_ref("plain-text", _envelope(result)) is None
