from typing import Any, ClassVar, Literal

from ..task_type import TaskType
from .common import ModelInferSpecStrict, ModelInferSpecTemplate

# ── Text-to-Image ────────────────────────────────────────────────────────────


class OmniText2ImageSpecStrict(ModelInferSpecStrict):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelInferSpecStrict.credential_fields,
        "omni",
        "storyboard",
    )
    taskType: Literal[TaskType.OMNI_TEXT2IMAGE]
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None


class OmniText2ImageSpecTemplate(ModelInferSpecTemplate):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelInferSpecTemplate.credential_fields,
        "omni",
        "storyboard",
    )
    taskType: Literal[TaskType.OMNI_TEXT2IMAGE]
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None


# ── Text-to-Speech ───────────────────────────────────────────────────────────


class OmniText2SpeechSpecStrict(ModelInferSpecStrict):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelInferSpecStrict.credential_fields,
        "omni",
        "storyboard",
    )
    taskType: Literal[TaskType.OMNI_TEXT2SPEECH]
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None


class OmniText2SpeechSpecTemplate(ModelInferSpecTemplate):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelInferSpecTemplate.credential_fields,
        "omni",
        "storyboard",
    )
    taskType: Literal[TaskType.OMNI_TEXT2SPEECH]
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None


# ── Text-to-Audio (BGM) ─────────────────────────────────────────────────────


class OmniText2AudioSpecStrict(ModelInferSpecStrict):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelInferSpecStrict.credential_fields,
        "omni",
        "storyboard",
    )
    taskType: Literal[TaskType.OMNI_TEXT2AUDIO]
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None


class OmniText2AudioSpecTemplate(ModelInferSpecTemplate):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelInferSpecTemplate.credential_fields,
        "omni",
        "storyboard",
    )
    taskType: Literal[TaskType.OMNI_TEXT2AUDIO]
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None


# ── Text-to-General (Narration) ──────────────────────────────────────────────


class OmniText2GeneralSpecStrict(ModelInferSpecStrict):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelInferSpecStrict.credential_fields,
        "omni",
        "storyboard",
    )
    taskType: Literal[TaskType.OMNI_TEXT2GENERAL]
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None


class OmniText2GeneralSpecTemplate(ModelInferSpecTemplate):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelInferSpecTemplate.credential_fields,
        "omni",
        "storyboard",
    )
    taskType: Literal[TaskType.OMNI_TEXT2GENERAL]
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None
