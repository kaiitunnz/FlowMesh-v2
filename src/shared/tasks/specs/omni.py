from typing import Any, ClassVar, Literal

from ..task_type import TaskType
from .common import ModelInferSpecStrict, ModelInferSpecTemplate


class OmniSpecStrict(ModelInferSpecStrict):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelInferSpecStrict.credential_fields,
        "omni",
        "storyboard",
    )
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None

    def uses_gpu(self) -> bool:
        return True


class OmniSpecTemplate(ModelInferSpecTemplate):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelInferSpecTemplate.credential_fields,
        "omni",
        "storyboard",
    )
    omni: dict[str, Any] | None = None
    storyboard: dict[str, Any] | None = None

    def uses_gpu(self) -> bool:
        return True


# ── Text-to-Image ────────────────────────────────────────────────────────────


class OmniText2ImageSpecStrict(OmniSpecStrict):
    taskType: Literal[TaskType.OMNI_TEXT2IMAGE]


class OmniText2ImageSpecTemplate(OmniSpecTemplate):
    taskType: Literal[TaskType.OMNI_TEXT2IMAGE]


# ── Text-to-Speech ───────────────────────────────────────────────────────────


class OmniText2SpeechSpecStrict(OmniSpecStrict):
    taskType: Literal[TaskType.OMNI_TEXT2SPEECH]


class OmniText2SpeechSpecTemplate(OmniSpecTemplate):
    taskType: Literal[TaskType.OMNI_TEXT2SPEECH]


# ── Text-to-Audio (BGM) ─────────────────────────────────────────────────────


class OmniText2AudioSpecStrict(OmniSpecStrict):
    taskType: Literal[TaskType.OMNI_TEXT2AUDIO]


class OmniText2AudioSpecTemplate(OmniSpecTemplate):
    taskType: Literal[TaskType.OMNI_TEXT2AUDIO]


# ── Text-to-General (Narration) ──────────────────────────────────────────────


class OmniText2GeneralSpecStrict(OmniSpecStrict):
    taskType: Literal[TaskType.OMNI_TEXT2GENERAL]


class OmniText2GeneralSpecTemplate(OmniSpecTemplate):
    taskType: Literal[TaskType.OMNI_TEXT2GENERAL]
