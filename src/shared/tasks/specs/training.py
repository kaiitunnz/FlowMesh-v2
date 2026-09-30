from typing import Any, ClassVar, Literal

from pydantic import model_validator

from ..placeholders import TemplateInt
from ..task_type import TaskType
from .common import ModelSpecStrict, ModelSpecTemplate


class TrainingSpecStrict(ModelSpecStrict):
    data: dict[str, Any] | None = None
    training: dict[str, Any] | None = None


class TrainingSpecTemplate(ModelSpecTemplate):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *ModelSpecTemplate.credential_fields,
        "data",
        "training",
    )
    data: dict[str, Any] | None = None
    training: dict[str, Any] | None = None


class SFTSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.SFT]

    checkpoint: dict[str, Any] | None = None


class SFTSpecTemplate(TrainingSpecTemplate):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *TrainingSpecTemplate.credential_fields,
        "checkpoint",
    )
    taskType: Literal[TaskType.SFT]

    checkpoint: dict[str, Any] | None = None


class LoRASFTSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.LORA_SFT]

    lora: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    sloSeconds: int | None = None


class LoRASFTSpecTemplate(TrainingSpecTemplate):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *TrainingSpecTemplate.credential_fields,
        "lora",
        "checkpoint",
    )
    taskType: Literal[TaskType.LORA_SFT]

    lora: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    sloSeconds: TemplateInt | None = None


class PPOSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.PPO]

    reward_model: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None


class PPOSpecTemplate(TrainingSpecTemplate):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *TrainingSpecTemplate.credential_fields,
        "reward_model",
        "generation",
    )
    taskType: Literal[TaskType.PPO]

    reward_model: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None


class DPOSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.DPO]


class DPOSpecTemplate(TrainingSpecTemplate):
    taskType: Literal[TaskType.DPO]


def _require_image_classification_model(model_name: str | None) -> None:
    if not model_name:
        raise ValueError(
            "image_classification_training requires model.source.identifier"
        )


class ImageClassificationTrainingSpecStrict(TrainingSpecStrict):
    taskType: Literal[TaskType.IMAGE_CLASSIFICATION_TRAINING]

    checkpoint: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _require_model(self) -> "ImageClassificationTrainingSpecStrict":
        _require_image_classification_model(self.model_name)
        return self


class ImageClassificationTrainingSpecTemplate(TrainingSpecTemplate):
    credential_fields: ClassVar[tuple[str, ...]] = (
        *TrainingSpecTemplate.credential_fields,
        "checkpoint",
    )
    taskType: Literal[TaskType.IMAGE_CLASSIFICATION_TRAINING]

    checkpoint: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _require_model(self) -> "ImageClassificationTrainingSpecTemplate":
        _require_image_classification_model(self.model_name)
        return self
