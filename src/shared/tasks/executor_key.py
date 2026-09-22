from enum import StrEnum

from .envelope import TaskSpecStrict, TaskSpecTemplate
from .specs.inference import (
    InferenceBackend,
    InferenceSpecStrict,
    InferenceSpecTemplate,
)
from .specs.misc import EmbeddingSpecStrict, EmbeddingSpecTemplate
from .task_type import TaskType


class ExecutorKey(StrEnum):
    """The executor a worker runs a task on."""

    DEFAULT = "default"
    VLLM = "vllm"
    VLLM_LORA = "vllm_lora"
    VLLM_EMBEDDING = "vllm_embedding"
    VLLM_SERVE = "vllm_serve"
    DEV_MODEL = "dev_model"
    PPO = "ppo"
    DPO = "dpo"
    SFT = "sft"
    LORA_SFT = "lora_sft"
    IMAGE_CLASSIFICATION_TRAINING = "image_classification_training"
    RAG = "rag"
    AGENT_EPISODE = "agent_episode"
    SERVICE_LEAF = "service_leaf"
    ECHO = "echo"
    DATA_PROFILING = "data_profiling"
    DATA_RETRIEVAL = "data_retrieval"
    DIFFUSERS = "diffusers"
    API = "api"
    SSH = "ssh"
    OMNI_TEXT2IMAGE = "omni_text2image"
    OMNI_TEXT2SPEECH = "omni_text2speech"
    OMNI_TEXT2AUDIO = "omni_text2audio"
    OMNI_TEXT2GENERAL = "omni_text2general"


_EXECUTOR_BY_TASK_TYPE: dict[TaskType, ExecutorKey] = {
    TaskType.DIFFUSION: ExecutorKey.DIFFUSERS,
    TaskType.SERVE: ExecutorKey.VLLM_SERVE,
    TaskType.AGENT: ExecutorKey.AGENT_EPISODE,
}


def resolve_executor_key(
    spec: TaskSpecStrict | TaskSpecTemplate,
) -> ExecutorKey | None:
    """The executor a spec runs on, or None when a template placeholder decides it."""
    match spec:
        case InferenceSpecStrict() | InferenceSpecTemplate():
            if isinstance(spec.enforce_cpu, str):
                return None
            if spec.backend() is InferenceBackend.TRANSFORMERS:
                return ExecutorKey.DEFAULT
            if spec.adapters:
                return ExecutorKey.VLLM_LORA
            return ExecutorKey.VLLM
        case EmbeddingSpecStrict() | EmbeddingSpecTemplate():
            if (model := spec.model) and model.vllm is not None:
                return ExecutorKey.VLLM_EMBEDDING
            return ExecutorKey.DEFAULT
    task_type = TaskType(spec.taskType)
    return _EXECUTOR_BY_TASK_TYPE.get(task_type) or ExecutorKey(task_type.value)


__all__ = ["ExecutorKey", "resolve_executor_key"]
