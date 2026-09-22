import importlib
from collections.abc import Iterator, Mapping
from typing import overload

from shared.tasks.executor_key import ExecutorKey

from .base_executor import Executor

_IMPORT_ERRORS: dict[str, str] = {}


def _import_executor(name: str, module: str) -> type[Executor] | None:
    try:
        pkg = importlib.import_module(module, package=__package__)
        if issubclass(cls := getattr(pkg, name), Executor):
            return cls
        error = f"{name} is not a subclass of Executor"
    except Exception as exc:
        error = str(exc)
    _IMPORT_ERRORS[name] = error
    return None


EXECUTOR_MODULES: dict[ExecutorKey, tuple[str, str]] = {
    ExecutorKey.VLLM: ("VLLMExecutor", ".vllm_executor"),
    ExecutorKey.VLLM_LORA: ("VLLMLoRAExecutor", ".vllm_lora_executor"),
    ExecutorKey.VLLM_EMBEDDING: ("VLLMEmbeddingExecutor", ".vllm_embedding_executor"),
    ExecutorKey.VLLM_SERVE: ("VLLMServeExecutor", ".vllm_serve_executor"),
    ExecutorKey.DEV_MODEL: ("DevModelExecutor", ".dev_model_executor"),
    ExecutorKey.PPO: ("PPOExecutor", ".ppo_executor"),
    ExecutorKey.DPO: ("DPOExecutor", ".dpo_executor"),
    ExecutorKey.SFT: ("SFTExecutor", ".sft_executor"),
    ExecutorKey.LORA_SFT: ("LoRASFTExecutor", ".lora_sft_executor"),
    ExecutorKey.IMAGE_CLASSIFICATION_TRAINING: (
        "ImageClassificationTrainingExecutor",
        ".image_classification_executor",
    ),
    ExecutorKey.DEFAULT: ("HFTransformersExecutor", ".transformers_executor"),
    ExecutorKey.RAG: ("RAGExecutor", ".rag_executor"),
    ExecutorKey.AGENT_EPISODE: ("AgentEpisodeExecutor", ".agent_episode_executor"),
    ExecutorKey.SERVICE_LEAF: ("ServiceLeafExecutor", ".service_leaf_executor"),
    ExecutorKey.ECHO: ("EchoExecutor", ".echo_executor"),
    ExecutorKey.DATA_PROFILING: ("DataProfilingExecutor", ".data_profiling_executor"),
    ExecutorKey.DATA_RETRIEVAL: ("DataRetrievalExecutor", ".data_retrieval_executor"),
    ExecutorKey.DIFFUSERS: ("DiffusersExecutor", ".diffusers_executor"),
    ExecutorKey.API: ("APIExecutor", ".api_executor"),
    ExecutorKey.SSH: ("SSHExecutor", ".ssh_executor"),
    ExecutorKey.OMNI_TEXT2IMAGE: (
        "OmniText2ImageExecutor",
        ".omni_text2image_executor",
    ),
    ExecutorKey.OMNI_TEXT2SPEECH: (
        "OmniText2SpeechExecutor",
        ".omni_text2speech_executor",
    ),
    ExecutorKey.OMNI_TEXT2AUDIO: (
        "OmniText2AudioExecutor",
        ".omni_text2audio_executor",
    ),
    ExecutorKey.OMNI_TEXT2GENERAL: (
        "OmniText2GeneralExecutor",
        ".omni_text2general_executor",
    ),
}


class ExecutorRegistry(Mapping[ExecutorKey, type[Executor] | None]):
    def __init__(self) -> None:
        self._executors: dict[ExecutorKey, type[Executor] | None] = {}

    def __getitem__(self, key: ExecutorKey) -> type[Executor] | None:
        if key in self._executors:
            return self._executors[key]
        if key not in EXECUTOR_MODULES:
            raise KeyError(f"Executor {key!r} not found in registry")
        name, module = EXECUTOR_MODULES[key]
        executor = _import_executor(name, module)
        self._executors[key] = executor
        return executor

    def __iter__(self) -> Iterator[ExecutorKey]:
        return iter(EXECUTOR_MODULES)

    def __len__(self) -> int:
        return len(EXECUTOR_MODULES)


EXECUTOR_REGISTRY = ExecutorRegistry()


@overload
def get_executor_class_name[T](key: ExecutorKey, default: T) -> str | T: ...
@overload
def get_executor_class_name(key: ExecutorKey, default: None = None) -> str | None: ...
def get_executor_class_name[T](
    key: ExecutorKey, default: T | None = None
) -> str | T | None:
    return mod[0] if (mod := EXECUTOR_MODULES.get(key)) else default


IMPORT_ERRORS: dict[str, str] = _IMPORT_ERRORS
