# ruff: noqa: E402
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip(
    "torch", reason="torch not installed (needs --extra inference)"
)

from shared.tasks.components.model import ModelConfig, ModelSource
from shared.tasks.specs import InferenceSpecStrict
from shared.tasks.task_type import TaskType
from tests.worker.factories import DEFAULT_WORKER_CONFIG
from worker.executors import transformers_executor
from worker.executors.base_executor import ExecutionError
from worker.executors.transformers_executor import HFTransformersExecutor


@pytest.fixture
def executor() -> HFTransformersExecutor:
    return HFTransformersExecutor(DEFAULT_WORKER_CONFIG)


@pytest.fixture
def cuda_available():
    with patch.object(torch.cuda, "is_available", return_value=True):
        yield


@pytest.mark.usefixtures("cuda_available")
class TestEnforceCpu:
    @pytest.mark.parametrize(
        "cfg", [{}, {"device_map": "auto"}, {"device_map": "cuda"}]
    )
    def test_enforce_cpu_wins(
        self, executor: HFTransformersExecutor, cfg: dict
    ) -> None:
        assert executor._pick_device(cfg, enforce_cpu=True) == "cpu"

    def test_without_enforce_cpu_a_gpu_is_preferred(
        self, executor: HFTransformersExecutor
    ) -> None:
        assert executor._pick_device({}) == "cuda"

    def test_device_map_is_honoured(self, executor: HFTransformersExecutor) -> None:
        assert executor._pick_device({"device_map": "auto"}) == "auto"
        assert executor._pick_device({"device_map": "cpu"}) == "cpu"


class TestEnforceCpuWithoutGpu:
    def test_enforce_cpu_is_a_noop_when_no_gpu_exists(
        self, executor: HFTransformersExecutor
    ) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            assert executor._pick_device({}, enforce_cpu=True) == "cpu"
            assert executor._pick_device({}) == "cpu"


def _spec(enforce_cpu: bool | None, ident: str = "org/model") -> InferenceSpecStrict:
    return InferenceSpecStrict(
        taskType=TaskType.INFERENCE,
        model=ModelConfig(source=ModelSource(identifier=ident)),
        data={"type": "list", "items": ["hi"]},
        enforce_cpu=enforce_cpu,
    )


@pytest.mark.usefixtures("cuda_available")
class TestEnforceCpuPlacesTheModel:
    @pytest.fixture
    def loaded(self) -> Iterator[MagicMock]:
        """The mocked ``from_pretrained``; its return value records where the model
        was moved."""
        with (
            patch.object(transformers_executor, "AutoTokenizer"),
            patch.object(transformers_executor, "AutoModelForCausalLM") as model_cls,
        ):
            yield model_cls.from_pretrained

    def test_an_enforce_cpu_task_loads_on_the_cpu(
        self, executor: HFTransformersExecutor, loaded: MagicMock
    ) -> None:
        executor._ensure_model(_spec(enforce_cpu=True))

        loaded.return_value.to.assert_called_once_with("cpu")
        assert executor._device == "cpu"

    def test_a_warm_gpu_model_is_not_reused_for_an_enforce_cpu_task(
        self, executor: HFTransformersExecutor, loaded: MagicMock
    ) -> None:
        executor._ensure_model(_spec(enforce_cpu=None))
        executor._ensure_model(_spec(enforce_cpu=True))

        assert [c.args for c in loaded.return_value.to.call_args_list] == [
            ("cuda",),
            ("cpu",),
        ]
        assert loaded.call_count == 2
        assert executor._device == "cpu"

    @pytest.mark.parametrize(
        "next_spec",
        [_spec(enforce_cpu=True), _spec(enforce_cpu=None, ident="org/other")],
        ids=["same-model-on-cpu", "other-model"],
    )
    def test_a_failed_load_leaves_no_model_to_reuse(
        self,
        executor: HFTransformersExecutor,
        loaded: MagicMock,
        next_spec: InferenceSpecStrict,
    ) -> None:
        warm, fresh = MagicMock(name="warm"), MagicMock(name="fresh")
        loaded.return_value = warm
        executor._ensure_model(_spec(enforce_cpu=None))

        loaded.side_effect = RuntimeError("out of memory")
        with pytest.raises(ExecutionError):
            executor._ensure_model(next_spec)
        assert executor._model is None

        loaded.side_effect = None
        loaded.return_value = fresh
        executor._ensure_model(next_spec)
        assert executor._model is fresh
