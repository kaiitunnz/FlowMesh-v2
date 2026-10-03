from typing import Any, get_args

import pytest

from shared.tasks.components.model import (
    ModelConfig,
    ModelConfigTemplate,
    ModelSource,
    ModelSourceTemplate,
)
from shared.tasks.components.resources import (
    GPURequirements,
    HardwareRequirements,
    ResourcesSpec,
)
from shared.tasks.envelope import TaskSpecStrict, TaskSpecTemplate
from shared.tasks.specs import (
    EchoSpecStrict,
    EmbeddingSpecStrict,
    InferenceSpecStrict,
    InferenceSpecTemplate,
    SSHSpecStrict,
    SSHSpecTemplate,
)
from shared.tasks.specs.common import TaskSpecStrictBase, TaskSpecTemplateBase
from shared.tasks.task_type import TaskType
from shared.tasks.worker_message import (
    CPUInfo,
    GpuInfo,
    GpuPlatformInfo,
    MemoryInfo,
    NetworkInfo,
    WorkerHardware,
)
from shared.utils.hardware import gpus_fit_dispatch

_DATA = {"type": "list", "items": ["hi"]}

# Every spec that does not answer for itself and so inherits ``False``. Listed by
# name so that adding a task type forces a deliberate choice rather than silently
# being treated as CPU-only and placed on a worker whose card is held.
_INHERITS_DEFAULT = {
    "AgentSpec",
    "ApiSpec",
    "DataProfilingSpec",
    "DataRetrievalSpec",
    "DevModelSpec",
    "EchoSpec",
    "RagSpec",
}


_SpecBase = type[TaskSpecStrictBase] | type[TaskSpecTemplateBase]


def _union_members(alias: Any) -> tuple[_SpecBase, ...]:
    return get_args(get_args(alias.__value__)[0])


def _model(**kwargs) -> ModelConfig:
    return ModelConfig(source=ModelSource(identifier="org/m"), **kwargs)


def _inference(**kwargs) -> InferenceSpecStrict:
    kwargs.setdefault("model", _model())
    return InferenceSpecStrict(taskType=TaskType.INFERENCE, data=_DATA, **kwargs)


def _embedding(model: ModelConfig) -> EmbeddingSpecStrict:
    return EmbeddingSpecStrict(taskType=TaskType.EMBEDDING, data=_DATA, model=model)


def _ssh(gpu: GPURequirements | None) -> SSHSpecStrict:
    resources = (
        ResourcesSpec(hardware=HardwareRequirements(gpu=gpu))
        if gpu is not None
        else None
    )
    return SSHSpecStrict(taskType=TaskType.SSH, resources=resources)


class TestEveryTaskTypeIsClassified:
    @pytest.mark.parametrize(
        ("alias", "base", "suffix"),
        [
            (TaskSpecStrict, TaskSpecStrictBase, "Strict"),
            (TaskSpecTemplate, TaskSpecTemplateBase, "Template"),
        ],
    )
    def test_only_expected_specs_inherit_the_default(
        self, alias: Any, base: _SpecBase, suffix: str
    ) -> None:
        inherited = {
            spec.__name__.removesuffix(suffix)
            for spec in _union_members(alias)
            if spec.uses_gpu is base.uses_gpu
        }
        assert inherited == _INHERITS_DEFAULT

    def test_strict_and_template_agree_on_which_specs_decide(self) -> None:
        def deciding(alias: Any, base: _SpecBase, suffix: str) -> set[str]:
            return {
                spec.__name__.removesuffix(suffix)
                for spec in _union_members(alias)
                if spec.uses_gpu is not base.uses_gpu
            }

        assert deciding(TaskSpecStrict, TaskSpecStrictBase, "Strict") == deciding(
            TaskSpecTemplate, TaskSpecTemplateBase, "Template"
        )

    def test_cpu_type_is_not_gpu_using(self) -> None:
        assert EchoSpecStrict(taskType=TaskType.ECHO).uses_gpu() is False


class TestInference:
    def test_vllm_backend_uses_gpu(self) -> None:
        assert _inference(model=_model(vllm={"dtype": "auto"})).uses_gpu() is True

    def test_auto_backend_uses_gpu(self) -> None:
        # No vllm/adapters/transformers hints: the runner prefers vLLM.
        assert _inference().uses_gpu() is True

    def test_transformers_defaults_to_gpu(self) -> None:
        spec = _inference(model=_model(transformers={"mode": "text-generation"}))
        assert spec.uses_gpu() is True

    @pytest.mark.parametrize(
        "device_map", ["auto", "balanced", "balanced_low_0", "cuda"]
    )
    def test_non_cpu_device_maps_use_gpu(self, device_map: str) -> None:
        spec = _inference(model=_model(transformers={"device_map": device_map}))
        assert spec.uses_gpu() is True

    def test_explicit_cpu_device_map_does_not(self) -> None:
        spec = _inference(model=_model(transformers={"device_map": "cpu"}))
        assert spec.uses_gpu() is False

    def test_enforce_cpu_wins(self) -> None:
        assert _inference(enforce_cpu=True).uses_gpu() is False

    def test_enforce_cpu_false_is_ignored(self) -> None:
        assert _inference(enforce_cpu=False).uses_gpu() is True

    def test_enforce_cpu_outranks_a_non_cpu_device_map(self) -> None:
        spec = _inference(
            model=_model(transformers={"device_map": "auto"}), enforce_cpu=True
        )
        assert spec.uses_gpu() is False

    def test_enforce_cpu_outranks_a_vllm_model(self) -> None:
        # validate_dispatchable rejects this pairing, but the answer must not
        # depend on validation having run first.
        spec = _inference(model=_model(vllm={"dtype": "auto"}), enforce_cpu=True)
        assert spec.uses_gpu() is False


class TestUnresolvedTemplates:
    def test_placeholder_enforce_cpu_is_not_read_as_cpu(self) -> None:
        # A truthiness test would answer False here and wave the task onto a
        # held card. Only a literal True pins to CPU.
        spec = InferenceSpecTemplate(
            taskType=TaskType.INFERENCE,
            data=_DATA,
            model=ModelConfigTemplate(source=ModelSourceTemplate(identifier="org/m")),
            enforce_cpu="${params.cpu}",
        )
        assert spec.uses_gpu() is True


class TestEmbedding:
    def test_vllm_embedding_uses_gpu(self) -> None:
        assert _embedding(_model(vllm={})).uses_gpu() is True

    def test_transformers_cpu_embedding_does_not(self) -> None:
        assert (
            _embedding(_model(transformers={"device_map": "cpu"})).uses_gpu() is False
        )


class TestSSH:
    def test_declared_count_is_gpu_using(self) -> None:
        assert _ssh(GPURequirements(count=1)).uses_gpu() is True

    def test_memory_without_count_is_still_gpu_using(self) -> None:
        # _resolve_gpu_devices resolves this to one device.
        assert _ssh(GPURequirements(memory="40Gi")).uses_gpu() is True

    def test_no_gpu_block_is_not(self) -> None:
        assert _ssh(None).uses_gpu() is False


class TestSSHGpuSelection:
    @pytest.mark.parametrize(
        ("gpu", "selects"),
        [
            (None, False),
            (GPURequirements(), False),
            (GPURequirements(count=1), True),
            (GPURequirements(type="A100"), True),
            (GPURequirements(memory="40Gi"), True),
        ],
    )
    @pytest.mark.parametrize("spec_cls", [SSHSpecStrict, SSHSpecTemplate])
    def test_only_a_count_type_or_memory_selects_devices(
        self,
        spec_cls: type[SSHSpecStrict] | type[SSHSpecTemplate],
        gpu: GPURequirements | None,
        selects: bool,
    ) -> None:
        resources = ResourcesSpec(hardware=HardwareRequirements(gpu=gpu))
        spec = spec_cls(taskType=TaskType.SSH, resources=resources)
        assert (spec.gpu_selection() is not None) is selects


def _uses_gpu(spec: TaskSpecStrictBase, relays_only: bool) -> bool:
    """Whether the dispatch is withheld from a worker whose only device is held, which
    happens exactly when it uses the GPU."""
    held = GpuInfo(
        index=0,
        name="A100",
        uuid="GPU-0",
        memory_total_bytes=80 * 1024**3,
        gpu_available=False,
    )
    hardware = WorkerHardware(
        cpu=CPUInfo(logical_cores=2, model="x"),
        memory=MemoryInfo(total_bytes=1024**3),
        gpu=GpuPlatformInfo(driver_version=None, cuda_version=None, devices=[held]),
        network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
    )
    return not gpus_fit_dispatch(hardware, spec, relays_only)


class TestDispatchUsesGpu:
    def test_a_relaying_dispatch_uses_no_gpu(self) -> None:
        # A resident service episode or an input preparation loads no local model,
        # whatever its spec would load on its own.
        spec = _inference(model=_model(vllm={"dtype": "auto"}))
        assert _uses_gpu(spec, relays_only=False) is True
        assert _uses_gpu(spec, relays_only=True) is False

    def test_a_declared_gpu_counts_for_any_spec(self) -> None:
        spec = EchoSpecStrict(
            taskType=TaskType.ECHO,
            resources=ResourcesSpec(
                hardware=HardwareRequirements(gpu=GPURequirements(count=1))
            ),
        )
        assert _uses_gpu(spec, relays_only=False) is True

    def test_a_declared_zero_count_exempts_only_a_cpu_spec(self) -> None:
        zero = ResourcesSpec(
            hardware=HardwareRequirements(gpu=GPURequirements(count=0))
        )
        echo = EchoSpecStrict(taskType=TaskType.ECHO, resources=zero)
        assert _uses_gpu(echo, relays_only=False) is False
        assert _uses_gpu(_inference(resources=zero), relays_only=False) is True


def _two_devices(held: int) -> WorkerHardware:
    devices = [
        GpuInfo(
            index=index,
            name="A100",
            uuid=f"GPU-{index}",
            memory_total_bytes=80 * 1024**3,
            gpu_available=index != held,
        )
        for index in range(2)
    ]
    return WorkerHardware(
        cpu=CPUInfo(logical_cores=2, model="x"),
        memory=MemoryInfo(total_bytes=1024**3),
        gpu=GpuPlatformInfo(driver_version=None, cuda_version=None, devices=devices),
        network=NetworkInfo(ip=None, bandwidth_bytes_per_sec=None),
    )


def _with_gpus(count: int | None) -> ResourcesSpec:
    return ResourcesSpec(
        hardware=HardwareRequirements(gpu=GPURequirements(count=count))
    )


class TestBindingWorker:
    """A worker that binds a task's executor to free devices is placed per device."""

    def test_a_binding_worker_fits_a_task_its_free_devices_satisfy(self) -> None:
        spec = _inference(model=_model(vllm={"dtype": "auto"}), resources=_with_gpus(1))
        hardware = _two_devices(held=0)

        assert gpus_fit_dispatch(hardware, spec, False, binds_devices=True)
        assert not gpus_fit_dispatch(hardware, spec, False)

    def test_a_binding_worker_refuses_more_devices_than_are_free(self) -> None:
        spec = _inference(model=_model(vllm={"dtype": "auto"}), resources=_with_gpus(2))

        assert not gpus_fit_dispatch(
            _two_devices(held=0), spec, False, binds_devices=True
        )

    def test_a_task_naming_its_own_devices_keeps_every_device(self) -> None:
        spec = _inference(
            model=_model(vllm={"env_vars": {"CUDA_VISIBLE_DEVICES": "0"}}),
            resources=_with_gpus(1),
        )

        assert spec.pins_cuda_devices()
        assert not gpus_fit_dispatch(
            _two_devices(held=1), spec, False, binds_devices=True
        )
