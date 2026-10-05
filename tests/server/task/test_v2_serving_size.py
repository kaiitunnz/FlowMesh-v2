"""A leaf's hardware and tensor-parallel size read as one canonical serving size."""

from typing import Any

import pytest

from server.task.v2.representations.serving_size import (
    DEFAULT_SERVING_SIZE,
    ServingSize,
)
from shared.tasks.components.resources import HardwareRequirements


def _size(tp: Any = None, **hardware: Any) -> ServingSize:
    return ServingSize.of(HardwareRequirements.model_validate(hardware), tp)


def test_an_undeclared_size_is_the_default() -> None:
    assert ServingSize.of(None) == DEFAULT_SERVING_SIZE
    assert _size() == DEFAULT_SERVING_SIZE
    assert _size(gpu={}) == DEFAULT_SERVING_SIZE
    assert DEFAULT_SERVING_SIZE == ServingSize(
        cpu=2,
        memory_bytes=4 * 1024**3,
        gpu_type="any",
        gpu_count=1,
        gpu_memory_bytes=None,
        tensor_parallel_size=1,
    )
    assert DEFAULT_SERVING_SIZE.is_default


@pytest.mark.parametrize(
    ("hardware", "field", "value"),
    [
        ({"cpu": 8}, "cpu", 8),
        ({"memory": "16Gi"}, "memory_bytes", 16 * 1024**3),
        ({"gpu": {"type": "H100"}}, "gpu_type", "h100"),
        ({"gpu": {"memory": "80Gi"}}, "gpu_memory_bytes", 80 * 1024**3),
    ],
)
def test_each_omitted_field_takes_the_default(
    hardware: dict[str, Any], field: str, value: Any
) -> None:
    size = _size(**hardware)
    assert getattr(size, field) == value
    assert size.model_copy(update={field: getattr(DEFAULT_SERVING_SIZE, field)}) == (
        DEFAULT_SERVING_SIZE
    )


@pytest.mark.parametrize(
    ("count", "tp", "expected"),
    [
        (None, None, (1, 1)),
        (2, None, (2, 2)),
        (None, 2, (2, 2)),
        (None, "4", (4, 4)),
        (2, 2, (2, 2)),
        (2, 4, (2, 2)),
        (4, 2, (4, 2)),
        (0, None, (1, 1)),
        (None, 0, (1, 1)),
        (2, -1, (2, 2)),
        (None, "{{ a.tp }}", (1, 1)),
        (2, "{{ a.tp }}", (2, 2)),
        (None, "two", (1, 1)),
    ],
)
def test_gpu_count_and_tensor_parallel_size_fill_and_cap_each_other(
    count: int | None, tp: Any, expected: tuple[int, int]
) -> None:
    gpu = {"count": count} if count is not None else {}
    size = _size(tp, gpu=gpu)
    assert (size.gpu_count, size.tensor_parallel_size) == expected


def test_equivalent_spellings_are_one_size() -> None:
    a = _size(None, cpu=2, memory="4096Mi", gpu={"type": "ANY", "count": 1})
    b = _size(1, memory=4 * 1024**3, gpu={"type": "*"})
    c = _size(None, memory="4294967296", gpu={"type": " auto "})
    assert a == b == c == DEFAULT_SERVING_SIZE
    assert _size(gpu={"type": "H100"}) == _size(gpu={"type": " h100 "})
    assert _size(2) == _size(None, gpu={"count": 2}) == _size(4, gpu={"count": 2})


@pytest.mark.parametrize(
    "hardware",
    [
        {"cpu": 4},
        {"memory": "8Gi"},
        {"gpu": {"type": "a100"}},
        {"gpu": {"count": 2}},
        {"gpu": {"count": 2, "type": "a100"}},
        {"gpu": {"memory": "40Gi"}},
    ],
)
def test_every_distinct_requirement_is_a_distinct_size(
    hardware: dict[str, Any],
) -> None:
    assert _size(**hardware) != DEFAULT_SERVING_SIZE
    assert _size(**hardware).key() != DEFAULT_SERVING_SIZE.key()
    assert _size(1, gpu={"count": 2}) != _size(2, gpu={"count": 2})


def test_the_key_is_canonical_and_ends_on_the_gpu_type() -> None:
    assert DEFAULT_SERVING_SIZE.key() == "cpu2,mem4Gi,tp1,gpu1xany"
    size = _size(2, cpu=8, memory="1536Mi", gpu={"count": 2, "type": "H100 NVL"})
    assert size.key() == "cpu8,mem1536Mi,tp2,gpu2xh100 nvl"
    assert _size(memory="1000").key() == "cpu2,mem1000,tp1,gpu1xany"
    assert (
        _size(gpu={"memory": 80 * 1024**3, "type": "a100"}).key()
        == "cpu2,mem4Gi,tp1,gpumem80Gi,gpu1xa100"
    )


def test_the_rendered_hardware_reads_back_as_the_same_size() -> None:
    size = _size(
        2, cpu=8, memory="12Gi", gpu={"count": 2, "type": "h100", "memory": "40Gi"}
    )
    assert size.hardware() == {
        "cpu": 8,
        "memory": "12Gi",
        "gpu": {"type": "h100", "count": 2, "memory": "40Gi"},
    }
    assert _size(2, **size.hardware()) == size
    assert size.hardware(gpu=False)["gpu"] == {"type": "any", "count": 0}
    assert DEFAULT_SERVING_SIZE.hardware()["gpu"] == {"type": "any", "count": 1}


@pytest.mark.parametrize("memory", ["4.5Gi", "lots", "0", 0, -1, 1.5, 4.0 * 1024**3])
def test_memory_placement_cannot_read_is_undeclared(memory: Any) -> None:
    assert _size(memory=memory) == DEFAULT_SERVING_SIZE


@pytest.mark.parametrize("memory", ["4.5Gi", "lots", "0", 0, -1])
def test_gpu_memory_placement_cannot_read_is_undeclared(memory: Any) -> None:
    assert _size(gpu={"memory": memory}) == DEFAULT_SERVING_SIZE


def test_a_non_positive_cpu_is_undeclared() -> None:
    assert _size(cpu=0) == DEFAULT_SERVING_SIZE
    assert _size(cpu=-2) == DEFAULT_SERVING_SIZE


@pytest.mark.parametrize(
    "hardware",
    [
        {"gpu": {"type": "${u.output}"}},
        {"memory": "${u.output}"},
        {"gpu": {"memory": "${u.output}"}},
    ],
)
def test_a_value_that_renders_from_upstream_is_undeclared(
    hardware: dict[str, Any],
) -> None:
    assert _size(**hardware) == DEFAULT_SERVING_SIZE


def test_a_boolean_reads_as_the_number_placement_and_vllm_read() -> None:
    # The local executor reads int(True) as one, and so does placement.
    assert _size(True, gpu={"count": 2}).tensor_parallel_size == 1
