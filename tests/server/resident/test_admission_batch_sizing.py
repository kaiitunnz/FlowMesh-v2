"""The admission slots an invocation reserves come from what its inputs resolved to."""

from server.resident.capacity import default_credit
from server.resident.service import _admitted_batch_size
from server.resident.state import AdmissionProfile
from server.task.v2.representations.operators import ServiceDependency
from shared.inference import InputResolutionBinding


def _dependency(
    batch_size: int | None = 1, max_batch_size: int | None = 1
) -> ServiceDependency:
    return ServiceDependency(
        service_ref="Qwen/Qwen3-4B",
        batch_size=batch_size,
        max_batch_size=max_batch_size,
    )


def _binding(cardinality: int, tokens: int | None = None) -> InputResolutionBinding:
    return InputResolutionBinding(
        source_digest="src",
        resolver_version="1",
        request_digest="req",
        cardinality=cardinality,
        projected_output_tokens=tokens,
    )


def test_a_resolution_sizes_admission_from_the_vector_it_materialized() -> None:
    dependency = _dependency(batch_size=None, max_batch_size=16)
    assert _admitted_batch_size(dependency, _binding(3)) == 3


def test_a_literal_leaf_reserves_the_conversations_it_names() -> None:
    assert _admitted_batch_size(_dependency(batch_size=4, max_batch_size=4), None) == 4


def test_an_unresolved_upstream_leaf_reserves_its_declared_bound() -> None:
    # Reserving the bound can only over-reserve, so an invocation is never admitted
    # against fewer slots than the conversations it goes on to run.
    dependency = _dependency(batch_size=None, max_batch_size=16)
    assert _admitted_batch_size(dependency, None) == 16


def test_an_undeclared_bound_with_no_resolution_sizes_nothing() -> None:
    # Nothing says how many conversations the invocation runs, and admitting it would
    # reserve fewer slots than it occupies.
    dependency = _dependency(batch_size=None, max_batch_size=None)
    assert _admitted_batch_size(dependency, None) is None


def test_a_resolution_below_the_declared_bound_reserves_only_what_it_runs() -> None:
    dependency = _dependency(batch_size=None, max_batch_size=16)
    batch_size = _admitted_batch_size(dependency, _binding(2, tokens=1024))
    assert batch_size is not None
    profile = AdmissionProfile(
        engine_batch_key="k",
        batch_size=batch_size,
        max_output_tokens=1024,
    )
    credit = default_credit(profile)
    assert credit.slots == 2
    assert credit.projected_tokens == 1024
