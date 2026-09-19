"""The sample ratio traces the fraction it names, over real workflow ids.

A trace id is derived from a workflow id rather than drawn at random, so it carries the
uuid4 version and variant bits verbatim and its low 64 bits sit in a narrow band. A
sampler reading those bits directly answers 0% for every ratio below one threshold and
100% above another, which looks like a working knob at 0.0 and 1.0 -- the only two
values a test reaching for round numbers would try.
"""

import pytest

from shared.telemetry.ids import trace_sampled, workflow_to_trace_id_int
from shared.utils.ids import new_workflow_id

_SAMPLE_SIZE = 10000
_TOLERANCE = 0.02


@pytest.fixture(scope="module")
def trace_ids() -> list[int]:
    return [workflow_to_trace_id_int(new_workflow_id()) for _ in range(_SAMPLE_SIZE)]


@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5, 0.6, 0.75, 0.9])
def test_the_traced_fraction_is_the_ratio_asked_for(
    ratio: float, trace_ids: list[int]
) -> None:
    observed = sum(trace_sampled(ratio, t) for t in trace_ids) / len(trace_ids)

    assert (
        abs(observed - ratio) < _TOLERANCE
    ), f"ratio {ratio} traced {observed:.3f} of {_SAMPLE_SIZE} real workflow ids"


def test_the_bounds_are_all_or_nothing(trace_ids: list[int]) -> None:
    assert all(trace_sampled(1.0, t) for t in trace_ids)
    assert not any(trace_sampled(0.0, t) for t in trace_ids)


def test_the_decision_is_stable_for_one_id(trace_ids: list[int]) -> None:
    """Every producer decides independently, so the decision cannot drift per call."""
    for trace_id in trace_ids[:100]:
        assert trace_sampled(0.5, trace_id) == trace_sampled(0.5, trace_id)


def test_a_wider_ratio_never_drops_what_a_narrower_one_kept(
    trace_ids: list[int],
) -> None:
    """Raising the ratio only adds traces, so a workflow cannot fall out of a sample."""
    for trace_id in trace_ids[:2000]:
        if trace_sampled(0.25, trace_id):
            assert trace_sampled(0.5, trace_id)
