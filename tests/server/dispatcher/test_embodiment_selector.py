"""Tests for the conservative dispatch-time embodiment selector."""

import pytest

from server.dispatcher.embodiment import (
    EmbodimentSnapshot,
    PrimaryEmbodimentSelector,
    candidate_feasible,
)
from server.task.v2.representations.plan import (
    EpisodeBoundaryKind,
    EpisodeSpec,
    InferenceEmbodimentCandidate,
    InferenceEmbodimentMenu,
    LocalExecutionEnvelope,
    ResidencyIntent,
    ServiceFamilyRequirement,
)
from shared.tasks.specs import InferenceEmbodimentKind

RESIDENT_ID = "phys:tsk-a:resident_served"
LOCAL_ID = "phys:tsk-a:self_contained"


def _resident() -> InferenceEmbodimentCandidate:
    return InferenceEmbodimentCandidate(
        alternative_id=RESIDENT_ID,
        kind=InferenceEmbodimentKind.RESIDENT_SERVED,
        episode=EpisodeSpec(boundary=EpisodeBoundaryKind.SERVICE_ISSUE),
        service_family_requirement=ServiceFamilyRequirement(family="m|chat"),
        residency_intent=ResidencyIntent(service_family="m|chat", conditional=True),
    )


def _local() -> InferenceEmbodimentCandidate:
    return InferenceEmbodimentCandidate(
        alternative_id=LOCAL_ID,
        kind=InferenceEmbodimentKind.SELF_CONTAINED,
        episode=EpisodeSpec(boundary=EpisodeBoundaryKind.TASK),
        local=LocalExecutionEnvelope(executor_key="vllm", gpu_count=1),
    )


def _menu(primary: str, max_batch_size: int = 1) -> InferenceEmbodimentMenu:
    return InferenceEmbodimentMenu(
        contract_fingerprint="fp",
        primary=primary,
        candidates=(_resident(), _local()),
        max_batch_size=max_batch_size,
    )


def _snapshot(
    workers: int = 2,
    resident: bool = True,
    relays: int | None = None,
    slots: int = 8,
) -> EmbodimentSnapshot:
    return EmbodimentSnapshot(
        local_capable_workers=workers,
        relay_capable_workers=workers if relays is None else relays,
        resident_capacity_enabled=resident,
        resident_admission_slots=slots,
    )


class TestMenuShape:
    def test_a_resident_candidate_needs_a_conditional_intent(self) -> None:
        with pytest.raises(ValueError, match="conditional residency intent"):
            InferenceEmbodimentCandidate(
                alternative_id=RESIDENT_ID,
                kind=InferenceEmbodimentKind.RESIDENT_SERVED,
                episode=EpisodeSpec(boundary=EpisodeBoundaryKind.SERVICE_ISSUE),
                service_family_requirement=ServiceFamilyRequirement(family="m|chat"),
                residency_intent=ResidencyIntent(service_family="m|chat"),
            )

    def test_a_local_candidate_carries_no_service_family(self) -> None:
        with pytest.raises(ValueError, match="no service family"):
            InferenceEmbodimentCandidate(
                alternative_id=LOCAL_ID,
                kind=InferenceEmbodimentKind.SELF_CONTAINED,
                episode=EpisodeSpec(boundary=EpisodeBoundaryKind.TASK),
                local=LocalExecutionEnvelope(executor_key="vllm"),
                service_family_requirement=ServiceFamilyRequirement(family="m|chat"),
            )

    def test_the_primary_must_be_a_menu_entry(self) -> None:
        with pytest.raises(ValueError, match="is not a menu entry"):
            _menu("phys:tsk-a:absent")


class TestPrimarySelector:
    @pytest.mark.parametrize("primary", [RESIDENT_ID, LOCAL_ID])
    def test_it_runs_the_declared_primary(self, primary: str) -> None:
        decision = PrimaryEmbodimentSelector()(_menu(primary), _snapshot())
        assert decision.alternative_id == primary
        assert decision.defer_reason is None

    def test_a_primary_the_deployment_rules_out_falls_through(self) -> None:
        # A deployment serving no resident capacity never admits the resident primary,
        # so waiting for it would only fail the task.
        decision = PrimaryEmbodimentSelector()(
            _menu(RESIDENT_ID), _snapshot(resident=False)
        )
        assert decision.alternative_id == LOCAL_ID
        assert decision.defer_reason is None

    def test_a_momentarily_unplaceable_primary_defers_rather_than_switching(
        self,
    ) -> None:
        # Resident capacity is served; no worker can carry the invocation right now.
        decision = PrimaryEmbodimentSelector()(_menu(RESIDENT_ID), _snapshot(relays=0))
        assert decision.alternative_id is None
        assert decision.defer_reason == "resident_served_infeasible"

    def test_it_defers_when_neither_embodiment_can_be_placed(self) -> None:
        decision = PrimaryEmbodimentSelector()(
            _menu(RESIDENT_ID), _snapshot(workers=0, resident=False)
        )
        assert decision.alternative_id is None
        assert decision.defer_reason == "resident_served_unavailable"

    def test_it_defers_when_no_worker_satisfies_the_task(self) -> None:
        decision = PrimaryEmbodimentSelector()(_menu(LOCAL_ID), _snapshot(workers=0))
        assert decision.alternative_id is None
        assert decision.defer_reason == "self_contained_infeasible"


class TestCandidateFeasibility:
    def test_a_local_candidate_needs_only_an_eligible_worker(self) -> None:
        assert candidate_feasible(_local(), _snapshot(resident=False)) is True

    def test_a_resident_candidate_needs_resident_capacity(self) -> None:
        assert candidate_feasible(_resident(), _snapshot(resident=False)) is False
        assert candidate_feasible(_resident(), _snapshot()) is True

    def test_a_resident_candidate_places_on_a_worker_that_only_relays(self) -> None:
        # No worker can run the model locally, but one can carry the invocation: the
        # resident embodiment is placeable and the local one is not.
        relay_only = _snapshot(workers=0, relays=2)
        assert candidate_feasible(_resident(), relay_only) is True
        assert candidate_feasible(_local(), relay_only) is False


class TestBatchAdmissionBound:
    def test_a_batch_within_the_admission_bound_runs_resident(self) -> None:
        decision = PrimaryEmbodimentSelector()(
            _menu(RESIDENT_ID, max_batch_size=8), _snapshot(slots=8)
        )
        assert decision.alternative_id == RESIDENT_ID

    def test_a_batch_past_the_admission_bound_runs_self_contained(self) -> None:
        # One conversation occupies one sequence, so no wait frees enough slots. The
        # menu holds an embodiment that can serve it, so the batch runs rather than
        # waiting for capacity that can never arrive.
        decision = PrimaryEmbodimentSelector()(
            _menu(RESIDENT_ID, max_batch_size=20), _snapshot(slots=8)
        )
        assert decision.alternative_id == LOCAL_ID

    def test_a_batch_no_embodiment_can_serve_says_why(self) -> None:
        decision = PrimaryEmbodimentSelector()(
            _menu(RESIDENT_ID, max_batch_size=20), _snapshot(workers=0, slots=8)
        )
        assert decision.alternative_id is None
        reason = decision.defer_reason or ""
        assert "20 conversations" in reason
        assert "RESIDENT_ADMISSION_SLOTS" in reason

    def test_a_snapshot_reporting_no_bound_does_not_rule_a_batch_out(self) -> None:
        # Admission enforces its own capacity; the scheduler rules a candidate out only
        # on evidence it holds.
        decision = PrimaryEmbodimentSelector()(
            _menu(RESIDENT_ID, max_batch_size=20), _snapshot(slots=0)
        )
        assert decision.alternative_id == RESIDENT_ID
