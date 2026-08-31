"""Conservative safe-capacity feasibility and per-family replica selection.

These prove a claim never admits past a replica's reported safe slots after accounting
for outstanding credit, that an unhealthy, draining, or deadline-missed candidate is
excluded, and that each selection strategy scores feasible candidates as specified:
best-fit fills the tightest replica, least-load spreads, and round-robin rotates.
"""

from server.resident import (
    AdmissionProfile,
    ClaimCredit,
    ReplicaCapacityReport,
    ReplicaState,
    SafeCapacityVector,
    build_selection_strategy,
    is_feasible,
    residual_after,
)
from server.resident.selection import ReplicaCandidate

_PROFILE = AdmissionProfile(engine_batch_key="k", max_output_tokens=64)
_CREDIT = ClaimCredit(slots=1, projected_tokens=64)


def _report(
    replica_id="rpl-1", *, slots=4, state=ReplicaState.WARM, healthy=True, **kw
):
    return ReplicaCapacityReport(
        replica_id=replica_id,
        incarnation=1,
        report_epoch=1,
        state=state,
        healthy=healthy,
        safe=SafeCapacityVector(admission_slots=slots),
        **kw,
    )


def test_feasible_until_safe_slots_exhausted():
    report = _report(slots=2)
    assert is_feasible(report, _PROFILE, held_slots=0)
    assert is_feasible(report, _PROFILE, held_slots=1)
    assert not is_feasible(report, _PROFILE, held_slots=2)


def test_unhealthy_or_unservable_excluded():
    assert not is_feasible(_report(healthy=False), _PROFILE, held_slots=0)
    assert not is_feasible(_report(state=ReplicaState.DRAINING), _PROFILE, held_slots=0)
    assert not is_feasible(
        _report(state=ReplicaState.MATERIALIZING), _PROFILE, held_slots=0
    )


def test_adapter_slot_constraint():
    profile = AdmissionProfile(engine_batch_key="k", adapter_ref="lora-x")
    assert not is_feasible(_report(adapter_slots_free=0), profile, held_slots=0)
    assert is_feasible(_report(adapter_slots_free=2), profile, held_slots=0)


def test_expired_deadline_excluded():
    profile = AdmissionProfile(
        engine_batch_key="k", deadline_at="2000-01-01T00:00:00+00:00"
    )
    assert not is_feasible(_report(), profile, held_slots=0)


def test_residual_shrinks_with_load():
    report = _report(slots=4)
    assert residual_after(report, held_slots=0, credit=_CREDIT) == 0.75
    assert residual_after(report, held_slots=2, credit=_CREDIT) == 0.25


def _candidates():
    tight = ReplicaCandidate("rpl-tight", _report("rpl-tight", slots=4), held_slots=2)
    loose = ReplicaCandidate("rpl-loose", _report("rpl-loose", slots=4), held_slots=0)
    return [loose, tight]


def test_best_fit_picks_tightest():
    chosen = build_selection_strategy("batch-aware-best-fit").select(
        _candidates(), _CREDIT
    )
    assert chosen is not None and chosen.replica_id == "rpl-tight"


def test_least_load_picks_loosest():
    chosen = build_selection_strategy("least-load").select(_candidates(), _CREDIT)
    assert chosen is not None and chosen.replica_id == "rpl-loose"


def test_round_robin_rotates():
    strategy = build_selection_strategy("round-robin")
    cands = [
        ReplicaCandidate("rpl-a", _report("rpl-a"), 0),
        ReplicaCandidate("rpl-b", _report("rpl-b"), 0),
    ]
    picks = [strategy.select(cands, _CREDIT).replica_id for _ in range(3)]
    assert picks == ["rpl-a", "rpl-b", "rpl-a"]


def test_unknown_strategy_falls_back_to_best_fit():
    strategy = build_selection_strategy("nonesuch")
    assert strategy.name == "batch-aware-best-fit"


def test_empty_candidates_select_none():
    assert build_selection_strategy(None).select([], _CREDIT) is None
