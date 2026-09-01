"""The derived reachability view: classified transitions, TTLs, backoff, fencing."""

import pytest

from server.network import NetworkReachabilityView, ReachabilityBounds
from server.network.state import (
    PolicyClass,
    ReachabilityState,
    RouteObservation,
    RouteObservationOutcome,
    Transport,
)

_ORIGIN = "rog-1"
_TARGET = "nde-1"
_INC = 1
_GEN = 0


def _view() -> NetworkReachabilityView:
    return NetworkReachabilityView(
        ReachabilityBounds(
            positive_ttl_sec=10.0,
            negative_ttl_sec=5.0,
            backoff_base_sec=1.0,
            backoff_max_sec=4.0,
        )
    )


def _key(
    transport: Transport = Transport.WORKER_DIRECT, *, incarnation: int = _INC
) -> dict:
    return {
        "origin_id": _ORIGIN,
        "policy_class": PolicyClass.DEFAULT,
        "target_node_id": _TARGET,
        "incarnation": incarnation,
        "listener_generation": _GEN,
        "transport": transport,
    }


def _obs(
    outcome: RouteObservationOutcome,
    transport: Transport = Transport.WORKER_DIRECT,
    *,
    incarnation: int = _INC,
) -> RouteObservation:
    return RouteObservation(
        origin_id=_ORIGIN,
        policy_class=PolicyClass.DEFAULT,
        target_node_id=_TARGET,
        incarnation=incarnation,
        listener_generation=_GEN,
        transport=transport,
        outcome=outcome,
    )


def test_absent_key_is_unknown_and_not_allocated() -> None:
    view = _view()
    assert view.state_for(**_key(), now=0.0) is ReachabilityState.UNKNOWN
    assert view.entries() == []  # demand-paged: a query allocates nothing


def test_optimistic_then_verified_with_positive_ttl() -> None:
    view = _view()
    view.mark_optimistic(**_key(), now=0.0)
    assert view.state_for(**_key(), now=0.0) is ReachabilityState.OPTIMISTIC
    view.observe(_obs(RouteObservationOutcome.VERIFIED), now=0.0)
    assert view.state_for(**_key(), now=0.0) is ReachabilityState.VERIFIED
    # Past the positive TTL it needs re-verification.
    assert view.state_for(**_key(), now=100.0) is ReachabilityState.UNKNOWN


@pytest.mark.parametrize(
    "outcome",
    [
        RouteObservationOutcome.DNS_FAILURE,
        RouteObservationOutcome.CONNECT_FAILURE,
        RouteObservationOutcome.TLS_FAILURE,
        RouteObservationOutcome.ROUTE_FAILURE,
        RouteObservationOutcome.TIMEOUT,
    ],
)
def test_path_failures_demote(outcome: RouteObservationOutcome) -> None:
    view = _view()
    view.observe(_obs(outcome), now=0.0)
    assert view.state_for(**_key(), now=0.0) is ReachabilityState.DEMOTED
    # Backoff holds it demoted, then it cools to UNKNOWN for an optimistic retry.
    assert view.state_for(**_key(), now=0.5) is ReachabilityState.DEMOTED
    assert view.state_for(**_key(), now=2.0) is ReachabilityState.UNKNOWN


@pytest.mark.parametrize(
    "outcome",
    [
        RouteObservationOutcome.AUTHORITY_DENIED,
        RouteObservationOutcome.TENANT_DENIED,
        RouteObservationOutcome.FENCE_INVALID,
        RouteObservationOutcome.APPLICATION_ERROR,
        RouteObservationOutcome.ENGINE_ERROR,
    ],
)
def test_non_path_outcomes_never_demote(outcome: RouteObservationOutcome) -> None:
    view = _view()
    view.mark_optimistic(**_key(), now=0.0)
    view.observe(_obs(RouteObservationOutcome.VERIFIED), now=0.0)
    view.observe(_obs(outcome), now=1.0)
    assert view.state_for(**_key(), now=1.0) is ReachabilityState.VERIFIED


def test_backoff_escalates_across_real_resolve_cycles() -> None:
    # The real service ordering is resolve -> mark_optimistic -> attempt -> observe,
    # repeated across cool-downs; the backoff must climb toward its cap, not reset.
    view = _view()  # base 1s, cap 4s
    now = 0.0
    prev_backoff = 0.0
    for _ in range(4):
        view.mark_optimistic(**_key(), now=now)
        view.observe(_obs(RouteObservationOutcome.CONNECT_FAILURE), now=now)
        entry = next(
            e for e in view.entries() if e.transport is Transport.WORKER_DIRECT
        )
        assert entry.backoff_until is not None
        backoff = entry.backoff_until - now
        assert backoff >= prev_backoff  # monotonic non-decreasing, no reset
        prev_backoff = backoff
        now += backoff + 0.01  # advance past the cool-down for the next cycle
    # It saturates at the cap rather than growing unbounded.
    assert prev_backoff == pytest.approx(4.0)


def test_negative_ttl_forgets_a_demotion_past_the_ceiling() -> None:
    # A backoff longer than the negative TTL is still forgotten at the TTL ceiling.
    view = NetworkReachabilityView(
        ReachabilityBounds(
            positive_ttl_sec=10.0,
            negative_ttl_sec=2.0,
            backoff_base_sec=30.0,
            backoff_max_sec=30.0,
        )
    )
    view.observe(_obs(RouteObservationOutcome.CONNECT_FAILURE), now=0.0)
    assert view.state_for(**_key(), now=1.0) is ReachabilityState.DEMOTED
    # Past the negative TTL (2s) it is forgotten even though backoff (30s) would hold.
    assert view.state_for(**_key(), now=3.0) is ReachabilityState.UNKNOWN


def test_directional_isolation() -> None:
    view = _view()
    view.observe(_obs(RouteObservationOutcome.CONNECT_FAILURE), now=0.0)
    # A -> B demotion does not affect a different transport to the same target.
    assert (
        view.state_for(**_key(Transport.NODE_RELAY), now=0.0)
        is ReachabilityState.UNKNOWN
    )


def test_incarnation_fences_a_recreated_replica() -> None:
    view = _view()
    view.observe(_obs(RouteObservationOutcome.CONNECT_FAILURE, incarnation=1), now=0.0)
    assert view.state_for(**_key(incarnation=1), now=0.0) is ReachabilityState.DEMOTED
    # A recreated incarnation on the same node/generation is a fresh, un-aliased key.
    assert view.state_for(**_key(incarnation=2), now=0.0) is ReachabilityState.UNKNOWN


def test_invalidate_node_drops_target_entries() -> None:
    view = _view()
    view.observe(_obs(RouteObservationOutcome.VERIFIED), now=0.0)
    view.invalidate_node(_TARGET)
    assert view.state_for(**_key(), now=0.0) is ReachabilityState.UNKNOWN
    assert view.entries() == []


def test_no_pairwise_fanout() -> None:
    view = _view()
    # One observation creates exactly one entry — no cross-product probing.
    view.observe(_obs(RouteObservationOutcome.VERIFIED), now=0.0)
    assert len(view.entries()) == 1
