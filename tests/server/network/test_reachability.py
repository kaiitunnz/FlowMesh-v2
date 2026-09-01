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


def _key(transport: Transport = Transport.WORKER_DIRECT) -> dict:
    return {
        "origin_id": _ORIGIN,
        "policy_class": PolicyClass.DEFAULT,
        "target_node_id": _TARGET,
        "listener_generation": _GEN,
        "transport": transport,
    }


def _obs(outcome: RouteObservationOutcome, transport=Transport.WORKER_DIRECT):
    return RouteObservation(
        origin_id=_ORIGIN,
        policy_class=PolicyClass.DEFAULT,
        target_node_id=_TARGET,
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


def test_backoff_grows_with_repeated_failures() -> None:
    view = _view()
    view.observe(_obs(RouteObservationOutcome.CONNECT_FAILURE), now=0.0)
    view.observe(_obs(RouteObservationOutcome.CONNECT_FAILURE), now=0.0)
    # Second failure doubles the backoff (base 1 -> 2s) and is still capped.
    assert view.state_for(**_key(), now=1.5) is ReachabilityState.DEMOTED
    assert view.state_for(**_key(), now=2.5) is ReachabilityState.UNKNOWN


def test_directional_isolation() -> None:
    view = _view()
    view.observe(_obs(RouteObservationOutcome.CONNECT_FAILURE), now=0.0)
    # A -> B demotion does not affect a different transport to the same target.
    assert (
        view.state_for(**_key(Transport.NODE_RELAY), now=0.0)
        is ReachabilityState.UNKNOWN
    )


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
