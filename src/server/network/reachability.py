"""The derived network reachability view.

A directional, endpoint-/policy-class-scoped view over classified route observations.
It records only whether a path has been seen to work, fail, or is untried. Entries are
keyed by ``(origin, policy class, target node, incarnation, listener generation,
transport)``, created lazily; a query for an unseen key returns ``UNKNOWN`` and
allocates nothing.

State machine: ``UNKNOWN`` → ``OPTIMISTIC`` on an attempt → ``VERIFIED`` on a success,
or ``DEMOTED`` on a network-path failure. A demotion holds until its retry backoff cools
or its negative TTL forgets it, whichever comes first, and the backoff climbs across
repeated failures toward its cap. Authority, tenant, fence, application, and engine
failures are not path evidence and leave the state unchanged. Keying by incarnation and
listener generation fences a stale advertisement: a recreated incarnation gets fresh
entries and never inherits a dead one's state.
"""

from dataclasses import dataclass

from .state import (
    PolicyClass,
    ReachabilityEntry,
    ReachabilityState,
    RouteObservation,
    RouteObservationOutcome,
    Transport,
    is_demoting,
)

_ReachabilityKey = tuple[str, PolicyClass, str, int, int, Transport]

# Cap the backoff exponent so a long-dead path cannot grow the shift without bound; the
# computed backoff is separately clamped to ``backoff_max_sec``.
_MAX_BACKOFF_SHIFT = 30


@dataclass(frozen=True)
class ReachabilityBounds:
    """TTL and backoff bounds for the reachability state machine."""

    positive_ttl_sec: float = 30.0
    negative_ttl_sec: float = 15.0
    backoff_base_sec: float = 1.0
    backoff_max_sec: float = 30.0


class NetworkReachabilityView:
    """The derived directional reachability evidence."""

    def __init__(self, bounds: ReachabilityBounds | None = None) -> None:
        self._bounds = bounds or ReachabilityBounds()
        self._entries: dict[_ReachabilityKey, ReachabilityEntry] = {}

    @staticmethod
    def _key(
        origin_id: str,
        policy_class: PolicyClass,
        target_node_id: str,
        incarnation: int,
        listener_generation: int,
        transport: Transport,
    ) -> _ReachabilityKey:
        return (
            origin_id,
            policy_class,
            target_node_id,
            incarnation,
            listener_generation,
            transport,
        )

    def mark_optimistic(
        self,
        origin_id: str,
        policy_class: PolicyClass,
        target_node_id: str,
        incarnation: int,
        listener_generation: int,
        transport: Transport,
        *,
        now: float,
    ) -> None:
        """Move an untried or cooled-down key to ``OPTIMISTIC`` before an attempt.

        A verified or still-demoted entry is left as is: an attempt does not erase a
        known outcome. A prior failure count is carried forward so a re-attempt after
        cool-down keeps escalating the backoff rather than resetting it.
        """
        key = self._key(
            origin_id,
            policy_class,
            target_node_id,
            incarnation,
            listener_generation,
            transport,
        )
        entry = self._entries.get(key)
        state = entry.state if entry is not None else ReachabilityState.UNKNOWN
        if state is ReachabilityState.VERIFIED and not self._expired(entry, now):
            return
        if state is ReachabilityState.DEMOTED and self._demoted_active(entry, now):
            return
        self._entries[key] = ReachabilityEntry(
            origin_id=origin_id,
            policy_class=policy_class,
            target_node_id=target_node_id,
            incarnation=incarnation,
            listener_generation=listener_generation,
            transport=transport,
            state=ReachabilityState.OPTIMISTIC,
            retries=entry.retries if entry is not None else 0,
        )

    def observe(self, obs: RouteObservation, *, now: float) -> None:
        """Settle a key from an attempt's classified outcome.

        A path failure demotes and escalates the backoff from the carried failure
        count; a success verifies and clears it. A non-path (authority/tenant/fence/
        application/engine) outcome is not path evidence and leaves the state unchanged.
        """
        key = self._key(
            obs.origin_id,
            obs.policy_class,
            obs.target_node_id,
            obs.incarnation,
            obs.listener_generation,
            obs.transport,
        )
        prior = self._entries.get(key)
        if is_demoting(obs.outcome):
            retries = (prior.retries + 1) if prior is not None else 1
            shift = min(retries - 1, _MAX_BACKOFF_SHIFT)
            backoff = min(
                self._bounds.backoff_base_sec * (2**shift),
                self._bounds.backoff_max_sec,
            )
            self._entries[key] = ReachabilityEntry(
                origin_id=obs.origin_id,
                policy_class=obs.policy_class,
                target_node_id=obs.target_node_id,
                incarnation=obs.incarnation,
                listener_generation=obs.listener_generation,
                transport=obs.transport,
                state=ReachabilityState.DEMOTED,
                expires_at=now + self._bounds.negative_ttl_sec,
                backoff_until=now + backoff,
                retries=retries,
            )
        elif obs.outcome is RouteObservationOutcome.VERIFIED:
            self._entries[key] = ReachabilityEntry(
                origin_id=obs.origin_id,
                policy_class=obs.policy_class,
                target_node_id=obs.target_node_id,
                incarnation=obs.incarnation,
                listener_generation=obs.listener_generation,
                transport=obs.transport,
                state=ReachabilityState.VERIFIED,
                expires_at=now + self._bounds.positive_ttl_sec,
            )
        # Non-path outcomes: no state change (they never demote or promote a path).

    def state_for(
        self,
        origin_id: str,
        policy_class: PolicyClass,
        target_node_id: str,
        incarnation: int,
        listener_generation: int,
        transport: Transport,
        *,
        now: float,
    ) -> ReachabilityState:
        """The current reachability, applying TTL expiry and backoff cool-down.

        A verified entry past its positive TTL returns ``UNKNOWN`` for re-verification.
        A demoted entry stays ``DEMOTED`` only while backing off and within its negative
        TTL; once either lapses it returns ``UNKNOWN`` for an optimistic retry.
        """
        entry = self._entries.get(
            self._key(
                origin_id,
                policy_class,
                target_node_id,
                incarnation,
                listener_generation,
                transport,
            )
        )
        if entry is None:
            return ReachabilityState.UNKNOWN
        if entry.state is ReachabilityState.VERIFIED:
            return (
                ReachabilityState.UNKNOWN
                if self._expired(entry, now)
                else ReachabilityState.VERIFIED
            )
        if entry.state is ReachabilityState.DEMOTED:
            return (
                ReachabilityState.DEMOTED
                if self._demoted_active(entry, now)
                else ReachabilityState.UNKNOWN
            )
        return entry.state

    def invalidate_node(self, target_node_id: str) -> None:
        """Drop every entry targeting a node whose advertisement was superseded."""
        self._entries = {
            key: entry
            for key, entry in self._entries.items()
            if entry.target_node_id != target_node_id
        }

    def entries(self) -> list[ReachabilityEntry]:
        return list(self._entries.values())

    def _demoted_active(self, entry: ReachabilityEntry | None, now: float) -> bool:
        # A demotion holds while it is still backing off and not yet past the negative
        # TTL forget ceiling — whichever bound elapses first releases it.
        return self._backing_off(entry, now) and not self._expired(entry, now)

    @staticmethod
    def _expired(entry: ReachabilityEntry | None, now: float) -> bool:
        return (
            entry is not None
            and entry.expires_at is not None
            and now >= entry.expires_at
        )

    @staticmethod
    def _backing_off(entry: ReachabilityEntry | None, now: float) -> bool:
        return (
            entry is not None
            and entry.backoff_until is not None
            and now < entry.backoff_until
        )
