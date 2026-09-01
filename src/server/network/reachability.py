"""The derived network reachability view.

A directional, endpoint-/policy-class-scoped view over classified route observations. It
holds no authority: it never admits capacity, mints a claim, or issues a route — it only
records whether a path has been seen to work, fail, or is untried. Entries are keyed by
``(origin, policy class, target node, listener generation, transport)`` and created
lazily (demand-paged); a query for an unseen key returns ``UNKNOWN`` without allocating.

State machine: ``UNKNOWN`` → ``OPTIMISTIC`` when a route is attempted → ``VERIFIED`` on
a successful observation, or ``DEMOTED`` on a network-path failure with a negative TTL
and bounded retry backoff. Authority, tenant, fence, application, and engine failures do
not evidence the path and leave the state unchanged. Keying by listener generation
fences a stale advertisement: a new generation gets fresh ``UNKNOWN`` entries.
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

_ReachabilityKey = tuple[str, PolicyClass, str, int, Transport]


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
        listener_generation: int,
        transport: Transport,
    ) -> _ReachabilityKey:
        return (origin_id, policy_class, target_node_id, listener_generation, transport)

    def mark_optimistic(
        self,
        origin_id: str,
        policy_class: PolicyClass,
        target_node_id: str,
        listener_generation: int,
        transport: Transport,
        *,
        now: float,
    ) -> None:
        """Move an untried or cooled-down key to ``OPTIMISTIC`` before an attempt.

        A verified or still-backing-off entry is left as is: an attempt does not erase a
        known outcome.
        """
        key = self._key(
            origin_id, policy_class, target_node_id, listener_generation, transport
        )
        entry = self._entries.get(key)
        state = entry.state if entry is not None else ReachabilityState.UNKNOWN
        if state is ReachabilityState.VERIFIED and not self._expired(entry, now):
            return
        if state is ReachabilityState.DEMOTED and self._backing_off(entry, now):
            return
        self._entries[key] = ReachabilityEntry(
            origin_id=origin_id,
            policy_class=policy_class,
            target_node_id=target_node_id,
            listener_generation=listener_generation,
            transport=transport,
            state=ReachabilityState.OPTIMISTIC,
        )

    def observe(self, obs: RouteObservation, *, now: float) -> None:
        """Settle a key from an attempt's classified outcome.

        A path failure demotes with a negative TTL and exponential backoff; a success
        verifies with a positive TTL. A non-path (authority/tenant/fence/application/
        engine) outcome is not evidence about the path and leaves the state unchanged.
        """
        key = self._key(
            obs.origin_id,
            obs.policy_class,
            obs.target_node_id,
            obs.listener_generation,
            obs.transport,
        )
        prior = self._entries.get(key)
        if is_demoting(obs.outcome):
            retries = (prior.retries + 1) if prior is not None else 1
            backoff = min(
                self._bounds.backoff_base_sec * (2 ** (retries - 1)),
                self._bounds.backoff_max_sec,
            )
            self._entries[key] = ReachabilityEntry(
                origin_id=obs.origin_id,
                policy_class=obs.policy_class,
                target_node_id=obs.target_node_id,
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
        listener_generation: int,
        transport: Transport,
        *,
        now: float,
    ) -> ReachabilityState:
        """The current reachability, applying TTL expiry and backoff cool-down.

        A verified entry past its positive TTL and a demoted entry past its backoff both
        return ``UNKNOWN`` so the resolver may re-attempt optimistically.
        """
        entry = self._entries.get(
            self._key(
                origin_id, policy_class, target_node_id, listener_generation, transport
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
                if self._backing_off(entry, now)
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
