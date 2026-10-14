"""Deciding who may hydrate which object, and arranging the transfer.

The control-plane actor over the holder directory. A worker that needs an object asks
here; the authority checks that the asking worker is the one running the task and that
the task's own binding names exactly this reference, resolves a live holder and how the
asking worker reaches it, mints one grant, and hands it to both ends over their
authenticated attachments. Then it steps out: the bytes move worker to worker, over a
connection the requester dials or over the relay, and no payload passes through here.

A carriage plan rides beside each grant, naming how the requester reaches the holder; a
holder reached over a peer transport is preferred. A dialed transfer reports what its
path did, which feeds the network plane's reachability evidence.

It reads consumer bindings as evidence and never writes one. A denial creates nothing —
no invocation, no claim, no credit — and settles nothing: the asking worker surfaces a
typed hydration failure and its own consumer decides what that means.

Every request, holding report and path report is handled on the server's loop, which
owns the directory, the routing records and the record of which transfer connects
which pair; the thread that reads worker events only hands them over.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from shared.content import (
    ContentCarriagePlan,
    ContentHydrationGrant,
    ContentPeerRoute,
    ContentReference,
    ContentRouteObservation,
)
from shared.schemas.command import MediatedOpMessage
from shared.schemas.network import (
    CONTENT_PEER_PROTOCOL,
    RouteObservationOutcome,
    Transport,
)
from shared.utils.ids import new_hydration_grant_id, new_relay_session_id

from ..network.service import NetworkPlane
from ..network.state import (
    ContentListenerTarget,
    NetworkEndpointAdvertisement,
    PolicyClass,
    ResolvedRoute,
    RouteOrigin,
)
from ..registries.worker import Worker, WorkerRegistry
from .directory import ContentHolderDirectory, ContentHolderRecord
from .sessions import ContentTransferSessions

# Whether the task's own binding entitles this worker to read this exact reference.
BindingCheck = Callable[[str, str, ContentReference], bool]

_PEER_TRANSPORTS = frozenset({Transport.WORKER_DIRECT, Transport.NODE_RELAY})


class HydrationDenial(StrEnum):
    """Why control refused to authorize a hydration."""

    NO_BINDING = "no_binding"
    NOT_TRACKED = "not_tracked"
    HOLDER_UNAVAILABLE = "holder_unavailable"
    # The read asked for a peer transport only, and no holder is reached over one.
    NO_PEER_ROUTE = "no_peer_route"
    # Control runs no worker content cache, so it resolves no holder for anyone.
    CACHE_OFF = "cache_off"


@dataclass(frozen=True)
class _Route:
    """How one requester reaches one holder, as the network plane resolved it."""

    origin: RouteOrigin
    target: ContentListenerTarget
    resolved: ResolvedRoute

    @property
    def head(self) -> Transport | None:
        candidates = self.resolved.candidates
        return candidates[0].transport if candidates else None

    @property
    def peer(self) -> bool:
        return self.head in _PEER_TRANSPORTS


@dataclass(frozen=True)
class _Transfer:
    """Which pair one dialed transfer connects, for the path report it may send."""

    requester_id: str
    origin: RouteOrigin
    target: ContentListenerTarget
    expires_at_epoch: float


class ContentHydrationAuthority:
    """Mints the grant one worker needs to read one object from another."""

    def __init__(
        self,
        directory: ContentHolderDirectory,
        worker_registry: WorkerRegistry,
        *,
        authorizes: BindingCheck,
        grant_ttl_sec: float,
        sessions: ContentTransferSessions,
        network: NetworkPlane | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._directory = directory
        self._workers = worker_registry
        self._authorizes = authorizes
        self._grant_ttl_sec = grant_ttl_sec
        self._sessions = sessions
        self._network = network
        self._logger = logger or logging.getLogger("content-authority")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._transfers: dict[str, _Transfer] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Handle every request on ``loop``, the server's own."""
        self._loop = loop

    def on_holding(self, worker_id: str, held: Sequence[tuple[str, str]]) -> None:
        """Take a worker's holding report off the calling thread."""
        self._submit(self.record_holding(worker_id, held))

    def on_request(
        self,
        worker_id: str,
        task_id: str,
        reference: ContentReference,
        *,
        request_id: int,
        peer_only: bool = False,
    ) -> None:
        """Take a worker's hydration request off the calling thread.

        The binding check reads the task runtime under its own lock, which other
        threads hold across store writes, so it runs here rather than on the loop.
        """
        try:
            bound = self.binds(task_id, worker_id, reference)
        except Exception:
            # The events thread must keep reading; an unanswerable check denies.
            self._logger.exception("content binding check failed for %s", task_id)
            bound = False
        self._submit(
            self.authorize(
                worker_id, reference, request_id, peer_only=peer_only, bound=bound
            )
        )

    def binds(self, task_id: str, worker_id: str, reference: ContentReference) -> bool:
        """Whether the task the worker runs is bound to exactly this reference."""
        return self._authorizes(task_id, worker_id, reference)

    def on_route_observation(
        self, worker_id: str, observation: ContentRouteObservation
    ) -> None:
        """Take a dialed transfer's path report off the calling thread."""
        self._submit(self.observe_route(worker_id, observation))

    def _submit(self, work: Coroutine[Any, Any, None]) -> None:
        if self._loop is None:
            work.close()
            self._logger.warning("content authority has no loop; dropping a request")
            return
        asyncio.run_coroutine_threadsafe(work, self._loop).add_done_callback(
            self._report_failure
        )

    def _report_failure(self, done: Future[None]) -> None:
        if not done.cancelled() and (exc := done.exception()) is not None:
            self._logger.error(
                "content authority request failed",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def record_holding(
        self, worker_id: str, held: Sequence[tuple[str, str]]
    ) -> None:
        """Note the objects a worker holds, from its write or its periodic report.

        A report is what keeps a record live, and it arrives under the reporting
        worker's current incarnation, so a restarted holder supersedes the records its
        previous one left behind rather than waiting for them to lapse.
        """
        worker = await self._workers.get_worker_async(worker_id)
        if worker is None:
            return
        await self._directory.record_many_async(
            held,
            worker_id=worker_id,
            node_id=worker.node_id,
            generation=worker.incarnation,
        )

    async def authorize(
        self,
        worker_id: str,
        reference: ContentReference,
        request_id: int,
        *,
        peer_only: bool = False,
        bound: bool,
    ) -> None:
        """Grant one hydration, or relay the reason it is refused, under the
        requester's ``request_id``.

        ``bound`` is the binding check's answer for the requesting task.

        With ``peer_only``, a holder the relay alone reaches is refused.
        """
        requester = await self._workers.get_worker_async(worker_id)
        if requester is None:
            # Nothing to answer: a request from a worker the registry no longer knows
            # has no attachment to relay a grant or a refusal over.
            return
        if not bound:
            await self._deny(requester, request_id, HydrationDenial.NO_BINDING)
            return
        reported = await self._directory.holders_async(
            reference.authorization_scope, reference.content_digest
        )
        if not reported:
            # Nothing ever reported holding it — no copy was ever announced, or every
            # holder's report has lapsed. Its consumer reads the shared store instead.
            await self._deny(requester, request_id, HydrationDenial.NOT_TRACKED)
            return
        live = await self._live_holders(reference, reported, requester)
        if not live:
            await self._deny(requester, request_id, HydrationDenial.HOLDER_UNAVAILABLE)
            return
        endpoints = await self._endpoints(requester, live)
        requester_host = self._dial_host(requester, endpoints)
        holder, route = await self._choose_holder(
            live, requester, requester_host, endpoints
        )
        if peer_only and (route is None or not route.peer):
            await self._deny(requester, request_id, HydrationDenial.NO_PEER_ROUTE)
            return
        grant = ContentHydrationGrant(
            grant_id=new_hydration_grant_id(),
            reference=reference,
            requester_subject=worker_id,
            requester_origin_id=requester.node_id,
            holder_id=holder.id,
            holder_generation=holder.incarnation,
            transfer_session_id=new_relay_session_id(),
            expires_at_epoch=time.time() + self._grant_ttl_sec,
        )
        # The transfer's routing record: the relay and the holder's node bridge frames
        # between these two ends by it. It carries no object identity.
        await self._sessions.open_async(
            grant.transfer_session_id,
            origin_node=requester.node_id,
            target_node=holder.node_id,
            origin_worker=worker_id,
            target_worker=holder.id,
            size_bytes=reference.size_bytes,
            origin_host=requester_host,
        )
        carriage = self._carriage(grant, holder, route)
        if route is not None and carriage.dialed:
            self._remember(grant, requester, route)
        grant_json = grant.model_dump(mode="json")
        await self._relay(
            holder,
            "content_serve_grant",
            {"grant": grant_json, "requester_host": requester_host},
        )
        await self._relay(
            requester,
            "content_grant",
            {
                "request_id": request_id,
                "grant": grant_json,
                "carriage": carriage.model_dump(mode="json"),
            },
        )

    async def observe_route(
        self, worker_id: str, observation: ContentRouteObservation
    ) -> None:
        """Fold a dialed transfer's path report into the reachability evidence, heard
        only from the requester the transfer was routed for."""
        transfer = self._transfers.get(observation.transfer_session_id)
        if transfer is None or transfer.requester_id != worker_id:
            return
        try:
            transport = Transport(observation.transport)
            outcome = RouteObservationOutcome(observation.outcome)
        except ValueError:
            return
        if self._network is not None and transport in _PEER_TRANSPORTS:
            self._network.record_observations(
                transfer.origin, transfer.target, [(transport, outcome)]
            )

    async def _endpoints(
        self, requester: Worker, live: list[Worker]
    ) -> dict[str, NetworkEndpointAdvertisement | None]:
        """Read, in one round trip, the requester's node advertisement and, when a
        peer transport could carry the transfer, every live holder's."""
        if self._network is None:
            return {}
        nodes = [requester.node_id]
        if self._dials(requester):
            nodes += [holder.node_id for holder in live]
        return await self._network.endpoints_for(nodes)

    def _dials(self, requester: Worker) -> bool:
        """Whether the requester could dial a holder at all."""
        return (
            self._network is not None
            and requester.capabilities.content_listener_port > 0
            and self._network.peer_trust(CONTENT_PEER_PROTOCOL).enabled
        )

    def _dial_host(
        self,
        requester: Worker,
        endpoints: dict[str, NetworkEndpointAdvertisement | None],
    ) -> str:
        """Return the host the requester's dialed connections come from, its node's
        advertised address, which that node's certificate covers; empty when it dials
        none."""
        if self._network is None or requester.capabilities.content_listener_port <= 0:
            return ""
        return NetworkPlane.node_host(endpoints.get(requester.node_id))

    async def _live_holders(
        self,
        reference: ContentReference,
        reported: list[ContentHolderRecord],
        requester: Worker,
    ) -> list[Worker]:
        """Every reported holder still running as the incarnation that reported."""
        candidates = [r for r in reported if r.worker_id != requester.id]
        found = await self._workers.get_workers_async([r.worker_id for r in candidates])
        live: list[Worker] = []
        for record, holder in zip(candidates, found, strict=True):
            if holder is None or holder.incarnation != record.generation:
                # The reporting worker is gone or came back as another incarnation, so
                # its report describes content no one can serve. A restarted worker
                # re-reports what it still holds, which lands under its new incarnation.
                await self._directory.forget_async(
                    reference.authorization_scope,
                    reference.content_digest,
                    record.worker_id,
                    record.generation,
                )
                continue
            live.append(holder)
        return live

    async def _choose_holder(
        self,
        live: list[Worker],
        requester: Worker,
        requester_host: str,
        endpoints: dict[str, NetworkEndpointAdvertisement | None],
    ) -> tuple[Worker, _Route | None]:
        """Return a live holder and its route, preferring one a peer transport
        reaches. Evidence, not authority.

        Without a peer transport to offer, the first live holder serves over the relay
        unresolved. Otherwise every live holder is ranked without a trace, and only the
        one picked — the first a peer transport reaches, else the first — is resolved.
        """
        if self._network is None or not requester_host or not self._dials(requester):
            return live[0], None
        network = self._network
        trust = network.peer_trust(CONTENT_PEER_PROTOCOL)

        def peer_reached(target: ContentListenerTarget) -> bool:
            ladder = network.preview(
                requester.node_id,
                target,
                trust=trust,
                policy_class=PolicyClass.CONTENT,
                endpoints=endpoints,
            )
            return (
                ladder is not None
                and not ladder.probe
                and bool(ladder.candidates)
                and ladder.candidates[0].transport in _PEER_TRANSPORTS
            )

        ranked = [(holder, self._target(holder, endpoints)) for holder in live]
        holder, target = next(((h, t) for h, t in ranked if peer_reached(t)), ranked[0])
        resolved = await network.resolve(
            requester.node_id,
            target,
            trust=trust,
            policy_class=PolicyClass.CONTENT,
            endpoints=endpoints,
        )
        if resolved is None or resolved[1].probe:
            return holder, None
        origin, route = resolved
        return holder, _Route(origin, target, route)

    def _target(
        self,
        holder: Worker,
        endpoints: dict[str, NetworkEndpointAdvertisement | None],
    ) -> ContentListenerTarget:
        """The holder's content listener as a route target."""
        address = NetworkPlane.listener_address(
            endpoints.get(holder.node_id), holder.capabilities.content_listener_port
        )
        return ContentListenerTarget(
            worker_id=holder.id,
            node_id=holder.node_id,
            incarnation=holder.incarnation,
            listener_generation=holder.incarnation,
            routes=(address or f"content://{holder.id}",),
            protocols=(CONTENT_PEER_PROTOCOL,) if address else (),
            directly_routable=bool(address)
            and self._network is not None
            and self._network.worker_directly_routable,
        )

    def _carriage(
        self, grant: ContentHydrationGrant, holder: Worker, route: _Route | None
    ) -> ContentCarriagePlan:
        plan = ContentCarriagePlan(
            transfer_session_id=grant.transfer_session_id,
            grant_id=grant.grant_id,
            holder_worker_id=holder.id,
            holder_generation=holder.incarnation,
            listener_generation=holder.incarnation,
        )
        if route is None or not route.peer or self._network is None:
            return plan
        return plan.model_copy(
            update={
                "routes": tuple(
                    ContentPeerRoute(
                        transport=candidate.transport,
                        endpoint=candidate.hops[0].endpoint,
                    )
                    for candidate in route.resolved.candidates
                    if candidate.transport in _PEER_TRANSPORTS and candidate.hops
                ),
                "route_epoch": route.resolved.route_epoch,
                "fresh_for_sec": min(self._network.route_ttl_sec, self._grant_ttl_sec),
            }
        )

    def _remember(
        self, grant: ContentHydrationGrant, requester: Worker, route: _Route
    ) -> None:
        now = time.time()
        for session_id, transfer in list(self._transfers.items()):
            if transfer.expires_at_epoch < now:
                del self._transfers[session_id]
        # A path report may trail the transfer it describes by a little.
        self._transfers[grant.transfer_session_id] = _Transfer(
            requester_id=requester.id,
            origin=route.origin,
            target=route.target,
            expires_at_epoch=grant.expires_at_epoch + self._grant_ttl_sec,
        )

    async def _deny(
        self, requester: Worker, request_id: int, denial: HydrationDenial
    ) -> None:
        self._logger.info(
            "content hydration refused for %s: %s", requester.id, denial.value
        )
        await self._relay(
            requester,
            "content_grant_denied",
            {"request_id": request_id, "reason": denial.value},
        )

    async def _relay(
        self, worker: Worker, frame_kind: str, payload: dict[str, Any]
    ) -> None:
        await self._workers.publish_mediated_op_async(
            worker,
            MediatedOpMessage(
                worker_id=worker.id, frame_kind=frame_kind, payload=payload
            ),
        )
