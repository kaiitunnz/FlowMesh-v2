"""The control plane's decision to authorize one hydration, and what it refuses."""

import asyncio
import threading
from collections.abc import Callable, Coroutine
from types import SimpleNamespace
from typing import Any, cast

import fakeredis
import pytest

from server.clients.redis import WORKERS_SET_KEY, RedisClient
from server.content import (
    ContentHolderDirectory,
    ContentHydrationAuthority,
    ContentTransferSessions,
    HydrationDenial,
)
from server.network.reverse_relay import CONTENT_RELAY_KEYSPACE, RelaySessionStore
from server.network.state import (
    PolicyClass,
    ResolvedRoute,
    RouteCandidate,
    RouteHop,
    TrustedPeerPolicy,
)
from shared.content import (
    ContentHydrationGrant,
    ContentReference,
    ContentRouteObservation,
    HoldingReport,
    reference_for,
)
from shared.schemas.network import (
    CONTENT_PEER_PROTOCOL,
    RouteObservationOutcome,
    Transport,
)
from tests.server.redis_helpers import fake_redis_client

_REFERENCE = reference_for("local", b"prepared", media_type="application/json")
_HELD = (_REFERENCE.authorization_scope, _REFERENCE.content_digest)


class _FakePipeline:
    """Queues the directory's writes and applies them when it is executed."""

    def __init__(self, redis: "_FakeRelayRedis") -> None:
        self._redis = redis
        self._queued: list[Any] = []

    def hset(self, key: str, mapping: dict[str, Any]) -> None:
        self._queued.append(lambda: self._redis.hash_set(key, mapping))

    def expire(self, key: str, ttl_sec: int) -> None:
        self._queued.append(lambda: None)

    async def execute(self) -> None:
        for write in self._queued:
            write()
        self._queued.clear()


class _FakeRelayRedis:
    """The relay's Redis: written through a pipeline, read by the relay's bridges."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, Any]] = {}

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self)

    def hash_set(self, key: str, mapping: dict[str, Any]) -> None:
        self.hashes.setdefault(key, {}).update(mapping)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return {
            str(k).encode(): str(v).encode()
            for k, v in self.hashes.get(key, {}).items()
        }


class _Workers:
    """Live workers by id, and the control frames the authority relayed to them."""

    def __init__(self, **incarnations: int) -> None:
        self._live = incarnations
        self.on_retire: Callable[[str], None] = lambda worker_id: None
        self.reads = 0
        self.frames: list[tuple[str, str, dict[str, Any]]] = []
        # Workers serving a content listener, and so dialing peers.
        self.listening: set[str] = set()
        # Registered workers whose heartbeat has lapsed.
        self.stale: set[str] = set()

    async def get_worker_async(self, worker_id: str) -> Any:
        self.reads += 1
        return self._worker(worker_id)

    async def get_workers_heard_async(
        self, worker_ids: list[str]
    ) -> list[tuple[Any, bool]]:
        self.reads += 1
        return [
            (self._worker(worker_id), worker_id in self.stale)
            for worker_id in worker_ids
        ]

    def _worker(self, worker_id: str) -> Any:
        if worker_id not in self._live:
            return None
        return SimpleNamespace(
            id=worker_id,
            node_id=f"nde-{worker_id[-1]}",
            incarnation=self._live[worker_id],
            capabilities=SimpleNamespace(
                content_listener_port=41000 if worker_id in self.listening else 0
            ),
        )

    def retire(self, worker_id: str) -> None:
        self._live.pop(worker_id, None)
        self.on_retire(worker_id)

    def registered(self) -> list[str]:
        return list(self._live)

    async def get_worker_ids_async(self) -> set[str]:
        return set(self._live)

    def restart(self, worker_id: str) -> None:
        self._live[worker_id] = self._live[worker_id] + 1

    async def publish_mediated_op_async(self, worker: Any, payload: Any) -> int:
        self.frames.append((worker.id, payload.frame_kind, payload.payload))
        return 0

    def kinds(self) -> list[str]:
        return [kind for _, kind, _ in self.frames]


def _authority(
    workers: _Workers,
    *,
    authorizes: bool | Callable[[str, str, ContentReference], bool] = True,
    redis: RedisClient | None = None,
    relay: _FakeRelayRedis | None = None,
    network: Any = None,
) -> ContentHydrationAuthority:
    client = redis if redis is not None else fake_redis_client(fakeredis.FakeServer())
    if registered := workers.registered():
        client.sync.sadd(WORKERS_SET_KEY, *registered)

    def unregister(worker_id: str) -> None:
        client.sync.srem(WORKERS_SET_KEY, worker_id)

    workers.on_retire = unregister
    return ContentHydrationAuthority(
        ContentHolderDirectory(client),
        cast(Any, workers),
        authorizes=(
            authorizes
            if callable(authorizes)
            else lambda task_id, worker_id, reference: authorizes
        ),
        grant_ttl_sec=60.0,
        sessions=ContentTransferSessions(
            cast(Any, _FakeRelayRedis()),
            cast(Any, relay or _FakeRelayRedis()),
            ttl_sec=600.0,
        ),
        network=network,
    )


def _holding(*held: tuple[str, str]) -> HoldingReport:
    return HoldingReport(seq=1, added=held, whole=True)


def _granted(workers: _Workers) -> ContentHydrationGrant:
    serve = next(p for _, kind, p in workers.frames if kind == "content_serve_grant")
    return ContentHydrationGrant.model_validate(serve["grant"])


def _denial(workers: _Workers) -> str:
    denied = next(p for _, kind, p in workers.frames if kind == "content_grant_denied")
    return str(denied["reason"])


def _authorize(
    authority: ContentHydrationAuthority,
    worker_id: str,
    reference: ContentReference,
    peer_only: bool = False,
) -> Coroutine[Any, Any, None]:
    bound = authority.binds("tsk-1", worker_id, reference)
    return authority.authorize(
        worker_id, reference, 7, peer_only=peer_only, bound=bound
    )


def test_a_bound_request_grants_both_ends_the_same_grant() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    assert workers.kinds() == ["content_serve_grant", "content_grant"]
    holder_frame, requester_frame = (payload for _, _, payload in workers.frames)
    assert holder_frame["grant"] == requester_frame["grant"]
    # Without a network plane the transfer is carried over the relay.
    assert requester_frame["carriage"]["routes"] == []
    grant = _granted(workers)
    assert grant.reference == _REFERENCE
    assert (grant.holder_id, grant.holder_generation) == ("wkr-1", 3)
    assert grant.requester_subject == "wkr-2"


def test_the_transfer_record_routes_between_the_two_ends() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    control = fake_redis_client(fakeredis.FakeServer())
    relay = _FakeRelayRedis()
    authority = _authority(workers, redis=control, relay=relay)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    grant = _granted(workers)
    # The record lands where the relay's bridges look a frame's session up.
    record = asyncio.run(
        RelaySessionStore(cast(Any, relay), CONTENT_RELAY_KEYSPACE).load(
            grant.transfer_session_id
        )
    )
    assert record["origin_worker"] == "wkr-2"
    assert record["target_worker"] == "wkr-1"
    assert _REFERENCE.content_digest not in str(record)
    assert not control.sync.scan_keys("ct:sess:*")


def test_a_request_no_binding_authorizes_is_refused() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers, authorizes=False)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    assert workers.kinds() == ["content_grant_denied"]
    assert _denial(workers) == HydrationDenial.NO_BINDING


def test_an_object_no_holder_reported_is_refused_as_untracked() -> None:
    workers = _Workers(**{"wkr-2": 5})
    authority = _authority(workers)

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    assert _denial(workers) == HydrationDenial.NOT_TRACKED


def test_a_holder_that_is_gone_is_refused_rather_than_granted() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))
    workers.retire("wkr-1")

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    assert _denial(workers) == HydrationDenial.HOLDER_UNAVAILABLE


def test_a_holder_that_came_back_as_another_incarnation_is_refused() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))
    workers.restart("wkr-1")

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    assert _denial(workers) == HydrationDenial.HOLDER_UNAVAILABLE


def test_a_live_holder_is_granted_over_one_whose_heartbeat_lapsed() -> None:
    """The lapsed holder recorded last, so it sorts first; its heartbeat puts it out."""
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 4, "wkr-3": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))
    asyncio.run(authority.record_holding("wkr-2", _holding(_HELD)))
    workers.stale.add("wkr-2")

    asyncio.run(_authorize(authority, "wkr-3", _REFERENCE))

    assert _granted(workers).holder_id == "wkr-1"


def test_a_holder_whose_heartbeat_lapsed_is_refused_until_it_resumes() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))
    workers.stale.add("wkr-1")

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))
    assert _denial(workers) == HydrationDenial.HOLDER_UNAVAILABLE

    workers.stale.clear()
    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))
    assert _granted(workers).holder_id == "wkr-1"


def test_a_worker_is_never_sent_to_itself_for_an_object() -> None:
    workers = _Workers(**{"wkr-1": 3})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-1", _REFERENCE))

    assert _denial(workers) == HydrationDenial.HOLDER_UNAVAILABLE


def test_a_scope_is_part_of_what_a_holder_reported() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    elsewhere = _REFERENCE.model_copy(update={"authorization_scope": "other"})
    asyncio.run(_authorize(authority, "wkr-2", elsewhere))

    assert _denial(workers) == HydrationDenial.NOT_TRACKED


def test_an_unknown_requester_is_answered_with_nothing() -> None:
    workers = _Workers(**{"wkr-1": 3})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-9", _REFERENCE))

    assert workers.frames == []


@pytest.mark.parametrize("field", ["grant_id", "transfer_session_id"])
def test_every_grant_is_freshly_minted(field: str) -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))
    first = _granted(workers)
    workers.frames.clear()
    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))
    second = _granted(workers)

    assert getattr(first, field) != getattr(second, field)


def test_a_grant_carries_no_service_admission_identity() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    fields = set(ContentHydrationGrant.model_fields)
    assert not fields & {"idempotency_key", "invocation_id", "claim_id"}
    assert "idm-" not in str(_granted(workers).model_dump())


def test_a_holder_report_costs_one_scripted_round_trip_per_chunk() -> None:
    """A report goes up in one scripted call per 256 copies, not commands per object.

    Every worker's report lands on the one loop that also serves every content request,
    so the cost of a large report there has to stay bounded per round trip.
    """
    workers = _Workers(**{"wkr-1": 3})
    redis = fake_redis_client(fakeredis.FakeServer())
    authority = _authority(workers, redis=redis)
    calls: list[int] = []
    run_script = redis.asyncio.eval

    async def counting(script: str, numkeys: int, *args: Any) -> Any:
        calls.append(numkeys)
        return await run_script(script, numkeys, *args)

    cast(Any, redis.asyncio).eval = counting
    held = [("local", f"{index:064x}") for index in range(300)]
    asyncio.run(authority.record_holding("wkr-1", _holding(*held)))

    assert len(calls) == 2
    directory = ContentHolderDirectory(redis)
    assert all(directory.holders(scope, digest) for scope, digest in held)


class _Network:
    """A network plane resolving each holder to the ladder the test gives it."""

    def __init__(self, ladders: dict[str, list[Transport]], *, probe=False) -> None:
        self._ladders = ladders
        self._probe = probe
        self.worker_directly_routable = True
        self.route_ttl_sec = 30.0
        self.observed: list[tuple[str, Transport, RouteObservationOutcome]] = []
        self.resolved: list[tuple[str, PolicyClass, str]] = []
        self.targets: list[Any] = []
        self.endpoint_reads = 0
        self.peer_enabled = True

    async def endpoints_for(self, node_ids) -> dict[str, Any]:
        self.endpoint_reads += 1
        return {
            node_id: SimpleNamespace(url=f"10.0.0.{node_id[-1]}:9101")
            for node_id in node_ids
        }

    def peer_trust(self, protocol: str) -> TrustedPeerPolicy:
        return TrustedPeerPolicy(
            enabled=self.peer_enabled, trust_domain="fm", protocol=protocol
        )

    async def resolve(
        self, origin_node_id, target, *, trust, policy_class, endpoints
    ) -> Any:
        assert origin_node_id in endpoints and target.node_id in endpoints
        self.resolved.append((target.worker_id, policy_class, trust.protocol))
        self.targets.append(target)
        return SimpleNamespace(
            origin_id="rog-1", origin_node=origin_node_id
        ), self._ladder(target, trust)

    def preview(self, origin_node_id, target, *, trust, policy_class, endpoints) -> Any:
        assert origin_node_id in endpoints and target.node_id in endpoints
        return self._ladder(target, trust)

    def _ladder(self, target, trust) -> ResolvedRoute:
        ladder = self._ladders.get(target.worker_id, [Transport.CONTROL_RELAY])
        if not trust.enabled:
            ladder = [Transport.CONTROL_RELAY]
        return ResolvedRoute(
            origin_id="rog-1",
            target_node_id=target.node_id,
            listener_generation=target.listener_generation,
            route_epoch=7,
            candidates=tuple(
                RouteCandidate(
                    transport=transport,
                    hops=(
                        RouteHop(
                            transport=transport,
                            endpoint=(
                                target.routes[0]
                                if transport is Transport.WORKER_DIRECT
                                else "10.0.0.9:9102"
                            ),
                            node_id=target.node_id,
                        ),
                    ),
                )
                for transport in ladder
            ),
            probe=self._probe,
        )

    def record_observations(self, origin, target, observations) -> None:
        for transport, outcome in observations:
            self.observed.append((target.worker_id, transport, outcome))


def _frame(workers: _Workers, kind: str) -> dict[str, Any]:
    return next(p for _, k, p in workers.frames if k == kind)


def _peers(**incarnations: int) -> _Workers:
    workers = _Workers(**incarnations)
    workers.listening = set(incarnations)
    return workers


def test_a_holder_reached_over_a_peer_transport_is_preferred() -> None:
    workers = _peers(**{"wkr-1": 3, "wkr-3": 4, "wkr-2": 5})
    network = _Network({"wkr-3": [Transport.WORKER_DIRECT, Transport.CONTROL_RELAY]})
    authority = _authority(workers, network=network)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))
    asyncio.run(authority.record_holding("wkr-3", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    assert _granted(workers).holder_id == "wkr-3"
    carriage = _frame(workers, "content_grant")["carriage"]
    assert carriage["routes"] == [
        {"transport": "worker_direct", "endpoint": "10.0.0.3:41000"}
    ]
    assert carriage["holder_worker_id"] == "wkr-3"
    assert 0 < carriage["fresh_for_sec"] <= 30.0
    assert "received_at" not in carriage
    assert {cls for _, cls, _ in network.resolved} == {PolicyClass.CONTENT}
    assert {protocol for _, _, protocol in network.resolved} == {CONTENT_PEER_PROTOCOL}


def test_the_route_rides_beside_the_grant_and_the_holder_learns_the_requester() -> None:
    workers = _peers(**{"wkr-1": 3, "wkr-2": 5})
    relay = _FakeRelayRedis()
    network = _Network({"wkr-1": [Transport.NODE_RELAY, Transport.CONTROL_RELAY]})
    authority = _authority(workers, network=network, relay=relay)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    serve = _frame(workers, "content_serve_grant")
    grant = _frame(workers, "content_grant")
    assert serve["grant"] == grant["grant"]
    assert "carriage" not in serve
    assert serve["requester_host"] == "10.0.0.2"
    assert [r["transport"] for r in grant["carriage"]["routes"]] == ["node_relay"]
    record = asyncio.run(
        RelaySessionStore(cast(Any, relay), CONTENT_RELAY_KEYSPACE).load(
            _granted(workers).transfer_session_id
        )
    )
    assert record["origin_host"] == "10.0.0.2"
    assert record["target_worker"] == "wkr-1"


def test_a_peer_only_read_the_relay_alone_reaches_is_refused_unminted() -> None:
    workers = _peers(**{"wkr-1": 3, "wkr-2": 5})
    relay = _FakeRelayRedis()
    authority = _authority(workers, network=_Network({}), relay=relay)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE, True))

    assert workers.kinds() == ["content_grant_denied"]
    assert _denial(workers) == HydrationDenial.NO_PEER_ROUTE
    assert relay.hashes == {}


def test_a_requester_that_dials_no_peer_is_routed_over_the_relay() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    workers.listening = {"wkr-1"}
    network = _Network({"wkr-1": [Transport.WORKER_DIRECT]})
    authority = _authority(workers, network=network)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    assert _frame(workers, "content_grant")["carriage"]["routes"] == []
    assert _frame(workers, "content_serve_grant")["requester_host"] == ""


def test_a_probe_route_never_carries_a_transfer() -> None:
    workers = _peers(**{"wkr-1": 3, "wkr-2": 5})
    network = _Network({"wkr-1": [Transport.WORKER_DIRECT]}, probe=True)
    authority = _authority(workers, network=network)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE, True))

    assert _denial(workers) == HydrationDenial.NO_PEER_ROUTE


def test_only_the_routed_requester_reports_on_a_transfers_path() -> None:
    workers = _peers(**{"wkr-1": 3, "wkr-2": 5})
    network = _Network({"wkr-1": [Transport.WORKER_DIRECT, Transport.CONTROL_RELAY]})
    authority = _authority(workers, network=network)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))
    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))
    session = _granted(workers).transfer_session_id

    def report(worker_id: str, outcome: str, session_id: str = session) -> None:
        observation = ContentRouteObservation(
            transfer_session_id=session_id, transport="worker_direct", outcome=outcome
        )
        asyncio.run(authority.observe_route(worker_id, observation))

    report("wkr-2", "connect_failure")
    report("wkr-1", "connect_failure")
    report("wkr-2", "connect_failure", "rly-unknown")
    report("wkr-2", "not-an-outcome")

    assert network.observed == [
        ("wkr-1", Transport.WORKER_DIRECT, RouteObservationOutcome.CONNECT_FAILURE)
    ]


def test_requests_from_the_events_thread_run_on_the_bound_loop() -> None:
    workers = _peers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        authority.bind_loop(loop)
        await loop.run_in_executor(
            None, lambda: authority.on_holding("wkr-1", _holding(_HELD))
        )
        for _ in range(100):
            await asyncio.sleep(0.01)
            if workers.frames:
                break
            await loop.run_in_executor(
                None,
                lambda: authority.on_request(
                    "wkr-2", "tsk-1", _REFERENCE, request_id=7
                ),
            )

    asyncio.run(run())
    assert workers.kinds()[:2] == ["content_serve_grant", "content_grant"]


def test_the_binding_check_runs_on_the_thread_that_hands_the_request_over() -> None:
    """The check takes the runtime lock store writes hold, so never on the loop."""
    workers = _peers(**{"wkr-1": 3, "wkr-2": 5})
    checked_on: list[int] = []

    def check(task_id: str, worker_id: str, reference: ContentReference) -> bool:
        checked_on.append(threading.get_ident())
        return True

    authority = _authority(workers, authorizes=check)

    async def run() -> int:
        loop = asyncio.get_running_loop()
        authority.bind_loop(loop)
        await loop.run_in_executor(
            None, lambda: authority.on_holding("wkr-1", _holding(_HELD))
        )
        for _ in range(100):
            await asyncio.sleep(0.01)
            if workers.frames:
                break
            await loop.run_in_executor(
                None,
                lambda: authority.on_request(
                    "wkr-2", "tsk-1", _REFERENCE, request_id=7
                ),
            )
        return threading.get_ident()

    loop_thread = asyncio.run(run())
    assert checked_on and loop_thread not in checked_on


def test_a_binding_check_that_raises_denies_without_breaking_the_events_thread() -> (
    None
):
    workers = _peers(**{"wkr-1": 3, "wkr-2": 5})

    def check(task_id: str, worker_id: str, reference: ContentReference) -> bool:
        raise RuntimeError("runtime unavailable")

    authority = _authority(workers, authorizes=check)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        authority.bind_loop(loop)
        await authority.record_holding("wkr-1", _holding(_HELD))
        await loop.run_in_executor(
            None,
            lambda: authority.on_request("wkr-2", "tsk-1", _REFERENCE, request_id=7),
        )
        for _ in range(100):
            if workers.frames:
                return
            await asyncio.sleep(0.01)

    asyncio.run(run())
    assert _denial(workers) == HydrationDenial.NO_BINDING


def _many_holders(count: int) -> tuple[_Workers, list[str]]:
    holders = [f"wkr-{index}" for index in range(3, 3 + count)]
    workers = _peers(**{"wkr-2": 5, **{holder: 1 for holder in holders}})
    return workers, holders


@pytest.mark.parametrize("count", [1, 12])
def test_without_a_peer_transport_a_grant_reads_each_holder_once_and_resolves_none(
    count: int,
) -> None:
    workers, holders = _many_holders(count)
    network = _Network({})
    network.peer_enabled = False
    authority = _authority(workers, network=network)
    for holder in holders:
        asyncio.run(authority.record_holding(holder, _holding(_HELD)))
    workers.reads = network.endpoint_reads = 0

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    assert _granted(workers).holder_id in holders
    # The requester, then every reported holder in one batch.
    assert workers.reads == 2
    assert network.endpoint_reads == 1
    assert network.resolved == []


@pytest.mark.parametrize(("count", "reachable"), [(1, 0), (12, 0), (12, 11)])
def test_with_peer_transports_the_peer_reached_holder_is_found_at_any_position(
    count: int, reachable: int
) -> None:
    workers, holders = _many_holders(count)
    network = _Network({holders[reachable]: [Transport.NODE_RELAY]})
    authority = _authority(workers, network=network)
    for holder in reversed(holders):
        asyncio.run(authority.record_holding(holder, _holding(_HELD)))
    workers.reads = network.endpoint_reads = 0

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    assert _granted(workers).holder_id == holders[reachable]
    # The requester, then every reported holder in one batch.
    assert workers.reads == 2
    # Every node's advertisement in one batch, and only the chosen holder resolved.
    assert network.endpoint_reads == 1
    assert [worker for worker, *_ in network.resolved] == [holders[reachable]]


@pytest.mark.parametrize(
    ("routable", "listening", "address", "direct"),
    [
        (True, True, "10.0.0.3:41000", True),
        (False, True, "10.0.0.3:41000", False),
        (True, False, "content://wkr-3", False),
    ],
)
def test_a_holders_content_listener_is_its_nodes_host_and_its_bound_port(
    routable: bool, listening: bool, address: str, direct: bool
) -> None:
    workers = _peers(**{"wkr-3": 4, "wkr-2": 5})
    if not listening:
        workers.listening.discard("wkr-3")
    network = _Network({})
    network.worker_directly_routable = routable
    authority = _authority(workers, network=network)
    asyncio.run(authority.record_holding("wkr-3", _holding(_HELD)))

    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))

    (target,) = network.targets
    assert target.routes == (address,)
    assert target.protocols == ((CONTENT_PEER_PROTOCOL,) if listening else ())
    assert target.directly_routable is direct


def test_an_answer_carries_the_number_of_the_request_it_answers() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", _holding(_HELD)))
    asyncio.run(_authorize(authority, "wkr-2", _REFERENCE))
    denying = _authority(workers, authorizes=False)
    asyncio.run(_authorize(denying, "wkr-2", _REFERENCE))

    assert _frame(workers, "content_grant")["request_id"] == 7
    assert _frame(workers, "content_grant_denied")["request_id"] == 7
    assert "request_id" not in _frame(workers, "content_grant")["grant"]
