"""The control plane's decision to authorize one hydration, and what it refuses."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

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
from shared.content import ContentHydrationGrant, reference_for
from shared.content.carriage import ContentRouteObservation
from shared.schemas.network import (
    CONTENT_PEER_PROTOCOL,
    RouteObservationOutcome,
    Transport,
)

_REFERENCE = reference_for("local", b"prepared", media_type="application/json")
_HELD = (_REFERENCE.authorization_scope, _REFERENCE.content_digest)


class _FakePipeline:
    """Queues the directory's writes and applies them when it is executed."""

    def __init__(self, redis: "_FakeRedis | _FakeRelayRedis") -> None:
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


class _FakeRedis:
    """Only the hash and expire surface the directory and session record use."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, Any]] = {}
        self.pipelines = 0

    def control_pipeline(self) -> _FakePipeline:
        self.pipelines += 1
        return _FakePipeline(self)

    def hash_set(self, key: str, mapping: dict[str, Any]) -> None:
        self.hashes.setdefault(key, {}).update(mapping)

    async def hash_getall(self, key: str) -> dict[str, Any]:
        return dict(self.hashes.get(key, {}))

    async def hash_delete(self, key: str, *fields: str) -> None:
        for field in fields:
            self.hashes.get(key, {}).pop(field, None)

    async def expire(self, key: str, ttl_sec: int) -> bool:
        return True


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
        self.frames: list[tuple[str, str, dict[str, Any]]] = []
        # Workers serving a content listener, and so dialing peers.
        self.listening: set[str] = set()

    async def get_worker_async(self, worker_id: str) -> Any:
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
    authorizes: bool = True,
    redis: Any = None,
    relay: _FakeRelayRedis | None = None,
    network: Any = None,
) -> ContentHydrationAuthority:
    fake = redis if redis is not None else _FakeRedis()
    client = cast(Any, SimpleNamespace(asyncio=fake))
    return ContentHydrationAuthority(
        ContentHolderDirectory(client, record_ttl_sec=300.0),
        cast(Any, workers),
        authorizes=lambda task_id, worker_id, reference: authorizes,
        grant_ttl_sec=60.0,
        sessions=ContentTransferSessions(
            cast(Any, relay or _FakeRelayRedis()), ttl_sec=600.0
        ),
        network=network,
    )


def _granted(workers: _Workers) -> ContentHydrationGrant:
    serve = next(p for _, kind, p in workers.frames if kind == "content_serve_grant")
    return ContentHydrationGrant.model_validate(serve["grant"])


def _denial(workers: _Workers) -> str:
    denied = next(p for _, kind, p in workers.frames if kind == "content_grant_denied")
    return str(denied["reason"])


def test_a_bound_request_grants_both_ends_the_same_grant() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))

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
    control = _FakeRedis()
    relay = _FakeRelayRedis()
    authority = _authority(workers, redis=control, relay=relay)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))

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
    assert not any(key.startswith("ct:sess:") for key in control.hashes)


def test_a_request_no_binding_authorizes_is_refused() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers, authorizes=False)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))

    assert workers.kinds() == ["content_grant_denied"]
    assert _denial(workers) == HydrationDenial.NO_BINDING


def test_an_object_no_holder_reported_is_refused_as_untracked() -> None:
    workers = _Workers(**{"wkr-2": 5})
    authority = _authority(workers)

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))

    assert _denial(workers) == HydrationDenial.NOT_TRACKED


def test_a_holder_that_is_gone_is_refused_rather_than_granted() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))
    workers.retire("wkr-1")

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))

    assert _denial(workers) == HydrationDenial.HOLDER_UNAVAILABLE


def test_a_holder_that_came_back_as_another_incarnation_is_refused() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))
    workers.restart("wkr-1")

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))

    assert _denial(workers) == HydrationDenial.HOLDER_UNAVAILABLE


def test_a_worker_is_never_sent_to_itself_for_an_object() -> None:
    workers = _Workers(**{"wkr-1": 3})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-1", "tsk-1", _REFERENCE))

    assert _denial(workers) == HydrationDenial.HOLDER_UNAVAILABLE


def test_a_scope_is_part_of_what_a_holder_reported() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    elsewhere = _REFERENCE.model_copy(update={"authorization_scope": "other"})
    asyncio.run(authority.authorize("wkr-2", "tsk-1", elsewhere))

    assert _denial(workers) == HydrationDenial.NOT_TRACKED


def test_an_unknown_requester_is_answered_with_nothing() -> None:
    workers = _Workers(**{"wkr-1": 3})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-9", "tsk-1", _REFERENCE))

    assert workers.frames == []


@pytest.mark.parametrize("field", ["grant_id", "transfer_session_id"])
def test_every_grant_is_freshly_minted(field: str) -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))
    first = _granted(workers)
    workers.frames.clear()
    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))
    second = _granted(workers)

    assert getattr(first, field) != getattr(second, field)


def test_a_grant_carries_no_service_admission_identity() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    authority = _authority(workers)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))

    fields = set(ContentHydrationGrant.model_fields)
    assert not fields & {"idempotency_key", "invocation_id", "claim_id"}
    assert "idm-" not in str(_granted(workers).model_dump())


def test_a_holder_report_costs_one_round_trip_whatever_it_holds() -> None:
    """A whole report goes up in one round trip, not two commands per object.

    A holder re-reports everything it caches on every cadence, and every worker's
    report lands on the one loop that also serves every content request, so the cost
    of a large cache there has to stay flat.
    """
    workers = _Workers(**{"wkr-1": 3})
    redis = _FakeRedis()
    authority = _authority(workers, redis=redis)

    held = [("local", f"{index:064x}") for index in range(50)]
    asyncio.run(authority.record_holding("wkr-1", held))

    assert redis.pipelines == 1
    assert len(redis.hashes) == 50


class _Network:
    """A network plane resolving each holder to the ladder the test gives it."""

    def __init__(self, ladders: dict[str, list[Transport]], *, probe=False) -> None:
        self._ladders = ladders
        self._probe = probe
        self.worker_directly_routable = True
        self.route_ttl_sec = 30.0
        self.observed: list[tuple[str, Transport, RouteObservationOutcome]] = []
        self.resolved: list[tuple[str, PolicyClass, str]] = []

    async def endpoint_for(self, node_id: str) -> Any:
        return SimpleNamespace(url=f"10.0.0.{node_id[-1]}:9101")

    def peer_trust(self, protocol: str) -> TrustedPeerPolicy:
        return TrustedPeerPolicy(enabled=True, trust_domain="fm", protocol=protocol)

    async def resolve(self, origin_node_id, target, *, trust, policy_class) -> Any:
        self.resolved.append((target.worker_id, policy_class, trust.protocol))
        ladder = self._ladders.get(target.worker_id, [Transport.CONTROL_RELAY])
        if not trust.enabled:
            ladder = [Transport.CONTROL_RELAY]
        route = ResolvedRoute(
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
        return SimpleNamespace(origin_id="rog-1", origin_node=origin_node_id), route

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
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))
    asyncio.run(authority.record_holding("wkr-3", [_HELD]))

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))

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
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))

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
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE, True))

    assert workers.kinds() == ["content_grant_denied"]
    assert _denial(workers) == HydrationDenial.NO_PEER_ROUTE
    assert relay.hashes == {}


def test_a_requester_that_dials_no_peer_is_routed_over_the_relay() -> None:
    workers = _Workers(**{"wkr-1": 3, "wkr-2": 5})
    workers.listening = {"wkr-1"}
    network = _Network({"wkr-1": [Transport.WORKER_DIRECT]})
    authority = _authority(workers, network=network)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))

    assert _frame(workers, "content_grant")["carriage"]["routes"] == []
    assert _frame(workers, "content_serve_grant")["requester_host"] == ""


def test_a_probe_route_never_carries_a_transfer() -> None:
    workers = _peers(**{"wkr-1": 3, "wkr-2": 5})
    network = _Network({"wkr-1": [Transport.WORKER_DIRECT]}, probe=True)
    authority = _authority(workers, network=network)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))

    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE, True))

    assert _denial(workers) == HydrationDenial.NO_PEER_ROUTE


def test_only_the_routed_requester_reports_on_a_transfers_path() -> None:
    workers = _peers(**{"wkr-1": 3, "wkr-2": 5})
    network = _Network({"wkr-1": [Transport.WORKER_DIRECT, Transport.CONTROL_RELAY]})
    authority = _authority(workers, network=network)
    asyncio.run(authority.record_holding("wkr-1", [_HELD]))
    asyncio.run(authority.authorize("wkr-2", "tsk-1", _REFERENCE))
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
        await loop.run_in_executor(None, lambda: authority.on_holding("wkr-1", [_HELD]))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if workers.frames:
                break
            await loop.run_in_executor(
                None, lambda: authority.on_request("wkr-2", "tsk-1", _REFERENCE)
            )

    asyncio.run(run())
    assert workers.kinds()[:2] == ["content_serve_grant", "content_grant"]
