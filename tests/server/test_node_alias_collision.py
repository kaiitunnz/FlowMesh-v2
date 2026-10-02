"""A starting node tells a crashed holder of its alias from a live one by whether the
holder's lease is refreshed between two refusals."""

import logging
from typing import cast

import fakeredis
import pytest

from server.clients.redis import node_alias_lease_key
from server.config import NodeRole
from server.hooks import PrincipalContext
from server.registries.node import NodeRegistry
from server.supervisor.services import lifecycle as lifecycle_module
from server.supervisor.services.lifecycle import Lifecycle, NodeAliasConflictError
from shared.schemas.node import NodeInfo
from tests.server.redis_helpers import fake_redis_client

_TTL_SEC = 120
_HB_SEC = 30
_ALIAS = "gpu-a"
_LEASE = node_alias_lease_key(_ALIAS)
_LOGGER = logging.getLogger("test.node_alias_collision")
_ACTOR = PrincipalContext(
    principal_id="system",
    org_id="org-1",
    external_id="system",
    principal_type="service",
    scopes=[],
)


def _info() -> NodeInfo:
    return NodeInfo(
        namespace="ns",
        cluster="cl",
        alias=_ALIAS,
        version="0.1.0",
        started_at="2026-10-02T00:00:00Z",
        tags=[],
        last_seen="2026-10-02T00:00:00Z",
        max_gpu_count=0,
    )


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def rds(server: fakeredis.FakeServer) -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(server=server, decode_responses=True)


@pytest.fixture
def registry(server: fakeredis.FakeServer) -> NodeRegistry:
    return NodeRegistry(fake_redis_client(server), _LOGGER, _TTL_SEC)


def _starting_node(server: fakeredis.FakeServer, registry: NodeRegistry) -> Lifecycle:
    return Lifecycle(
        redis=fake_redis_client(server).sync,
        node_registry=registry,
        node_info=_info(),
        role=NodeRole.ROOT,
        base_url="",
        hb_sec=_HB_SEC,
        hb_ttl_sec=_TTL_SEC,
        logger=_LOGGER,
        system_principal=_ACTOR,
    )


def _age(rds: fakeredis.FakeRedis, seconds: float) -> None:
    remaining = cast(int, rds.pttl(_LEASE))
    rds.pexpire(_LEASE, max(1, remaining - int(seconds * 1000)))


def test_a_lease_refreshed_between_two_refusals_fails_registration(
    monkeypatch: pytest.MonkeyPatch,
    server: fakeredis.FakeServer,
    rds: fakeredis.FakeRedis,
    registry: NodeRegistry,
) -> None:
    holder = registry.register_node(_info())

    def wait(seconds: float) -> None:
        _age(rds, seconds)
        # The holder is alive: its heartbeat refreshes its lease.
        registry.update_node_hb(holder, "ts", _TTL_SEC)

    monkeypatch.setattr(lifecycle_module.time, "sleep", wait)

    with pytest.raises(NodeAliasConflictError, match="NODE_ALIAS 'gpu-a'"):
        _starting_node(server, registry)._register_until_alias_free()
    assert rds.hget(_LEASE, "node_id") == holder


def test_a_lease_that_only_runs_down_is_taken_over_at_half_its_ttl(
    monkeypatch: pytest.MonkeyPatch,
    server: fakeredis.FakeServer,
    rds: fakeredis.FakeRedis,
    registry: NodeRegistry,
) -> None:
    registry.register_node(_info())
    waited: list[float] = []

    def wait(seconds: float) -> None:
        # The holder crashed: nothing refreshes its lease.
        _age(rds, seconds)
        waited.append(seconds)

    monkeypatch.setattr(lifecycle_module.time, "sleep", wait)

    node_id = _starting_node(server, registry)._register_until_alias_free()

    assert rds.hget(_LEASE, "node_id") == node_id
    assert _TTL_SEC / 2 <= sum(waited) < _TTL_SEC / 2 + _HB_SEC
