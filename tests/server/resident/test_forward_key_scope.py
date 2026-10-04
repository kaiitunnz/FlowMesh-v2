"""The deployment forward key reaches only a dev_model replica's sidecar."""

import asyncio

import pytest

from tests.server.resident.node_harness import Node, admitted_boundary

FORWARD_KEY = "deployment-forward-key"


def _bind_keys(node: Node) -> list[str | None]:
    return [
        p["engine"]["api_key"]
        for _w, kind, p in node.delivery.relays
        if kind == "resident_sidecar_bind"
    ]


@pytest.mark.parametrize(
    ("substrate", "expected"), [("serve", None), ("dev_model", FORWARD_KEY)]
)
def test_only_a_dev_model_replica_is_bound_with_the_forward_key(
    substrate: str, expected: str | None
) -> None:
    async def run() -> None:
        node = Node(substrate=substrate, forward_api_key=FORWARD_KEY)
        await admitted_boundary(node)
        keys = _bind_keys(node)
        assert keys and all(key == expected for key in keys)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("substrate", "expected"), [("serve", None), ("dev_model", FORWARD_KEY)]
)
def test_a_restart_reattaches_the_forward_key_only_to_a_dev_model_replica(
    substrate: str, expected: str | None
) -> None:
    node = Node(substrate=substrate, forward_api_key=FORWARD_KEY)
    warm = node.warm()
    node.persist()

    node.restart()

    endpoint = node.replica(warm.replica_id).endpoint
    assert endpoint is not None and endpoint.api_key == expected
