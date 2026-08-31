"""The inference adapter delivers a request and classifies delivery failures.

It returns the replica's completion on success, maps a connection refusal to a
pre-acceptance failure (releasing the credit), and maps a mid-request loss to a
post-acceptance failure (entering reconciliation).
"""

import asyncio

import httpx
import pytest

from server.resident import AdmissionHandoff, HttpInferenceAdapter, ReplicaEndpoint
from server.resident.adapter import AdapterError

_HANDOFF = AdmissionHandoff(
    token="hnd-x",
    claim_id="scl-x",
    invocation_id="inv-x",
    family="fam",
    replica_id="rpl-1",
    incarnation=1,
    endpoint=ReplicaEndpoint(base_url="http://replica/v1", model="m", api_key="k"),
)


def _adapter(handler):
    return HttpInferenceAdapter(transport=httpx.MockTransport(handler))


def test_issue_returns_completion_and_authenticates():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "hello world"}}]}
        )

    out = asyncio.run(_adapter(handler).issue(_HANDOFF, '{"prompt": "hi"}'))
    assert out == "hello world"
    assert seen["url"] == "http://replica/v1/chat/completions"
    assert seen["auth"] == "Bearer k"


def test_connection_refusal_is_pre_acceptance():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(AdapterError) as excinfo:
        asyncio.run(_adapter(handler).issue(_HANDOFF, "hi"))
    assert excinfo.value.pre_acceptance is True


def test_mid_request_loss_is_post_acceptance():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("lost")

    with pytest.raises(AdapterError) as excinfo:
        asyncio.run(_adapter(handler).issue(_HANDOFF, "hi"))
    assert excinfo.value.pre_acceptance is False


def test_malformed_body_is_post_acceptance():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(AdapterError) as excinfo:
        asyncio.run(_adapter(handler).issue(_HANDOFF, "hi"))
    assert excinfo.value.pre_acceptance is False
