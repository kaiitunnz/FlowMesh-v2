"""The inference adapter delivers a request and classifies delivery failures.

It returns the replica's completion on success, maps a connection refusal to a
pre-acceptance failure (releasing the credit), and maps a mid-request loss to a
post-acceptance failure (entering reconciliation).
"""

import asyncio
import json

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


def test_connection_refusal_is_pre_acceptance_connection_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(AdapterError) as excinfo:
        asyncio.run(_adapter(handler).issue(_HANDOFF, "hi"))
    assert excinfo.value.pre_acceptance is True
    assert excinfo.value.connection_failure is True


def test_http_status_is_pre_acceptance_but_not_a_connection_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "overloaded"})

    with pytest.raises(AdapterError) as excinfo:
        asyncio.run(_adapter(handler).issue(_HANDOFF, "hi"))
    assert excinfo.value.pre_acceptance is True
    assert excinfo.value.connection_failure is False


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


def test_forward_key_is_presented_when_the_replica_reports_none():
    keyless = _HANDOFF.model_copy(
        update={
            "endpoint": ReplicaEndpoint(
                base_url="http://replica/v1", model="m", api_key=None
            )
        }
    )
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    adapter = HttpInferenceAdapter(
        transport=httpx.MockTransport(handler), forward_api_key="fwd-secret"
    )
    asyncio.run(adapter.issue(keyless, "hi"))
    assert seen["auth"] == "Bearer fwd-secret"


def test_full_chat_request_is_forwarded_faithfully():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    payload = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            "temperature": 0.1,
        }
    )
    asyncio.run(_adapter(handler).issue(_HANDOFF, payload))
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["temperature"] == 0.1
    assert seen["body"]["model"] == "m"  # the replica's model is pinned
