"""The replica sidecar lane gates a fence and serves its engine over the session.

An origin session wired sink-to-sink with the sidecar drives a real bootstrap, ack, and
authorized stream; a fence naming the wrong incarnation is refused before any engine
call; a definite engine 4xx is carried as a definite failure so the origin releases.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import pytest

from shared.network.relay_frame import RelayFrame
from shared.network.session import FramedRelaySession, RelaySessionRole
from shared.resident.contracts import (
    AdmissionHandoff,
    ReplicaEndpoint,
    RouteAuthorization,
)
from shared.resident.envelope import ServeRequestEnvelope, freeze_request_envelope
from shared.resident.wire import (
    KIND_ACK,
    KIND_CHUNK,
    KIND_DONE,
    KIND_FAILED,
    KIND_HEAD,
    KIND_REJECT,
)
from worker.resident.engine import (
    EngineResponse,
    HttpEngineDelivery,
    RawEngineResponse,
)
from worker.resident.replica_sidecar import ResidentReplicaSidecar

_CHUNKS = ["resi", "dent ", "reply"]

_SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
    b"data: [DONE]\n\n",
]

_SERVE_TASK = "tsk-serve"


class _RawEngine:
    """A raw serve delivery stand-in that records the envelope it was replayed with."""

    def __init__(self, status: int, content_type: str, parts: list[bytes]) -> None:
        self._status = status
        self._content_type = content_type
        self._parts = parts
        self.seen: list[ServeRequestEnvelope] = []

    async def __call__(
        self, endpoint: ReplicaEndpoint, envelope: ServeRequestEnvelope
    ) -> RawEngineResponse:
        self.seen.append(envelope)
        parts = self._parts

        async def chunks() -> AsyncIterator[bytes]:
            for part in parts:
                yield part

        async def aclose() -> None:
            return None

        return RawEngineResponse(
            status=self._status,
            headers=(("content-type", self._content_type),),
            chunks=chunks(),
            aclose=aclose,
        )


def _make_raw_engine(status: int, content_type: str, parts: list[bytes]) -> _RawEngine:
    return _RawEngine(status, content_type, parts)


async def _fake_engine(
    endpoint: ReplicaEndpoint,
    request: str | None,
    adapter_name: str | None = None,
    adapter_source: str | None = None,
) -> EngineResponse:
    async def chunks() -> AsyncIterator[str]:
        for part in _CHUNKS:
            yield part

    async def aclose() -> None:
        return None

    return EngineResponse(chunks=chunks(), aclose=aclose)


async def _failing_engine(
    endpoint: ReplicaEndpoint,
    request: str | None,
    adapter_name: str | None = None,
    adapter_source: str | None = None,
) -> EngineResponse:
    response = httpx.Response(400, request=httpx.Request("POST", "http://engine/v1"))
    raise httpx.HTTPStatusError(
        "bad request", request=response.request, response=response
    )


def _http_engine(
    reply: Callable[[httpx.Request], httpx.Response],
) -> HttpEngineDelivery:
    delivery = HttpEngineDelivery()
    delivery._clients[None] = httpx.AsyncClient(transport=httpx.MockTransport(reply))
    return delivery


def _answer(
    status: int = 200, **body: Any
) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(status, **body)


def _unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


class _ToPeer:
    """Forwards each produced frame into a target coroutine."""

    def __init__(self) -> None:
        self.on_peer: Callable[[RelayFrame], Awaitable[None]] | None = None

    async def send(self, frame: RelayFrame) -> None:
        assert self.on_peer is not None
        await self.on_peer(frame)


def _handoff(incarnation: int = 1) -> dict:
    return AdmissionHandoff(
        token="hnd-1",
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family="fam",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=incarnation,
        listener_generation=1,
    ).model_dump(mode="json")


def _auth() -> dict:
    return RouteAuthorization(
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
    ).model_dump(mode="json")


def _harness(
    engine=_fake_engine, interface: str = "chat"
) -> tuple[FramedRelaySession, ResidentReplicaSidecar]:
    origin_sink, replica_sink = _ToPeer(), _ToPeer()
    sidecar = ResidentReplicaSidecar(sink=replica_sink, engine_open=engine)
    sidecar.bind(
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
        endpoint=ReplicaEndpoint(
            base_url="http://engine/v1", model="m", interface=interface
        ),
    )
    origin = FramedRelaySession(
        session_id="s1",
        correlation_id="inv-1",
        operation_id="idm-1",
        role=RelaySessionRole.ORIGIN,
        sink=origin_sink,
    )
    origin_sink.on_peer = sidecar.on_frame
    replica_sink.on_peer = origin.on_frame
    return origin, sidecar


def test_gated_invocation_streams_from_the_engine() -> None:
    async def run() -> None:
        origin, sidecar = _harness()
        await origin.send_wire("bootstrap", handoff=_handoff(), request='{"p":"hi"}')
        ack = await origin.recv_wire(timeout=5.0)
        assert ack is not None and ack["kind"] == KIND_ACK
        await origin.send_wire("stream", auth=_auth())
        parts: list[str] = []
        while True:
            msg = await origin.recv_wire(timeout=5.0)
            assert msg is not None
            if msg["kind"] == KIND_CHUNK:
                parts.append(str(msg["data"]))
            elif msg["kind"] == KIND_DONE:
                break
        assert "".join(parts) == "".join(_CHUNKS)
        await sidecar.aclose()

    asyncio.run(run())


def test_wrong_incarnation_is_refused_before_the_engine() -> None:
    async def run() -> None:
        origin, sidecar = _harness()
        await origin.send_wire(
            "bootstrap", handoff=_handoff(incarnation=9), request=None
        )
        reply = await origin.recv_wire(timeout=5.0)
        assert reply is not None and reply["kind"] == KIND_REJECT
        assert reply["reason"] == "wrong_incarnation"
        await sidecar.aclose()

    asyncio.run(run())


def test_definite_engine_failure_is_carried_definite() -> None:
    async def run() -> None:
        origin, sidecar = _harness(engine=_failing_engine)
        await origin.send_wire("bootstrap", handoff=_handoff(), request=None)
        ack = await origin.recv_wire(timeout=5.0)
        assert ack is not None and ack["kind"] == KIND_ACK
        await origin.send_wire("stream", auth=_auth())
        failed = await origin.recv_wire(timeout=5.0)
        assert failed is not None and failed["kind"] == KIND_FAILED
        assert failed["definite"] is True
        await sidecar.aclose()

    asyncio.run(run())


async def _stream_outcome(
    engine: Callable[..., Awaitable[EngineResponse]],
    request: str | None = None,
    interface: str = "chat",
) -> dict[str, Any] | None:
    origin, sidecar = _harness(engine=engine, interface=interface)
    await origin.send_wire("bootstrap", handoff=_handoff(), request=request)
    ack = await origin.recv_wire(timeout=5.0)
    assert ack is not None and ack["kind"] == KIND_ACK
    await origin.send_wire("stream", auth=_auth())
    outcome = await origin.recv_wire(timeout=1.0)
    await sidecar.aclose()
    return outcome


@pytest.mark.parametrize(
    ("reply", "interface"),
    [
        (_answer(json={"choices": []}), "chat"),
        (_answer(text="not json"), "chat"),
        (_answer(json=["not", "an", "object"]), "chat"),
        (_answer(json={"choices": [{"text": "hi"}]}), "chat"),
        (_answer(json={"choices": [{"message": {"role": "assistant"}}]}), "chat"),
        (_answer(json={"choices": ["hi"]}), "chat"),
        (_answer(json={"choices": [{"message": "hi"}]}), "chat"),
        (_answer(json={"object": "list"}), "embedding"),
    ],
    ids=[
        "no-choice",
        "non-json-body",
        "non-object-body",
        "choice-without-message",
        "message-without-content",
        "non-object-choice",
        "non-object-message",
        "embeddings-without-data",
    ],
)
def test_an_engine_response_the_replica_cannot_read_fails_definite(
    reply: Callable[[httpx.Request], httpx.Response], interface: str
) -> None:
    # The engine answered, so the boundary settles as a definite failure and releases
    # its credit.
    outcome = asyncio.run(
        _stream_outcome(_http_engine(reply), request="hi", interface=interface)
    )
    assert outcome is not None and outcome["kind"] == KIND_FAILED
    assert outcome["definite"] is True


def test_a_request_the_replica_cannot_build_fails_definite() -> None:
    reply = _answer(json={"choices": [{"message": {"content": "hi"}}]})
    malformed_batch = '[{"prompt": "no messages"}]'
    outcome = asyncio.run(_stream_outcome(_http_engine(reply), malformed_batch))
    assert outcome is not None and outcome["kind"] == KIND_FAILED
    assert outcome["definite"] is True


_OVER_WINDOW = (
    "This model's maximum context length is 1024 tokens. However, you requested "
    "1000 output tokens and your prompt contains 30 input tokens, for a total of "
    "1030 tokens. Please reduce the length of the input prompt or the number of "
    "requested output tokens."
)


def _engine_error(
    status: int, message: str, kind: str, param: Any
) -> Callable[[httpx.Request], httpx.Response]:
    return _answer(
        status,
        json={"error": {"message": message, "type": kind, "param": param, "code": 400}},
    )


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (
            _engine_error(400, _OVER_WINDOW, "BadRequestError", "input_tokens"),
            f"engine 400: {_OVER_WINDOW}",
        ),
        (
            _engine_error(
                400,
                "1 validation error:\n  {'type': 'string_type', 'loc': ('body', "
                "'messages'), 'input': 'the tenant prompt'}",
                "BadRequestError",
                "messages",
            ),
            "engine 400 BadRequestError",
        ),
        (
            _engine_error(400, "the tenant prompt", "Bad Request\nInjected", None),
            "engine 400",
        ),
        (_answer(400, text="<html>bad request</html>"), "engine 400"),
        (
            _engine_error(500, "the tenant prompt", "InternalServerError", None),
            "engine 500 InternalServerError",
        ),
        (
            _engine_error(400, "the tenant prompt", "BadRequestError", ["messages"]),
            "engine 400 BadRequestError",
        ),
        (
            _engine_error(400, "the tenant prompt", "BadRequestError", {"a": 1}),
            "engine 400 BadRequestError",
        ),
        (_answer(400, json=["the tenant prompt"]), "engine 400"),
    ],
    ids=[
        "over-window",
        "echoing-validation",
        "unsafe-type",
        "non-json",
        "server",
        "list-param",
        "dict-param",
        "list-body",
    ],
)
def test_an_engine_refusal_names_only_what_the_engine_may_disclose(
    reply: Callable[[httpx.Request], httpx.Response], reason: str
) -> None:
    outcome = asyncio.run(_stream_outcome(_http_engine(reply), request="hi"))
    assert outcome is not None and outcome["kind"] == KIND_FAILED
    assert outcome["reason"] == reason


def test_an_engine_refusal_message_is_bounded_to_one_line() -> None:
    message = _OVER_WINDOW + "\n\x1b[31m" + "x" * 1000
    reply = _engine_error(400, message, "BadRequestError", "input_tokens")
    outcome = asyncio.run(_stream_outcome(_http_engine(reply), request="hi"))
    assert outcome is not None
    reason = outcome["reason"]
    assert reason.startswith(f"engine 400: {_OVER_WINDOW}")
    assert len(reason) <= len("engine 400: ") + 300
    assert reason.isprintable()


def _batch(*prompts: str) -> str:
    return json.dumps([{"messages": [{"role": "user", "content": p}]} for p in prompts])


def _per_conversation(request: httpx.Request) -> httpx.Response:
    prompt = json.loads(request.content)["messages"][-1]["content"]
    if prompt == "lost":
        raise httpx.ReadTimeout("engine stalled", request=request)
    if prompt == "refused":
        return httpx.Response(400, json={"error": "bad request"})
    return httpx.Response(200, text="not json")


@pytest.mark.parametrize("definite", ["refused", "unreadable"])
@pytest.mark.parametrize("lost_first", [True, False])
def test_a_batch_with_a_lost_conversation_is_not_a_definite_failure(
    definite: str, lost_first: bool
) -> None:
    # The lost conversation may still be generating, so the boundary holds its credit
    # whichever order the conversations were declared in.
    prompts = ("lost", definite) if lost_first else (definite, "lost")
    outcome = asyncio.run(
        _stream_outcome(_http_engine(_per_conversation), _batch(*prompts))
    )
    assert outcome is None or outcome.get("definite") is False


def test_an_unreachable_engine_is_not_a_definite_failure() -> None:
    outcome = asyncio.run(_stream_outcome(_http_engine(_unreachable), "hi"))
    assert outcome is None or outcome.get("definite") is False


def _serve_envelope(
    method: str = "POST",
    path: str = "v1/chat/completions",
    body: bytes = b"{}",
    query: str = "",
) -> ServeRequestEnvelope:
    return freeze_request_envelope(
        method=method,
        upstream_path=path,
        query=query,
        headers=[("content-type", "application/json")],
        body=body,
    )


def _serve_handoff(envelope: ServeRequestEnvelope) -> dict:
    return AdmissionHandoff(
        token="hnd-1",
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        family=f"serve/{_SERVE_TASK}",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
        serve_task_id=_SERVE_TASK,
        binding_generation=0,
        descriptor_digest=envelope.digest(),
    ).model_dump(mode="json")


def _serve_auth() -> dict:
    return RouteAuthorization(
        claim_id="scl-1",
        invocation_id="inv-1",
        idempotency_key="idm-1",
        tenant="t1",
        origin_id="rog-1",
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
        serve_task_id=_SERVE_TASK,
        binding_generation=0,
    ).model_dump(mode="json")


def _serve_harness(
    engine_open_raw: Callable[..., Awaitable[RawEngineResponse]],
) -> tuple[FramedRelaySession, ResidentReplicaSidecar]:
    origin_sink, replica_sink = _ToPeer(), _ToPeer()
    sidecar = ResidentReplicaSidecar(
        sink=replica_sink, engine_open=_fake_engine, engine_open_raw=engine_open_raw
    )
    sidecar.bind(
        replica_id="rpl-1",
        incarnation=1,
        listener_generation=1,
        endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
        serve_task_id=_SERVE_TASK,
        binding_generation=0,
    )
    origin = FramedRelaySession(
        session_id="s1",
        correlation_id="inv-1",
        operation_id="idm-1",
        role=RelaySessionRole.ORIGIN,
        sink=origin_sink,
    )
    origin_sink.on_peer = sidecar.on_frame
    replica_sink.on_peer = origin.on_frame
    return origin, sidecar


def test_serve_invocation_reverse_proxies_the_raw_engine_response() -> None:
    async def run() -> None:
        envelope = _serve_envelope(
            body=b'{"messages":[{"role":"user","content":"hi"}],"stream":true}'
        )
        engine = _make_raw_engine(200, "text/event-stream", _SSE_CHUNKS)
        origin, sidecar = _serve_harness(engine)
        await origin.send_body_wire(
            "bootstrap",
            envelope.body,
            handoff=_serve_handoff(envelope),
            request=envelope.header_fields(),
        )
        ack = await origin.recv_wire(timeout=5.0)
        assert ack is not None and ack["kind"] == KIND_ACK
        await origin.send_wire("stream", auth=_serve_auth())
        head = await origin.recv_wire(timeout=5.0)
        # The engine response head is relayed before the body: the raw serve path frames
        # the engine's own status and content type, which the parsed path never emits.
        assert head is not None and head["kind"] == KIND_HEAD
        assert head["status"] == 200
        assert head["headers"] == [["content-type", "text/event-stream"]]
        parts: list[bytes] = []
        while True:
            received = await origin.recv_body_wire(timeout=5.0)
            assert received is not None
            msg, body = received
            if msg["kind"] == KIND_CHUNK:
                parts.append(body)
            elif msg["kind"] == KIND_DONE:
                break
        assert b"".join(parts) == b"".join(_SSE_CHUNKS)
        # The sidecar replays the client's frozen envelope rather than rebuilding it.
        assert engine.seen[0].path == "/v1/chat/completions"
        assert engine.seen[0].body == envelope.body
        await sidecar.aclose()

    asyncio.run(run())


def test_serve_relays_an_engine_error_status_without_failing() -> None:
    async def run() -> None:
        envelope = _serve_envelope(
            body=b'{"messages":[{"role":"user","content":"hi"}]}'
        )
        origin, sidecar = _serve_harness(
            _make_raw_engine(400, "application/json", [b'{"error":"bad request"}'])
        )
        await origin.send_body_wire(
            "bootstrap",
            envelope.body,
            handoff=_serve_handoff(envelope),
            request=envelope.header_fields(),
        )
        ack = await origin.recv_wire(timeout=5.0)
        assert ack is not None and ack["kind"] == KIND_ACK
        await origin.send_wire("stream", auth=_serve_auth())
        head = await origin.recv_wire(timeout=5.0)
        # A raw proxy relays the engine's 4xx as a response, not a fenced failure: the
        # client sees the engine's own error envelope, and the credit still releases on
        # the terminal.
        assert head is not None and head["kind"] == KIND_HEAD
        assert head["status"] == 400
        parts: list[bytes] = []
        terminal = None
        while True:
            received = await origin.recv_body_wire(timeout=5.0)
            assert received is not None
            msg, chunk = received
            if msg["kind"] == KIND_CHUNK:
                parts.append(chunk)
            else:
                terminal = msg
                break
        assert terminal is not None and terminal["kind"] == KIND_DONE
        assert b"".join(parts) == b'{"error":"bad request"}'
        await sidecar.aclose()

    asyncio.run(run())


def _raw_engine_stream_fail() -> Callable[..., Awaitable[RawEngineResponse]]:
    async def engine(
        endpoint: ReplicaEndpoint, envelope: ServeRequestEnvelope
    ) -> RawEngineResponse:
        async def chunks() -> AsyncIterator[bytes]:
            yield b"data: partial\n\n"
            raise RuntimeError("engine stream blew up")

        async def aclose() -> None:
            return None

        return RawEngineResponse(
            status=200,
            headers=(("content-type", "text/event-stream"),),
            chunks=chunks(),
            aclose=aclose,
        )

    return engine


def _raw_engine_open_boom() -> Callable[..., Awaitable[RawEngineResponse]]:
    async def engine(
        endpoint: ReplicaEndpoint, envelope: ServeRequestEnvelope
    ) -> RawEngineResponse:
        raise RuntimeError("unexpected open failure")

    return engine


def test_serve_rejects_an_envelope_mutated_after_admission() -> None:
    # The gate recomputes the descriptor from the relayed envelope, so altering any part
    # of the request in flight — here the path — is refused before any engine I/O.
    async def run() -> None:
        admitted = _serve_envelope(body=b'{"messages":[]}')
        engine = _make_raw_engine(200, "application/json", [b"{}"])
        origin, sidecar = _serve_harness(engine)
        tampered = admitted.model_copy(update={"path": "/v1/embeddings"})
        await origin.send_body_wire(
            "bootstrap",
            tampered.body,
            handoff=_serve_handoff(admitted),
            request=tampered.header_fields(),
        )
        reply = await origin.recv_wire(timeout=5.0)
        assert reply is not None and reply["kind"] == KIND_REJECT
        assert reply["reason"] == "wrong_digest"
        assert engine.seen == []
        await sidecar.aclose()

    asyncio.run(run())


def test_serve_rejects_a_malformed_envelope_before_any_engine_call() -> None:
    async def run() -> None:
        envelope = _serve_envelope()
        engine = _make_raw_engine(200, "application/json", [b"{}"])
        origin, sidecar = _serve_harness(engine)
        await origin.send_body_wire(
            "bootstrap",
            envelope.body,
            handoff=_serve_handoff(envelope),
            request="not-an-envelope",
        )
        reply = await origin.recv_wire(timeout=5.0)
        assert reply is not None and reply["kind"] == KIND_REJECT
        assert engine.seen == []
        await sidecar.aclose()

    asyncio.run(run())


def test_serve_terminates_on_a_stream_error_after_the_head() -> None:
    async def run() -> None:
        origin, sidecar = _serve_harness(_raw_engine_stream_fail())
        envelope = _serve_envelope(body=b'{"messages":[]}')
        await origin.send_body_wire(
            "bootstrap",
            envelope.body,
            handoff=_serve_handoff(envelope),
            request=envelope.header_fields(),
        )
        ack = await origin.recv_wire(timeout=5.0)
        assert ack is not None and ack["kind"] == KIND_ACK
        await origin.send_wire("stream", auth=_serve_auth())
        head = await origin.recv_wire(timeout=5.0)
        assert head is not None and head["kind"] == KIND_HEAD
        # An unexpected error after the head still emits a terminal rather than dying
        # silently and leaving the origin to stall on the stream deadline.
        terminal = None
        while True:
            received = await origin.recv_body_wire(timeout=5.0)
            assert received is not None
            msg, _chunk = received
            if msg["kind"] in (KIND_DONE, KIND_FAILED):
                terminal = msg
                break
        assert terminal["kind"] == KIND_FAILED
        assert terminal["definite"] is False
        await sidecar.aclose()

    asyncio.run(run())


def test_serve_terminates_on_an_unexpected_open_error() -> None:
    async def run() -> None:
        origin, sidecar = _serve_harness(_raw_engine_open_boom())
        envelope = _serve_envelope(body=b'{"messages":[]}')
        await origin.send_body_wire(
            "bootstrap",
            envelope.body,
            handoff=_serve_handoff(envelope),
            request=envelope.header_fields(),
        )
        ack = await origin.recv_wire(timeout=5.0)
        assert ack is not None and ack["kind"] == KIND_ACK
        await origin.send_wire("stream", auth=_serve_auth())
        # An unexpected open failure emits a terminal, not a silent session death.
        failed = await origin.recv_wire(timeout=5.0)
        assert failed is not None and failed["kind"] == KIND_FAILED
        await sidecar.aclose()

    asyncio.run(run())


def test_serve_bootstrap_is_refused_when_the_replica_serves_another_task() -> None:
    async def run() -> None:
        origin_sink, replica_sink = _ToPeer(), _ToPeer()
        sidecar = ResidentReplicaSidecar(
            sink=replica_sink,
            engine_open=_fake_engine,
            engine_open_raw=_make_raw_engine(200, "text/event-stream", _SSE_CHUNKS),
        )
        # Bound for a different serve task: the fence names tsk-serve, this replica
        # serves tsk-other, so the gate refuses before any engine call.
        sidecar.bind(
            replica_id="rpl-1",
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
            serve_task_id="tsk-other",
            binding_generation=0,
        )
        origin = FramedRelaySession(
            session_id="s1",
            correlation_id="inv-1",
            operation_id="idm-1",
            role=RelaySessionRole.ORIGIN,
            sink=origin_sink,
        )
        origin_sink.on_peer = sidecar.on_frame
        replica_sink.on_peer = origin.on_frame
        envelope = _serve_envelope(body=b'{"messages":[]}')
        await origin.send_body_wire(
            "bootstrap",
            envelope.body,
            handoff=_serve_handoff(envelope),
            request=envelope.header_fields(),
        )
        reply = await origin.recv_wire(timeout=5.0)
        assert reply is not None and reply["kind"] == KIND_REJECT
        assert reply["reason"] == "wrong_serve_task"
        await sidecar.aclose()

    asyncio.run(run())


def test_not_yet_bound_signals_a_transient_loss_not_a_definite_reject() -> None:
    async def run() -> None:
        origin_sink, replica_sink = _ToPeer(), _ToPeer()
        # No bind for rpl-1: the bind frame has not arrived yet (a cold-start race).
        sidecar = ResidentReplicaSidecar(sink=replica_sink, engine_open=_fake_engine)
        origin = FramedRelaySession(
            session_id="s1",
            correlation_id="inv-1",
            operation_id="idm-1",
            role=RelaySessionRole.ORIGIN,
            sink=origin_sink,
        )
        origin_sink.on_peer = sidecar.on_frame
        replica_sink.on_peer = origin.on_frame
        await origin.send_wire("bootstrap", handoff=_handoff(), request='{"p":"hi"}')
        reply = await origin.recv_wire(timeout=5.0)
        # Transient, not a definite fence reject: the origin holds and re-drives.
        assert reply is not None and reply["kind"] == KIND_FAILED
        assert reply["definite"] is False
        await sidecar.aclose()

    asyncio.run(run())


def test_unload_adapter_calls_the_engine_for_a_bound_replica() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []

        async def fake_unload(endpoint: ReplicaEndpoint, name: str) -> None:
            calls.append((endpoint.base_url, name))

        sink = _ToPeer()
        sidecar = ResidentReplicaSidecar(
            sink=sink, engine_open=_fake_engine, engine_unload=fake_unload
        )
        sidecar.bind(
            replica_id="rpl-1",
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
        )
        await sidecar.unload_adapter("rpl-1", "my-lora")
        assert calls == [("http://engine/v1", "my-lora")]

        # An unbound replica is a no-op: nothing to unload against.
        await sidecar.unload_adapter("rpl-unknown", "my-lora")
        assert calls == [("http://engine/v1", "my-lora")]

    asyncio.run(run())


def test_unload_adapter_swallows_an_engine_error() -> None:
    async def run() -> None:
        async def failing_unload(endpoint: ReplicaEndpoint, name: str) -> None:
            request = httpx.Request("POST", "http://engine/v1/unload_lora_adapter")
            raise httpx.HTTPStatusError(
                "boom",
                request=request,
                response=httpx.Response(500, request=request),
            )

        sidecar = ResidentReplicaSidecar(
            sink=_ToPeer(), engine_open=_fake_engine, engine_unload=failing_unload
        )
        sidecar.bind(
            replica_id="rpl-1",
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(base_url="http://engine/v1", model="m"),
        )
        # Best effort: a failed unload does not raise out of the lane.
        await sidecar.unload_adapter("rpl-1", "my-lora")

    asyncio.run(run())


def test_reap_invocation_tears_down_the_inflight_serve() -> None:
    async def run() -> None:
        aclosed = asyncio.Event()

        async def hanging_engine(
            endpoint: ReplicaEndpoint,
            request: str | None,
            adapter_name: str | None = None,
            adapter_source: str | None = None,
        ) -> EngineResponse:
            async def chunks() -> AsyncIterator[str]:
                await asyncio.Event().wait()  # never yields — the engine is slow
                yield ""  # pragma: no cover

            async def aclose() -> None:
                aclosed.set()

            return EngineResponse(chunks=chunks(), aclose=aclose)

        origin, sidecar = _harness(engine=hanging_engine)
        await origin.send_wire("bootstrap", handoff=_handoff(), request='{"p":"hi"}')
        ack = await origin.recv_wire(timeout=5.0)
        assert ack is not None and ack["kind"] == KIND_ACK
        await origin.send_wire("stream", auth=_auth())
        for _ in range(100):
            await asyncio.sleep(0.01)
            if sidecar._inflight.get("inv-1") is not None:
                break
        assert sidecar._inflight.get("inv-1") is not None  # in-flight

        # A fenced terminal reaps the serve task: teardown closes the engine request.
        sidecar.reap_invocation("inv-1")
        await asyncio.wait_for(aclosed.wait(), timeout=5.0)
        await sidecar.aclose()

    asyncio.run(run())


def test_a_claim_on_a_withdrawn_engine_re_drives_as_an_unreachable_engine_does() -> (
    None
):
    # An engine withdrawn and unbound answers a later claim with the same transient
    # loss an unreachable engine leaves: never a definite failure, so the origin holds
    # the credit and control re-drives.
    async def run() -> dict[str, Any] | None:
        origin, sidecar = _harness()
        sidecar.bind(
            replica_id="rpl-1",
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(
                base_url="http://localhost/v1", model="m", socket_path="/run/a.sock"
            ),
        )
        sidecar.bind(
            replica_id="rpl-2",
            incarnation=1,
            listener_generation=1,
            endpoint=ReplicaEndpoint(
                base_url="http://localhost/v1", model="m", socket_path="/run/b.sock"
            ),
        )
        sidecar.unbind_engine("/run/a.sock")
        assert set(sidecar._bindings) == {"rpl-2"}
        await origin.send_wire("bootstrap", handoff=_handoff(), request='{"p":"hi"}')
        reply = await origin.recv_wire(timeout=5.0)
        await sidecar.aclose()
        return reply

    withdrawn = asyncio.run(run())
    assert withdrawn is not None and withdrawn["kind"] == KIND_FAILED
    assert withdrawn["definite"] is False
    unreachable = asyncio.run(_stream_outcome(_http_engine(_unreachable), "hi"))
    assert unreachable is None or unreachable.get("definite") is False
