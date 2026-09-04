"""Unit coverage for the worker-hosted external-tool executor.

The executor validates the operation fence and one-use delivery nonce, runs the provider
egress in this worker process, and returns one opaque reply frame — or none at all when
a crash leaves the boundary ambiguous. A fence failure returns a reject frame with no
provider call, and a delivery for a stale worker incarnation is rejected before egress.
"""

import threading
import time
from typing import Any

import pytest

from shared.tools.providers import SearchResult
from shared.tools.schema import (
    SEARCH_INTERFACE,
    RemoteToolOperationEnvelope,
    ToolRequest,
    tool_request_digest,
)
from shared.tools.wire import (
    FRAME_CANCEL,
    FRAME_OPERATION,
    FRAME_REAP,
    FRAME_REPLY,
    KIND_OPERATION,
    KIND_REJECT,
    KIND_RESULT,
    decode_msg,
    encode_msg,
)
from worker.external_tool_executor import WorkerExternalToolExecutor

WORKER = "wrk-1"
GEN = 7


class _Sink:
    def __init__(self) -> None:
        self.frames: list[tuple[str, str, bytes]] = []
        self._ev = threading.Event()

    def __call__(self, session_id: str, kind: str, frame: bytes) -> None:
        self.frames.append((session_id, kind, frame))
        self._ev.set()

    def wait(self, timeout: float = 2.0) -> bool:
        got = self._ev.wait(timeout)
        self._ev.clear()
        return got


def _fence(**over: Any) -> RemoteToolOperationEnvelope:
    base: dict[str, Any] = dict(
        interface=SEARCH_INTERFACE,
        provider="fake",
        idempotency_key="idm-1",
        request_digest=tool_request_digest(SEARCH_INTERFACE, "q", 3),
        target_id=WORKER,
        target_generation=GEN,
        delivery_nonce="xdn-1",
        policy_class="default",
        deadline_epoch=time.time() + 30,
        max_results=3,
        timeout_sec=5.0,
        result_char_cap=6000,
    )
    base.update(over)
    return RemoteToolOperationEnvelope(**base)


def _op(envelope: RemoteToolOperationEnvelope, query: str = "q") -> bytes:
    return encode_msg(
        KIND_OPERATION,
        envelope=envelope.model_dump(mode="json"),
        request=ToolRequest(
            interface=SEARCH_INTERFACE, query=query, max_results=3
        ).model_dump(mode="json"),
    )


def _executor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generation: int = GEN,
    raises: Exception | None = None,
    block: threading.Event | None = None,
) -> tuple[WorkerExternalToolExecutor, _Sink, list[str]]:
    calls: list[str] = []

    class _FakeProvider:
        def search(
            self, query: str, *, max_results: int, timeout_sec: float
        ) -> list[SearchResult]:
            calls.append(query)
            if block is not None:
                block.wait(2.0)
            if raises is not None:
                raise raises
            return [SearchResult(title="T", url="https://x", snippet="S")]

    monkeypatch.setattr(
        "shared.tools.providers.build_search_provider", lambda cfg: _FakeProvider()
    )
    sink = _Sink()
    ex = WorkerExternalToolExecutor(
        worker_id=WORKER,
        generation=generation,
        provider="fake",
        api_key=None,
        result_sink=sink,
    )
    return ex, sink, calls


def test_valid_operation_egresses_in_the_worker_and_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex, sink, calls = _executor(monkeypatch)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    assert sink.wait()
    session_id, kind, frame = sink.frames[0]
    assert (session_id, kind) == ("xtr-1", FRAME_REPLY)
    body = decode_msg(frame)
    assert body["kind"] == KIND_RESULT and body["outcome"]["status"] == "success"
    # The provider ran in this worker process, exactly once.
    assert calls == ["q"]


def test_replayed_nonce_is_rejected_before_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex, sink, calls = _executor(monkeypatch)
    op = _op(_fence())
    ex.submit("xtr-1", FRAME_OPERATION, op)
    assert sink.wait()
    ex.submit("xtr-2", FRAME_OPERATION, op)  # same one-use nonce
    assert sink.wait()
    assert decode_msg(sink.frames[1][2]) == {"kind": KIND_REJECT, "reason": "replay"}
    # The replay never reached the provider.
    assert calls == ["q"]


def test_stale_generation_is_rejected_before_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The worker came up at GEN + 1; a delivery stamped for GEN is fence-rejected.
    ex, sink, calls = _executor(monkeypatch, generation=GEN + 1)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence(target_generation=GEN)))
    assert sink.wait()
    assert decode_msg(sink.frames[0][2]) == {
        "kind": KIND_REJECT,
        "reason": "generation",
    }
    assert calls == []


def test_wrong_audience_and_digest_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex, sink, calls = _executor(monkeypatch)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence(target_id="wrk-other")))
    assert sink.wait()
    assert decode_msg(sink.frames[0][2])["reason"] == "audience"
    ex.submit("xtr-2", FRAME_OPERATION, _op(_fence(request_digest="deadbeef")))
    assert sink.wait()
    assert decode_msg(sink.frames[1][2])["reason"] == "digest"
    assert calls == []


def test_crashed_egress_sends_no_reply_leaving_it_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex, sink, _ = _executor(monkeypatch, raises=RuntimeError("boom"))
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    # No reply frame: the origin's carriage times out into an ambiguous delivery.
    assert sink.wait(0.5) is False


def test_cancel_reaps_and_fences_a_late_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = threading.Event()
    ex, sink, _ = _executor(monkeypatch, block=block)
    ex.submit("xtr-1", FRAME_OPERATION, _op(_fence()))
    ex.submit("xtr-1", FRAME_CANCEL, b"")
    assert sink.wait()
    assert sink.frames[0][1] == FRAME_REAP
    # Release the blocked egress: its late reply is fenced by the cancelled set.
    block.set()
    assert sink.wait(0.5) is False
