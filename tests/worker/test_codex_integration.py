"""A live Codex app-server proves the bounded held-facade recovery outside simulation.

A real ``codex app-server`` runs against a local Responses-compatible backend; the
model defers one mediated facade call, the fabric settles it, and the outcome injects
back over ``thread/inject_items``. The boundary is detected as a JSON envelope the model
emits on its turn output — a text convention, not a native tool-call wire — so the
facade origination is the one stubbed edge. What is real and load-bearing is the
recovery: across a ``kill -9`` the same rollout resumes by thread id under a stable
``CODEX_HOME``, and the adapter's committed-key dedup keeps the settled outcome injected
exactly once into the live rollout — observed by counting the injections the backend
sees, so a re-injection would fail the assertion. Exactly-once of the mediated *effect*
under a lost capsule is the fabric idempotency-key property, proven at the engine level;
it is not claimed here.

This is a live process, not a fake transport, and not a CPU Docker end-to-end test,
whose environment lacks the Codex binary. The proof is bounded to single-forward,
single-facade recovery.
"""

import json
import os
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "openai_codex", reason="needs the openai-codex worker harness dependency"
)

from shared.harness import (  # noqa: E402
    DeliveredOutcome,
    HarnessResult,
    HarnessResultKind,
    OutcomeKind,
)
from worker.executors.harness.codex import CodexAppServerHarnessAdapter  # noqa: E402
from worker.executors.harness.codex_transport import (  # noqa: E402
    CodexTransportConfig,
    CodexTransportError,
    RealCodexAppServerTransport,
)

_FACADE_ENVELOPE = json.dumps({"facade": {"tool": "spawn_agent", "region": "reviewer"}})
_FINAL_TEXT = "final"


def _codex_available() -> bool:
    # _resolve_codex_bin is private to the SDK; the exact version pin keeps it stable.
    from openai_codex.client import CodexConfig, _resolve_codex_bin

    try:
        return _resolve_codex_bin(CodexConfig()).exists()
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.codex_integration,
    pytest.mark.skipif(not _codex_available(), reason="codex binary unavailable"),
]


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    ).encode()


def _message_output(text: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "message",
            "id": "msg_0",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }
    ]


class _ResponsesStub:
    """A local Responses backend that defers a facade and counts injected outcomes.

    ``max_injected`` is the most function_call_output items keyed to a mediated
    correlation seen in a request's history — one if the outcome injected exactly once,
    more if a recovery re-injected it. ``stall_release``, when set, holds each turn open
    until the event fires, standing in for a hung app-server.
    """

    def __init__(self, stall_release: threading.Event | None = None) -> None:
        self.max_injected = 0
        self._stall = stall_release
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        stub = self

        def _injected(body: dict[str, Any]) -> int:
            return sum(
                1
                for item in body.get("input", [])
                if item.get("type") == "function_call_output"
                and str(item.get("call_id", "")).startswith("fab-")
            )

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n).decode()) if n else {}
                injected = _injected(body)
                with stub._lock:
                    stub.max_injected = max(stub.max_injected, injected)
                if stub._stall is not None:
                    stub._stall.wait(30)
                text = _FINAL_TEXT if injected else _FACADE_ENVELOPE
                output = _message_output(text)
                events: list[dict[str, Any]] = [
                    {"type": "response.created", "response": {"id": "r"}},
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": output[0],
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "r",
                            "status": "completed",
                            "output": output,
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    },
                ]
                data = _sse(events)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # the app-server may already be gone (kill or timeout)

        return Handler

    @property
    def base_url(self) -> str:
        port = int(self._server.server_address[1])
        return f"http://127.0.0.1:{port}/v1"

    def __enter__(self) -> "_ResponsesStub":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


class _Fabric:
    """The fabric's role: settle a boundary once into a stable keyed outcome."""

    def __init__(self) -> None:
        self._settled: dict[str, DeliveredOutcome] = {}

    def settle(
        self, result: HarnessResult, value: str = _FINAL_TEXT
    ) -> DeliveredOutcome:
        assert result.request is not None
        corr = result.request.call_correlation
        assert corr is not None
        if corr not in self._settled:
            self._settled[corr] = DeliveredOutcome(
                call_correlation=corr,
                idempotency_key=f"idm-{corr}",
                kind=OutcomeKind.RESULT,
                value=value,
            )
        return self._settled[corr]


TransportFactory = Callable[..., RealCodexAppServerTransport]


@pytest.fixture
def transports() -> Iterator[TransportFactory]:
    made: list[RealCodexAppServerTransport] = []

    def _make(base_url: str, home: Path, **kwargs: Any) -> RealCodexAppServerTransport:
        transport = RealCodexAppServerTransport(
            CodexTransportConfig(
                base_url=base_url, model="gpt-5-codex", codex_home=home, **kwargs
            )
        )
        made.append(transport)
        return transport

    yield _make
    for transport in made:
        transport.close()


def test_kill9_before_injection_injects_the_outcome_once(
    tmp_path: Path, transports: TransportFactory
) -> None:
    home = tmp_path / "codex_home"
    fabric = _Fabric()
    with _ResponsesStub() as stub:
        issue = transports(stub.base_url, home)
        first = CodexAppServerHarnessAdapter(issue, "v1").start(
            "a", capsule=None, outcomes=[]
        )
        assert first.kind is HarnessResultKind.BOUNDARY
        assert (
            first.request is not None and first.request.child_region_ref == "reviewer"
        )

        # The app-server dies after the boundary issues, before its outcome injects.
        os.kill(issue.pid, 9)

        recover = transports(stub.base_url, home)
        done = CodexAppServerHarnessAdapter(recover, "v1").start(
            "a", capsule=first.capsule, outcomes=[fabric.settle(first)]
        )

    assert done.kind is HarnessResultKind.COMPLETION and done.value == _FINAL_TEXT
    assert stub.max_injected == 1


def test_kill9_after_injection_does_not_reinject(
    tmp_path: Path, transports: TransportFactory
) -> None:
    home = tmp_path / "codex_home"
    fabric = _Fabric()
    with _ResponsesStub() as stub:
        run = transports(stub.base_url, home)
        adapter = CodexAppServerHarnessAdapter(run, "v1")
        first = adapter.start("a", capsule=None, outcomes=[])
        assert first.kind is HarnessResultKind.BOUNDARY
        outcome = fabric.settle(first)

        completed = adapter.start("a", capsule=first.capsule, outcomes=[outcome])
        assert completed.kind is HarnessResultKind.COMPLETION
        os.kill(run.pid, 9)

        # Recover from the durable capsule that already committed the key; the adapter's
        # dedup must keep it from injecting the outcome into the rollout a second time.
        recover = transports(stub.base_url, home)
        again = CodexAppServerHarnessAdapter(recover, "v1").start(
            "a", capsule=completed.capsule, outcomes=[outcome]
        )

    assert again.kind is HarnessResultKind.COMPLETION
    assert stub.max_injected == 1


def test_stalled_turn_raises_a_transport_error(
    tmp_path: Path, transports: TransportFactory
) -> None:
    home = tmp_path / "codex_home"
    release = threading.Event()
    with _ResponsesStub(stall_release=release) as stub:
        transport = transports(stub.base_url, home, turn_timeout_sec=2.0)
        adapter = CodexAppServerHarnessAdapter(transport, "v1")
        try:
            with pytest.raises(CodexTransportError):
                adapter.start("a", capsule=None, outcomes=[])
        finally:
            release.set()
