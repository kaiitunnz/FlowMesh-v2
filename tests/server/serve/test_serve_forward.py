"""The root-hosted forward serve ingress admits over the same gate as proxy.

A forward request enters at the root's per-task port; the listener resolves the serve
task from that port, freezes the request, and hands it to ``admit_forward_request``,
which authenticates the FlowMesh principal, checks task-read access, and admits it over
the shared root relay. A bad credential or a denied read is refused with the engine's
own status and raises no claim. These cover that admission gate; the port bind/commit
and end-to-end streaming are covered by the gated-serve edge tests and the e2e.
"""

import asyncio

import pytest
from fastapi import HTTPException

from server.serve import (
    ForwardIngressDirectory,
    GatedServe,
    ServeBindingStore,
    ServeTerminalStore,
)
from server.serve.forward_listener import ServeForwardDenied
from server.serve.ingress import ServeAccessMode, ServeIngressRegistry
from server.task.v2.representations.operators import ServiceInterface
from shared.resident.envelope import ServeRequestEnvelope, freeze_request_envelope


class _Principal:
    def __init__(self) -> None:
        self.principal_id = "usr-1"
        self.org_id = "org-1"


class _FakeControl:
    """A control double that records originations; runs loop work inline."""

    def __init__(self) -> None:
        self.originations: list = []
        self.redrives: list = []

    def call_on_loop(self, fn) -> None:
        fn()

    def originate_serve(self, origination) -> None:
        self.originations.append(origination)

    def redrive_serve(self, origination) -> None:
        self.redrives.append(origination)


class _FakeRelay:
    def open(self, *args, **kwargs) -> None: ...
    def authorize(self, *args, **kwargs) -> None: ...
    def close(self, *args, **kwargs) -> None: ...


class _FakeForwardListener:
    def __init__(self) -> None:
        self.bound: list[tuple[str, int, int]] = []
        self.released: list[int] = []

    def schedule_bind(self, task_id: str, exposure_generation: int, port: int) -> None:
        self.bound.append((task_id, exposure_generation, port))

    def schedule_release(self, port: int) -> None:
        self.released.append(port)


def _edge() -> GatedServe:
    edge = GatedServe(
        bindings=ServeBindingStore(),
        terminals=ServeTerminalStore(),
        control=_FakeControl(),  # type: ignore[arg-type]
        relay=_FakeRelay(),  # type: ignore[arg-type]
        ingresses=ServeIngressRegistry("serve-edge"),
        exposures=ForwardIngressDirectory("serve.example", 34000, 34009),
        forward_listener=_FakeForwardListener(),  # type: ignore[arg-type]
    )
    edge._bindings.adopt(
        "tsk-1",
        service_ref="org/model",
        interface=ServiceInterface.CHAT,
        isolation=None,
        adapter=None,
        adapter_source=None,
        engine_batch_key="org/model|chat",
        max_output_tokens=None,
        access_mode=ServeAccessMode.FORWARD,
    )
    # Reserve and commit a live exposure for the task, as adoption + a bound report do.
    exposure = edge.exposures.reserve(
        serve_task_id="tsk-1", binding_generation=0, requested_port=None
    )
    assert exposure is not None
    edge.exposures.mark_binding("tsk-1", exposure.exposure_generation)
    edge.exposures.commit(
        serve_task_id="tsk-1",
        exposure_generation=exposure.exposure_generation,
        listener_generation=1,
    )
    return edge


def _envelope(method: str = "POST") -> ServeRequestEnvelope:
    return freeze_request_envelope(
        method=method,
        upstream_path="v1/chat/completions",
        query="",
        headers=[("content-type", "application/json")],
        body=b"{}",
    )


def test_admit_forward_request_admits_a_live_forward_exposure() -> None:
    edge = _edge()
    result = asyncio.run(edge.admit_forward_request("Bearer k", "tsk-1", _envelope()))
    assert result is not None
    control = edge.control
    assert len(control.originations) == 1  # type: ignore[attr-defined]
    orig = control.originations[0]  # type: ignore[attr-defined]
    # With no identity plugin installed the credential resolves the default admin
    # principal — the documented unrestricted-admin default — and it admits.
    assert orig.subject.id == "admin" and orig.subject.tenant == "local"
    assert orig.family == "serve/tsk-1"


def test_admit_forward_denies_a_bad_credential_without_raising_a_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edge = _edge()

    async def _reject(_credential: str, _logger: object) -> _Principal:
        raise HTTPException(status_code=401, detail="bad key")

    monkeypatch.setattr("server.serve.service.authenticate_api_key", _reject)

    # Authentication is control's alone; a bad credential is refused with the engine's
    # own status and no ServiceClaim is ever raised, so no credit is spent.
    with pytest.raises(ServeForwardDenied) as caught:
        asyncio.run(edge.admit_forward_request("Bearer bad", "tsk-1", _envelope()))
    assert caught.value.status == 401
    assert edge.control.originations == []  # type: ignore[attr-defined]


def test_admit_forward_denies_a_forbidden_task_read_without_raising_a_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edge = _edge()

    async def _principal(_credential: str, _logger: object) -> _Principal:
        return _Principal()

    async def _forbid(*_args: object, **_kwargs: object) -> None:
        raise HTTPException(status_code=403, detail="denied")

    monkeypatch.setattr("server.serve.service.authenticate_api_key", _principal)
    monkeypatch.setattr("server.serve.service.require_permission", _forbid)

    with pytest.raises(ServeForwardDenied) as caught:
        asyncio.run(edge.admit_forward_request("Bearer k", "tsk-1", _envelope()))
    assert caught.value.status == 403
    assert edge.control.originations == []  # type: ignore[attr-defined]


def test_admit_forward_fails_closed_without_a_live_exposure() -> None:
    # A forward task whose exposure is not live is unavailable: refused before any
    # credit rather than served over the proxy.
    edge = _edge()
    edge.exposures.drain("tsk-1")
    with pytest.raises(ServeForwardDenied) as caught:
        asyncio.run(edge.admit_forward_request("Bearer k", "tsk-1", _envelope()))
    assert caught.value.status == 503
    assert edge.control.originations == []  # type: ignore[attr-defined]


def test_admit_forward_refuses_a_method_the_binding_forbids() -> None:
    edge = _edge()
    binding = edge._bindings.get("tsk-1")
    assert binding is not None
    edge._bindings._bindings["tsk-1"] = binding.model_copy(
        update={"allowed_methods": ("GET",)}
    )
    with pytest.raises(ServeForwardDenied) as caught:
        asyncio.run(
            edge.admit_forward_request("Bearer k", "tsk-1", _envelope(method="POST"))
        )
    assert caught.value.status == 405
    assert edge.control.originations == []  # type: ignore[attr-defined]


def test_admit_forward_refuses_an_unknown_task() -> None:
    edge = _edge()
    with pytest.raises(ServeForwardDenied) as caught:
        asyncio.run(edge.admit_forward_request("Bearer k", "tsk-missing", _envelope()))
    assert caught.value.status == 404
    assert edge.control.originations == []  # type: ignore[attr-defined]
