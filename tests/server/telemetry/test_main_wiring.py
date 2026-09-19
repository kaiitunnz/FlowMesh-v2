"""The server entrypoint hands every telemetry producer its instrument.

Each producer here is built and unit-tested by its own module against an object the
test constructs directly, so all of them stay green whether or not the entrypoint ever
builds them. Only importing the entrypoint shows whether the substrate is live, which
is why this imports the real module rather than mirroring its wiring.
"""

import atexit
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@contextmanager
def _imported_server_main() -> Iterator[Any]:
    """Import the entrypoint fresh, then undo what importing it registers.

    The module registers an ``atexit`` metrics export on import, so a test that
    imports it repeatedly would leave a handler per import to fire against pytest's
    already-closed streams at interpreter shutdown.

    Its Redis client connects while the module body runs and exits the process when it
    cannot, so the import is stubbed: every producer asserted here is handed its
    instrument by the same module body regardless, and a suite that reached Redis would
    pass or fail on whether the machine running it happens to have one.
    """
    sys.modules.pop("server.main", None)
    with patch("server.clients.RedisClient", return_value=MagicMock()):
        module = importlib.import_module("server.main")
    try:
        yield module
    finally:
        atexit.unregister(module._export_metrics_on_exit)
        sys.modules.pop("server.main", None)


@pytest.fixture
def server_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setenv("SERVER_METRICS_TELEMETRY_LEVEL", "fine")
    monkeypatch.setenv("SERVER_METRICS_CLICKHOUSE_URL", "http://localhost:8123")
    monkeypatch.setenv("RESULTS_DIR", (tmp_path / "results").as_posix())
    monkeypatch.setenv("SERVER_METRICS_DIR", (tmp_path / "metrics").as_posix())
    with _imported_server_main() as module:
        yield module


def test_the_runtime_receives_the_tracer_its_engines_need(server_main: Any) -> None:
    # The runtime builds a span emitter per engine from these two, on both the submit
    # and the rehydrate path; without them every engine gets the null emitter.
    assert server_main.RUNTIME._tracer is not None
    assert server_main.RUNTIME._telemetry is not None


def test_the_workflow_root_span_has_an_emitter(server_main: Any) -> None:
    from server.orchestration.telemetry import WorkflowSpanEmitter

    assert isinstance(
        server_main.EVENT_MONITOR.finalizer._workflow_span_emitter,
        WorkflowSpanEmitter,
    )


def test_the_runtime_notifies_the_completion_finalizer(server_main: Any) -> None:
    # A terminal the control plane settles publishes no task event, so without this
    # wiring such a workflow never emits its span and never closes its log stream.
    assert (
        server_main.RUNTIME._on_workflow_settled
        == server_main.EVENT_MONITOR.finalizer.request
    )


def test_the_fleet_sampler_is_built_and_enabled(server_main: Any) -> None:
    assert server_main.FLEET_SAMPLER is not None
    assert server_main.FLEET_SAMPLER._enabled


def test_the_telemetry_store_reaches_app_state(server_main: Any) -> None:
    from server.telemetry import ClickHouseTelemetryStore

    assert isinstance(server_main.app.state.telemetry_store, ClickHouseTelemetryStore)


def test_an_unconfigured_store_still_sets_the_attribute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The routes read the attribute unconditionally, so it exists even when unset."""
    monkeypatch.delenv("SERVER_METRICS_CLICKHOUSE_URL", raising=False)
    monkeypatch.setenv("RESULTS_DIR", (tmp_path / "results").as_posix())
    monkeypatch.setenv("SERVER_METRICS_DIR", (tmp_path / "metrics").as_posix())
    with _imported_server_main() as module:
        assert module.app.state.telemetry_store is None


def test_off_builds_no_provider_and_no_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the default level the entrypoint builds no instrument to pay for."""
    from opentelemetry.sdk.metrics import Meter as SdkMeter
    from opentelemetry.sdk.trace import Tracer as SdkTracer

    monkeypatch.setenv("SERVER_METRICS_TELEMETRY_LEVEL", "off")
    monkeypatch.delenv("SERVER_METRICS_CLICKHOUSE_URL", raising=False)
    monkeypatch.setenv("RESULTS_DIR", (tmp_path / "results").as_posix())
    monkeypatch.setenv("SERVER_METRICS_DIR", (tmp_path / "metrics").as_posix())

    with _imported_server_main() as module:
        # Neither instrument is backed by an SDK provider, so nothing was built
        # and no exporter thread is running.
        assert not isinstance(module.SERVER_TRACER, SdkTracer)
        assert not isinstance(module.SERVER_METER, SdkMeter)
        assert not module.CONTROL_TRACER.enabled
        assert not module.FLEET_SAMPLER._enabled
        assert module.app.state.telemetry_store is None
