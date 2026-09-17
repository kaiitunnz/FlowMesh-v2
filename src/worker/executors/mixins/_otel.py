"""OpenTelemetry tracing wiring for worker executors.

Sets up a single process-wide ``TracerProvider`` with a JSONL exporter that
appends ``ReadableSpan.to_json()`` to ``<out_dir>/logs/spans.jsonl`` for
whichever task is currently executing, plus an optional OTLP exporter beside
it. The JSONL destination is held in a ``ContextVar`` updated by
``task_trace_context`` on enter / exit, so the two off-lane subsystems that
run concurrently with the next task (the egress sidecar's thread pool, the
resident lane host's own loop thread) resolve no path rather than appending
to an unrelated task's file.

The ``trace_id`` is pinned to the workflow id via a custom ``IdGenerator``
that reads the active workflow id from a ``ContextVar``. ``task_trace_context``
sets it for the executors' own shipped span; ``workflow_trace_context`` sets
it alone, for the runner's ``flowmesh.task`` span, so every task type derives
a workflow-consistent trace id even when no executor-level context runs.
Sub-spans inherit the trace id from the OTel parent context automatically.
"""

import re
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator

from shared.schemas.governance import SpanType
from shared.telemetry.config import TelemetryConfig, TelemetryLevel
from shared.utils.ids import PREFIX_WORKFLOW

_HEX_ONLY = re.compile(r"[^0-9a-f]")
_TRACER_NAME = "flowmesh.worker"
_SERVICE_NAME = "flowmesh-worker"
_DEFAULT_OTLP_TIMEOUT_SEC = 10.0

_workflow_id_var: ContextVar[str | None] = ContextVar(
    "flowmesh_workflow_id", default=None
)
_current_spans_path_var: ContextVar[Path | None] = ContextVar(
    "flowmesh_spans_path", default=None
)
_lock = threading.Lock()

_telemetry_config = TelemetryConfig(
    level=TelemetryLevel.OFF,
    traces_enabled=True,
    metrics_enabled=True,
    sample_ratio=1.0,
    otlp_endpoint=None,
)
_otlp_timeout_sec = _DEFAULT_OTLP_TIMEOUT_SEC


def workflow_to_trace_id_int(workflow_id: str) -> int:
    """Stable 128-bit trace id derived from the workflow id.

    Strips the ``wfl-`` prefix before hex extraction so the prefix's ``f``
    doesn't shift the bit pattern.
    """
    body = workflow_id.lower().removeprefix(f"{PREFIX_WORKFLOW}-")
    hex_only = _HEX_ONLY.sub("", body)
    if not hex_only:
        return 0
    return int(hex_only.zfill(32)[:32], 16)


class _FlowMeshIdGenerator(IdGenerator):
    """Pin trace_id to the active workflow id; random span_ids."""

    def __init__(self) -> None:
        self._fallback = RandomIdGenerator()

    def generate_span_id(self) -> int:
        return self._fallback.generate_span_id()

    def generate_trace_id(self) -> int:
        workflow_id = _workflow_id_var.get()
        if workflow_id:
            value = workflow_to_trace_id_int(workflow_id)
            if value != 0:
                return value
        return self._fallback.generate_trace_id()


class _JSONLSpanExporter(SpanExporter):
    """Append each completed span to the active task's spans.jsonl file."""

    def __init__(self, path_provider: Callable[[], Path | None]) -> None:
        self._path_provider = path_provider

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        path = self._path_provider()
        if path is None:
            return SpanExportResult.SUCCESS
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for span in spans:
                fh.write(span.to_json(indent=None) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _resolve_path() -> Path | None:
    return _current_spans_path_var.get()


_PROVIDER_INITIALIZED = False


def configure(
    config: TelemetryConfig,
    *,
    otlp_timeout_sec: float = _DEFAULT_OTLP_TIMEOUT_SEC,
) -> None:
    """Set the telemetry config the worker's tracer provider is built from.

    Must be called before the first task runs (before anything reaches
    ``get_tracer()``) — the provider builds once per process and a call after
    that point has no effect on the OTLP attachment. Gates only the new
    ``flowmesh.*`` spans and the OTLP exporter; the shipped JSONL sink is
    unconditional regardless of what is configured here.
    """
    global _telemetry_config, _otlp_timeout_sec
    _telemetry_config = config
    _otlp_timeout_sec = otlp_timeout_sec


def emits(minimum: TelemetryLevel) -> bool:
    """Whether the configured level is at least ``minimum`` and traces are on.

    The single boolean check every new ``flowmesh.*`` span call site gates on
    before opening a span — ``off`` (the default) emits none of them.
    """
    return _telemetry_config.traces_enabled and _telemetry_config.emits(minimum)


def _ensure_tracer_provider() -> None:
    global _PROVIDER_INITIALIZED
    if _PROVIDER_INITIALIZED:
        return
    with _lock:
        if _PROVIDER_INITIALIZED:
            return
        provider = TracerProvider(
            resource=Resource.create({"service.name": _SERVICE_NAME}),
            id_generator=_FlowMeshIdGenerator(),
        )
        provider.add_span_processor(
            SimpleSpanProcessor(_JSONLSpanExporter(_resolve_path))
        )
        if emits(TelemetryLevel.COARSE) and _telemetry_config.otlp_endpoint:
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=_telemetry_config.otlp_endpoint,
                        timeout=_otlp_timeout_sec,
                    )
                )
            )
        trace.set_tracer_provider(provider)
        _PROVIDER_INITIALIZED = True


def get_tracer():
    _ensure_tracer_provider()
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def workflow_trace_context(workflow_id: str) -> Iterator[None]:
    """Bind trace_id derivation to ``workflow_id`` for the duration of a task.

    The runner's half of the split described on ``task_trace_context``: it
    covers every task type, including the ones whose executor never opens a
    trace context of its own, so a ``flowmesh.task`` span opened with no
    inbound ``traceparent`` still roots into its workflow's trace rather than
    a random one.
    """
    token = _workflow_id_var.set(workflow_id)
    try:
        yield
    finally:
        _workflow_id_var.reset(token)


@contextmanager
def task_trace_context(workflow_id: str, spans_path: Path) -> Iterator[None]:
    """Bind trace_id derivation and the JSONL exporter's destination for a task.

    The executors' half of the split: ``workflow_id`` here is redundant with
    ``workflow_trace_context`` (the runner already set the same value) for the
    six executors that call this, but ``spans_path`` is not — only they upload
    ``spans.jsonl``, so only they route the exporter to it. Restores the
    previous state of both on exit.
    """
    workflow_token = _workflow_id_var.set(workflow_id)
    spans_token = _current_spans_path_var.set(spans_path)
    try:
        yield
    finally:
        _current_spans_path_var.reset(spans_token)
        _workflow_id_var.reset(workflow_token)


def new_span_attributes(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Attributes for a new ``flowmesh.*`` span — never ``data_id``.

    ``_group_spans`` drops any span lacking ``data_id`` before grouping, which
    is what keeps every new span invisible to the governance analyzer by
    construction. Setting it here would defeat that.
    """
    attrs: dict[str, Any] = {}
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            attrs[key] = value
    return attrs


def attributes_with_type(
    span_type: SpanType,
    *,
    data_id: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {"flowmesh.type": span_type.value}
    if data_id is not None:
        attrs["data_id"] = data_id
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            attrs[key] = value
    return attrs
