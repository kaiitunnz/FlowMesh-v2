"""The span an origin opens around one resident attempt's carriage."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry.trace import Span, Tracer

from shared.schemas.network import Transport
from shared.telemetry.propagation import extract_context
from shared.telemetry.provider import payload_free_span
from shared.telemetry.semconv import PHYSICAL_TRANSPORT, transport_span_name


@contextmanager
def transport_span(
    tracer: Tracer | None,
    selected_transport: str,
    traceparent: str | None,
    attributes: dict[str, Any],
) -> Iterator[Span | None]:
    """Open ``flowmesh.transport.<transport>`` for the transport control selected.

    The span parents on ``traceparent``, the stamp control put on the attempt, since an
    origin's drive runs outside the context that admitted it. Yield ``None`` without a
    tracer.
    """
    if tracer is None:
        yield None
        return
    with payload_free_span(
        tracer,
        transport_span_name(Transport(selected_transport)),
        context=extract_context(traceparent),
        attributes=attributes,
    ) as span:
        yield span


def record_realized_transport(span: Span | None, transport: str) -> None:
    """Record the transport an attempt's frames rode, after any fallback."""
    if span is not None:
        span.set_attribute(PHYSICAL_TRANSPORT, transport)


__all__ = ["record_realized_transport", "transport_span"]
