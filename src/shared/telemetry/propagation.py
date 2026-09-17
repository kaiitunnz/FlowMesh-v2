"""Decoding a carried ``traceparent`` back into an OTel context, and injecting the
ambient context into an HTTP header dict.

The decode counterpart of :func:`shared.telemetry.control.format_traceparent`: every
hop that carries a W3C ``traceparent`` string (``WorkerTaskMessage.traceparent``, a
mediated-op payload's ``traceparent`` key, ``RelayFrame.tp``) hands the string here to
get back a context a worker-side span can enter as its parent. The inject side forwards
the caller's own ambient context onto an outbound HTTP request, which is the right
context for a worker->server call that runs inside a task's span.
"""

from opentelemetry.context import Context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

__all__ = ["extract_context", "inject_ambient_traceparent"]

_PROPAGATOR = TraceContextTextMapPropagator()


def extract_context(traceparent: str | None) -> Context | None:
    """The context a carried ``traceparent`` names, or ``None`` when absent.

    Returns ``None`` rather than the empty current context when ``traceparent`` is
    ``None`` or empty, so a caller can tell "no inbound context" apart from "the
    root context" and decide whether to enter it at all.
    """
    if not traceparent:
        return None
    return _PROPAGATOR.extract({"traceparent": traceparent})


def inject_ambient_traceparent(headers: dict[str, str]) -> dict[str, str]:
    """Inject the ambient trace context's ``traceparent`` into an HTTP header dict.

    With no active span (telemetry off, or a call outside any span) the propagator
    injects nothing, so the header dict is returned unchanged — zero bytes on the
    wire. Mutates and returns ``headers`` in place.
    """
    _PROPAGATOR.inject(headers)
    return headers
