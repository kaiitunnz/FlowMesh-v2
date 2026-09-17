"""Decoding a carried ``traceparent`` back into an OTel context.

The decode counterpart of :func:`shared.telemetry.control.format_traceparent`: every
hop that carries a W3C ``traceparent`` string (``WorkerTaskMessage.traceparent``, a
mediated-op payload's ``traceparent`` key, ``RelayFrame.tp``) hands the string here to
get back a context a worker-side span can enter as its parent.
"""

from opentelemetry.context import Context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

__all__ = ["extract_context"]

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
