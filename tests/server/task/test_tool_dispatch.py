"""The runtime routes a mediated boundary to exactly one handler by (kind, interface).

Only ``(INVOCATION, "model")`` reaches the model settler and only
``(INVOCATION, "search/v1")`` the tool broker; an unrecognized interface reaches
neither — it never silently falls through to the model settler.
"""

from server.orchestration.tool_dispatch import (
    MODEL_INTERFACE,
    SEARCH_INTERFACE,
    ToolInvocationEnvelope,
)
from shared.harness import BoundaryEventKind
from tests.server.task.test_v2_orchestration import FakeRegistry, _runtime


def _env(interface: str) -> ToolInvocationEnvelope:
    return ToolInvocationEnvelope(
        kind=BoundaryEventKind.INVOCATION,
        interface=interface,
        invocation_id="inv-1",
        task_id="tsk-x",
        activation_id="act-1",
        call_correlation="c0",
    )


def test_dispatch_routes_by_exact_kind_and_interface() -> None:
    runtime = _runtime(FakeRegistry())
    model: list[ToolInvocationEnvelope] = []
    broker: list[ToolInvocationEnvelope] = []
    runtime.set_model_settler(model.append)
    runtime.set_tool_broker(broker.append)

    runtime._dispatch_boundary(_env(MODEL_INTERFACE))
    runtime._dispatch_boundary(_env(SEARCH_INTERFACE))
    # An unknown interface routes to neither handler (a typed unavailable settle instead
    # of a misroute to the model settler).
    runtime._dispatch_boundary(_env("mystery/v1"))

    assert [e.interface for e in model] == [MODEL_INTERFACE]
    assert [e.interface for e in broker] == [SEARCH_INTERFACE]


def test_a_search_interface_never_reaches_the_model_settler() -> None:
    runtime = _runtime(FakeRegistry())
    model: list[ToolInvocationEnvelope] = []
    runtime.set_model_settler(model.append)
    # No broker installed: a search boundary must still not fall through to the model.
    runtime._dispatch_boundary(_env(SEARCH_INTERFACE))
    assert model == []
