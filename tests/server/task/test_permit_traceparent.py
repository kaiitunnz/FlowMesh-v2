"""The two permit-mint traceparent stamps (contract §4.2) reach the worker-side
handler, and are absent — not null — from the wire when telemetry is off.

Both stamps are post-mint: the boundary span id derives from the minted permit's own
``invocation_id``, which does not exist until ``mint_operation_permit`` /
``authorize_model_turn`` return.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, cast

from server.config import OrchestrationConfig
from shared.telemetry.config import TelemetryLevel
from shared.telemetry.ids import SpanIdKind, derived_span_id, workflow_to_trace_id_int
from shared.tools.contract import AgentModelTurnProposal, MediatedOperationPermit
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _NoopSecretVault,
    _register,
)
from tests.server.task.test_worker_originated_boundary import (
    _MODEL_WF,
    _SEARCH_WF,
    _dispatch_agent,
    _hold_dispatch,
    _permit_frames,
    _WorkerStub,
)
from tests.server.telemetry_helpers import recording_control_tracer


def _runtime(control: Any = None) -> Any:
    from server.task.runtime import TaskRuntime

    return TaskRuntime(
        cast(Any, FakeRegistry()),
        cast(Any, _WorkerStub()),
        OrchestrationConfig(),
        Path(tempfile.gettempdir()),
        logging.getLogger("permit-traceparent-test"),
        secret_vault=cast(Any, _NoopSecretVault()),
        control=control,
    )


def test_worker_originated_tool_permit_carries_a_traceparent_to_the_worker() -> None:
    control, _exporter = recording_control_tracer(TelemetryLevel.COARSE)
    runtime = _runtime(control)

    async def run() -> None:
        _wfl, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _dispatch_agent(runtime, writer)

        permits = _permit_frames(runtime)
        assert len(permits) == 1
        raw = permits[0]
        assert "traceparent" in raw
        permit = MediatedOperationPermit.model_validate(raw)
        assert permit.traceparent is not None

        workflow_id = runtime._tasks[writer].workflow_id
        expected_trace_id = workflow_to_trace_id_int(workflow_id)
        expected_span_id = derived_span_id(SpanIdKind.INVOCATION, permit.invocation_id)
        trace_id_hex, span_id_hex = permit.traceparent.split("-")[1:3]
        assert int(trace_id_hex, 16) == expected_trace_id
        assert int(span_id_hex, 16) == expected_span_id

    asyncio.run(run())


def test_worker_originated_tool_permit_carries_no_traceparent_key_when_off() -> None:
    runtime = _runtime(control=None)

    async def run() -> None:
        _wfl, ids = await _register(runtime, _SEARCH_WF)
        writer = ids["writer"]
        _dispatch_agent(runtime, writer)

        permits = _permit_frames(runtime)
        assert len(permits) == 1
        assert "traceparent" not in permits[0]

    asyncio.run(run())


def test_held_model_turn_permit_carries_a_traceparent_to_the_worker() -> None:
    control, _exporter = recording_control_tracer(TelemetryLevel.COARSE)
    runtime = _runtime(control)

    async def run() -> None:
        _wfl, ids = await _register(runtime, _MODEL_WF)
        writer = ids["writer"]
        _hold_dispatch(runtime, writer)
        runtime.authorize_model_turn(
            AgentModelTurnProposal(
                agent_task_id=writer, call_correlation="t0", request_digest="deadbeef"
            )
        )

        permits = _permit_frames(runtime)
        assert len(permits) == 1
        raw = permits[0]
        assert "traceparent" in raw
        permit = MediatedOperationPermit.model_validate(raw)
        assert permit.traceparent is not None

        workflow_id = runtime._tasks[writer].workflow_id
        expected_trace_id = workflow_to_trace_id_int(workflow_id)
        expected_span_id = derived_span_id(SpanIdKind.INVOCATION, permit.invocation_id)
        trace_id_hex, span_id_hex = permit.traceparent.split("-")[1:3]
        assert int(trace_id_hex, 16) == expected_trace_id
        assert int(span_id_hex, 16) == expected_span_id

    asyncio.run(run())


def test_held_model_turn_permit_carries_no_traceparent_key_when_off() -> None:
    runtime = _runtime(control=None)

    async def run() -> None:
        _wfl, ids = await _register(runtime, _MODEL_WF)
        writer = ids["writer"]
        _hold_dispatch(runtime, writer)
        runtime.authorize_model_turn(
            AgentModelTurnProposal(
                agent_task_id=writer, call_correlation="t0", request_digest="d"
            )
        )

        permits = _permit_frames(runtime)
        assert len(permits) == 1
        assert "traceparent" not in permits[0]

    asyncio.run(run())
