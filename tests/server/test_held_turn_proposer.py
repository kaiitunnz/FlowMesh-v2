"""A held model turn is authorized only for the dispatch holding the agent, on the
worker whose authenticated stream relayed the proposal."""

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from server.task.runtime import TaskRuntime
from shared.grpc.supervisor.v1 import supervisor_pb2
from shared.schemas.event import parse_event
from shared.tools.contract import AgentModelTurnProposal
from tests.server.servicer_helpers import ServicerHarness, WorkerContext
from tests.server.task.test_task_merge import _monitor
from tests.server.task.test_worker_originated_boundary import (
    _MODEL_WF,
    HELD_DISPATCH,
    _deny_frames,
    _hold_dispatch,
    _permit_frames,
    _register,
    _runtime,
)


async def _proposal_stream(
    writer: str, claimed_worker: str, dispatch_id: str
) -> AsyncIterator[supervisor_pb2.EventMessage]:
    proposal = AgentModelTurnProposal(
        agent_task_id=writer,
        call_correlation="t0",
        request_digest="d",
        dispatch_id=dispatch_id,
    )
    message = supervisor_pb2.EventMessage()
    message.payload.update(
        {
            "type": "MEDIATED_OP_PROPOSE",
            "worker_id": claimed_worker,
            "payload": {"proposal": proposal.model_dump(mode="json")},
        }
    )
    yield message


async def _relay_proposal(
    runtime: TaskRuntime, writer: str, claimed_worker: str, dispatch_id: str
) -> str:
    """Relay one proposal through a supervisor whose stream authenticates as its own
    worker, and hand what it relays to the root; returns the authenticated id."""
    harness = ServicerHarness()
    stream_worker = await harness.register()
    await harness.servicer.PushEvents(
        _proposal_stream(writer, claimed_worker, dispatch_id),
        cast(Any, WorkerContext()),
    )
    monitor = _monitor(runtime)
    for relayed in harness.relay.events:
        monitor._handle_worker_event(cast(Any, parse_event(relayed)))
    return stream_worker


@pytest.mark.asyncio
async def test_a_proposal_naming_the_holder_from_another_stream_is_denied() -> None:
    runtime = _runtime()
    _, ids = await _register(runtime, _MODEL_WF)
    writer = ids["writer"]
    _hold_dispatch(runtime, writer, worker="wkr-holder")

    stream_worker = await _relay_proposal(runtime, writer, "wkr-holder", HELD_DISPATCH)

    assert not _permit_frames(runtime)
    frames = cast(Any, runtime._worker_registry).frames
    assert [(worker, kind) for worker, kind, _ in frames] == [(stream_worker, "deny")]


@pytest.mark.asyncio
async def test_a_proposal_from_a_superseded_dispatch_is_denied() -> None:
    runtime = _runtime()
    _, ids = await _register(runtime, _MODEL_WF)
    writer = ids["writer"]
    _hold_dispatch(runtime, writer, worker="wkr-1")

    await _relay_proposal(runtime, writer, "wkr-1", "dsp-superseded")

    assert not _permit_frames(runtime)
    assert [d["reason"] for d in _deny_frames(runtime)] == ["model turn not held"]


@pytest.mark.asyncio
async def test_the_holding_dispatch_on_its_own_stream_gets_a_permit() -> None:
    runtime = _runtime()
    _, ids = await _register(runtime, _MODEL_WF)
    writer = ids["writer"]
    _hold_dispatch(runtime, writer, worker="wkr-1")

    stream_worker = await _relay_proposal(runtime, writer, "wkr-1", HELD_DISPATCH)

    assert stream_worker == "wkr-1"
    assert len(_permit_frames(runtime)) == 1 and not _deny_frames(runtime)
