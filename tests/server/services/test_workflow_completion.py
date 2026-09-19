"""A workflow closes however its last task settled.

Most terminals reach the server as a published task event. Some never do: a task no
worker can satisfy, an exhausted retry and a model boundary the gateway fails are all
settled by the control plane itself, which records metrics but publishes nothing. Such
a workflow used to keep its log stream open forever and emit no workflow span. The
runtime now notifies the completion finalizer from its own terminal-persist step, so
both kinds of terminal close through the one path.
"""

import asyncio
import logging
import threading
import time
from typing import Any, cast

from server.services.completion import WorkflowCompletionFinalizer
from server.task.models import TaskStatus
from shared.harness import BoundaryEventKind
from shared.utils.time import ts_to_iso
from tests.server.services.test_workflow_span_close import (
    _RecordingWorkflowSpanEmitter,
    _RedisMirroringTaskState,
)
from tests.server.task.test_episode_cancel_safety import _run_step
from tests.server.task.test_v2_orchestration import (
    FakeRegistry,
    _register,
    _runtime,
    _worker,
)
from worker.executors.harness.scripted import ScriptedHarnessAdapter, ScriptedStep

_TS = "2026-09-16T00:00:00Z"

_AGENT = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: completion-agent}
spec:
  graph:
    nodes:
      - name: writer
        spec:
          taskType: agent
          v2:
            authority: {invoke: [model], delegate: []}
            tools: [{name: model}]
          harness: {backend: scripted, version: v1, params: {script: []}}
"""

_CHAIN = """
apiVersion: flowmesh/v2
kind: Workflow
metadata: {name: completion}
spec:
  graph:
    nodes:
      - name: head
        spec: {taskType: echo, data: {type: list, items: [x]}}
      - name: tail
        dependsOn: [head]
        spec: {taskType: echo, data: {type: list, items: [y]}}
"""


def _finalizer(runtime: Any, redis: Any, emitter: Any) -> WorkflowCompletionFinalizer:
    finalizer = WorkflowCompletionFinalizer(
        redis_client=cast(Any, redis),
        runtime=runtime,
        logger=logging.getLogger("test.completion"),
        workflow_span_emitter=cast(Any, emitter),
    )
    runtime.set_completion_notifier(finalizer.request)
    return finalizer


def _wired(
    runtime: Any, workflow_id: str, task_ids: list[str]
) -> tuple[WorkflowCompletionFinalizer, Any, _RecordingWorkflowSpanEmitter]:
    redis = _RedisMirroringTaskState(runtime, task_ids)
    redis.keys[f"workflow:{workflow_id}"] = "1"
    emitter = _RecordingWorkflowSpanEmitter()
    return _finalizer(runtime, redis, emitter), redis, emitter


def test_a_failure_the_control_plane_settles_closes_the_workflow() -> None:
    """The dispatcher's own failure path publishes no event to close on.

    `Dispatcher.fail_task` settles the task in the runtime and records its metrics,
    but never publishes to the task event stream -- so before the runtime notified,
    nothing ever ran the completion check for a workflow that ended this way.
    """

    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        registry.submitted_at = _TS
        workflow_id, ids = await _register(runtime, _CHAIN)
        head, tail = ids["head"], ids["tail"]
        finalizer, redis, emitter = _wired(runtime, workflow_id, [head, tail])

        runtime.mark_failed(
            head,
            None,
            {"reason": "no_eligible_worker"},
            _TS,
            error="No worker satisfies the task requirements",
        )
        finalizer.drain()

        assert redis.set_members(f"workflow:{workflow_id}:tasks") == set()
        assert f"workflow:{workflow_id}:logs:closed" in redis.keys
        assert emitter.emitted == [workflow_id]

    asyncio.run(run())


def test_a_failed_model_boundary_closes_the_workflow() -> None:
    """A gateway failure settles the agent through the engine's advance, not a mark.

    It reaches the terminal persist by a different route than the dispatcher does, and
    publishes no event either; covering the persist step covers both.
    """

    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        registry.submitted_at = _TS
        workflow_id, ids = await _register(runtime, _AGENT)
        writer = ids["writer"]
        finalizer, redis, emitter = _wired(runtime, workflow_id, [writer])

        adapter = ScriptedHarnessAdapter(
            [
                ScriptedStep(
                    op="boundary",
                    kind=BoundaryEventKind.INVOCATION,
                    call="m0",
                    interface="model",
                    payload="draft",
                ),
                ScriptedStep(op="complete", value_from="m0"),
            ],
            "v1",
        )
        result = _run_step(runtime, adapter, writer)
        runtime.mark_succeeded(
            writer, "wkr-1", {"agent_episode": result.model_dump(mode="json")}, _TS
        )
        finalizer.drain()
        assert f"workflow:{workflow_id}:logs:closed" not in redis.keys

        assert runtime.settle_episode_invocation(writer, "m0", error="upstream refused")
        finalizer.drain()

        writer_record = runtime.get_record(writer)
        assert writer_record is not None and writer_record.status == TaskStatus.FAILED
        assert f"workflow:{workflow_id}:logs:closed" in redis.keys
        assert emitter.emitted == [workflow_id]

    asyncio.run(run())


def test_a_restart_closes_a_workflow_that_finished_during_the_outage() -> None:
    """A crash between the durable settle and the close would strand the stream.

    Rehydration notifies for every workflow it restores, and the finalizer's own
    guards drop the ones still running or already closed.
    """

    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        registry.submitted_at = _TS
        workflow_id, ids = await _register(runtime, _CHAIN)
        head, tail = ids["head"], ids["tail"]
        runtime.mark_failed(head, None, {}, _TS, error="boom")

        restarted = _runtime(registry)
        finalizer, redis, emitter = _wired(restarted, workflow_id, [head, tail])
        await restarted.rehydrate()
        finalizer.drain()

        assert f"workflow:{workflow_id}:logs:closed" in redis.keys
        assert emitter.emitted == [workflow_id]
        # The span is re-emitted from the durable finishes, so a restart reproduces it
        # rather than giving the workflow a second, differently-dated span.
        finishes = [
            record.finished_ts
            for task_id in (head, tail)
            if (record := restarted.get_record(task_id)) is not None
            and record.finished_ts is not None
        ]
        assert emitter.extents[0][1] == ts_to_iso(max(finishes))

    asyncio.run(run())


def test_a_workflow_with_work_left_is_not_closed() -> None:
    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        registry.submitted_at = _TS
        workflow_id, ids = await _register(runtime, _CHAIN)
        head, tail = ids["head"], ids["tail"]
        finalizer, redis, emitter = _wired(runtime, workflow_id, [head, tail])

        runtime.mark_dispatched(head, cast(Any, _worker()))
        runtime.mark_succeeded(head, "wkr-1", {}, _TS)
        finalizer.drain()

        tail_record = runtime.get_record(tail)
        assert tail_record is not None and tail_record.status != TaskStatus.DONE
        assert f"workflow:{workflow_id}:logs:closed" not in redis.keys
        assert emitter.emitted == []

    asyncio.run(run())


def test_the_log_stream_closes_even_when_the_span_cannot_be_emitted() -> None:
    """Closing the stream is a lifecycle action; emitting the span is observation."""

    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        workflow_id, ids = await _register(runtime, _CHAIN)
        head, tail = ids["head"], ids["tail"]

        class _Exploding:
            def emit(self, *_args: str) -> None:
                raise RuntimeError("exporter down")

        redis = _RedisMirroringTaskState(runtime, [head, tail])
        redis.keys[f"workflow:{workflow_id}"] = "1"
        finalizer = _finalizer(runtime, redis, _Exploding())

        runtime.mark_failed(head, None, {}, _TS, error="boom")
        finalizer.drain()

        assert f"workflow:{workflow_id}:logs:closed" in redis.keys

    asyncio.run(run())


def test_requesting_a_close_never_waits_on_the_runtime() -> None:
    """The settle path notifies under the scheduler lock, so the queue must be free.

    If the finalizer held its request queue while reading the runtime, a settle that
    notifies would block behind a read that is itself waiting for the settle's lock.
    """

    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        workflow_id, ids = await _register(runtime, _CHAIN)
        head, tail = ids["head"], ids["tail"]
        finalizer, _redis, _emitter = _wired(runtime, workflow_id, [head, tail])
        runtime.mark_failed(head, None, {}, _TS, error="boom")

        reading = threading.Event()
        release = threading.Event()
        original = runtime.workflow_settlement

        def _blocking(wid: str) -> Any:
            reading.set()
            release.wait(5.0)
            return original(wid)

        runtime.workflow_settlement = _blocking  # type: ignore[assignment]
        finalizer.request(workflow_id)
        drainer = threading.Thread(target=finalizer.drain, daemon=True)
        drainer.start()
        assert reading.wait(5.0)

        requested = threading.Event()

        def _request() -> None:
            finalizer.request(workflow_id)
            requested.set()

        threading.Thread(target=_request, daemon=True).start()
        assert requested.wait(2.0), "request() waited on the in-flight close"
        release.set()
        drainer.join(5.0)

    asyncio.run(run())


def test_the_vault_still_purges_when_the_last_task_settles() -> None:
    """Completion-close and credential reclaim share one settled check.

    The shared check must stay the reclaim's own condition: the finalizer's extra
    durable guards live in the finalizer, or a workflow's credentials would start
    being purged at a different moment than before.
    """

    async def run() -> None:
        purged: list[str] = []

        class _RecordingVault:
            async def store(self, workflow_id: str, ref: str, secret: Any) -> None:
                return None

            def resolve(self, workflow_id: str, ref: str | None) -> None:
                return None

            def purge(self, workflow_id: str) -> None:
                purged.append(workflow_id)

        registry = FakeRegistry()
        runtime = _runtime(registry)
        runtime._secret_vault = cast(Any, _RecordingVault())
        workflow_id, ids = await _register(runtime, _CHAIN)
        head, tail = ids["head"], ids["tail"]

        runtime.mark_dispatched(head, cast(Any, _worker()))
        runtime.mark_succeeded(head, "wkr-1", {}, _TS)
        assert purged == []

        runtime.mark_dispatched(tail, cast(Any, _worker()))
        runtime.mark_succeeded(tail, "wkr-1", {}, _TS)
        assert purged == [workflow_id]

    asyncio.run(run())


def test_the_finalizer_thread_closes_without_being_polled() -> None:
    """The drain runs on its own thread, so a settle with no further events closes."""

    async def run() -> None:
        registry = FakeRegistry()
        runtime = _runtime(registry)
        registry.submitted_at = _TS
        workflow_id, ids = await _register(runtime, _CHAIN)
        head, tail = ids["head"], ids["tail"]
        finalizer, redis, _emitter = _wired(runtime, workflow_id, [head, tail])

        stop = threading.Event()
        thread = threading.Thread(target=finalizer.run, args=(stop, 0.05), daemon=True)
        thread.start()
        try:
            runtime.mark_failed(head, None, {}, _TS, error="boom")
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if f"workflow:{workflow_id}:logs:closed" in redis.keys:
                    break
                time.sleep(0.02)
        finally:
            stop.set()
            thread.join(5.0)

        assert f"workflow:{workflow_id}:logs:closed" in redis.keys

    asyncio.run(run())
