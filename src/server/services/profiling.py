"""Stage-level timing seam for the v2 server-side control plane.

A submission running on the v2 track pays control-plane cost the v1 static-DAG
track does not: template compilation, orchestration-ledger drive and settle,
ledger snapshot serialization, resident admission, mediated-boundary permit
minting, and relay establishment. The seam times those stages and hands each
measurement to a :class:`ControlStageSink` — :class:`~.metrics.MetricsRecorder`
in the server — keyed by ``workflow_id`` and, for stages that run mid-episode,
``invocation_id``.

Each measurement carries the :class:`StageWindow` it fired in, and whether it
ran inside an enclosing timed stage. Both are properties of the control point
rather than of the stage: the ledger snapshot serializes during submission,
dispatch and outcome settlement alike, and is enclosed by the dispatch stage at
some of those sites and a sibling of it at others.

Timing is off unless a deployment enables it; :data:`NULL_PROFILER` is the
disabled form.
"""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from enum import StrEnum
from time import perf_counter
from typing import Protocol


class StageWindow(StrEnum):
    """Aggregate a stage measurement lands in.

    ``SUBMIT`` runs inside the submit request, before any task's queue window
    opens; ``QUEUE`` runs between a task's submission and its start;
    ``POST_START`` runs mid-episode, after the task has started.
    """

    SUBMIT = "submit"
    QUEUE = "queue"
    POST_START = "post_start"


class ControlPlaneStage(StrEnum):
    """A timed v2 control-plane stage."""

    COMPILE_LOWER = "compile_lower"
    COMPILE_ASSEMBLE = "compile_assemble"
    COMPILE_FINALIZE = "compile_finalize"
    COMPILE_EPISODES = "compile_episodes"
    COMPILE_VALIDATE = "compile_validate"
    ENGINE_BUILD = "engine_build"
    DS_INITIAL_ADVANCE = "ds_initial_advance"
    DS_DRIVE = "ds_drive"
    DISPATCH = "dispatch"
    ADMISSION = "admission"
    PERMIT = "permit"
    RELAY = "relay"
    LEDGER_SNAPSHOT = "ledger_snapshot"


class ControlStageSink(Protocol):
    """Receives one timed control-plane stage measurement."""

    def record_control_stage(
        self,
        stage: ControlPlaneStage,
        window: StageWindow,
        seconds: float,
        *,
        workflow_id: str | None,
        invocation_id: str | None = None,
        nested: bool = False,
    ) -> None: ...


class ControlPlaneProfiler:
    """Times control-plane stages into a sink."""

    enabled = True

    def __init__(self, sink: ControlStageSink) -> None:
        self._sink = sink

    @contextmanager
    def stage(
        self,
        stage: ControlPlaneStage,
        window: StageWindow,
        *,
        workflow_id: str | None,
        invocation_id: str | None = None,
        nested: bool = False,
    ) -> Iterator[None]:
        """Time the enclosed block and record it, whether or not it raises.

        ``nested`` marks a call site an enclosing timed stage already covers, so
        the window total counts it once.
        """
        started = perf_counter()
        try:
            yield
        finally:
            self._sink.record_control_stage(
                stage,
                window,
                perf_counter() - started,
                workflow_id=workflow_id,
                invocation_id=invocation_id,
                nested=nested,
            )


class _NullProfiler:
    """Disabled profiler; every stage is a reused no-op context."""

    enabled = False
    _context = nullcontext()

    def stage(
        self,
        stage: ControlPlaneStage,
        window: StageWindow,
        *,
        workflow_id: str | None,
        invocation_id: str | None = None,
        nested: bool = False,
    ) -> nullcontext[None]:
        return self._context


NULL_PROFILER = _NullProfiler()

Profiler = ControlPlaneProfiler | _NullProfiler


def build_profiler(sink: ControlStageSink, *, enabled: bool) -> Profiler:
    """A live profiler when enabled, else :data:`NULL_PROFILER`."""
    return ControlPlaneProfiler(sink) if enabled else NULL_PROFILER
