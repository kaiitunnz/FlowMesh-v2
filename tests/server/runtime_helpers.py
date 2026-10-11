"""Shared builders for task runtimes under test."""

import logging
from collections.abc import Callable
from typing import Any

from server.orchestration.state import InvocationState
from server.task.workflow_retry import WorkflowRetryScheduler
from tests.server.stored_state import StoredLedgers


def manual_durability_retry(
    fire: Callable[[str], None], logger: logging.Logger
) -> WorkflowRetryScheduler:
    """A durability retry that runs only when a test drives it."""
    return WorkflowRetryScheduler(fire, logger, base_delay_sec=0.0, run_thread=False)


def durable_invocation(
    registry: StoredLedgers, workflow_id: str, invocation_id: str
) -> InvocationState | None:
    """The state an invocation has in a workflow's durable ledger, if any."""
    if (stored := registry.load_ledger(workflow_id)) is None:
        return None
    return next(
        (
            i.state
            for i in stored.snapshot.invocations
            if i.invocation_id == invocation_id
        ),
        None,
    )


def refuse_ledger_saves(registry: Any) -> None:
    """Refuse every ledger save the registry is asked for."""

    def down(*_: Any, **__: Any) -> None:
        raise ConnectionError("control redis unavailable")

    setattr(registry, "save_ledger", down)


def accept_ledger_saves(registry: Any) -> None:
    """Take ledger saves again after ``refuse_ledger_saves``."""
    vars(registry).pop("save_ledger", None)
