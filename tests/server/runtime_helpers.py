"""Shared builders for task runtimes under test."""

import logging
from collections.abc import Callable

from server.task.workflow_retry import WorkflowRetryScheduler


def manual_durability_retry(
    fire: Callable[[str], None], logger: logging.Logger
) -> WorkflowRetryScheduler:
    """A durability retry that runs only when a test drives it."""
    return WorkflowRetryScheduler(fire, logger, base_delay_sec=0.0, run_thread=False)
