"""Worker-wide test isolation for process-global state.

The worker's telemetry config and tracer provider are module globals written once per
process: ``otel.configure`` sets the config, and ``_ensure_tracer_provider`` builds the
provider and hands it to OpenTelemetry's own global. Production writes them once at
startup; a test session writes them many times, and anything a test leaves behind is
inherited by every test after it. ``Runner.__init__`` calls ``configure`` on its own, so
a test that merely builds a Runner leaks its level -- which is enough to make an "emits
nothing at off" assertion pass or fail on what ran before it.
"""

import os
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.util._once import Once

from worker.executors.utils.collective import loopback_collective_env
from worker.hw import visible_gpus
from worker.telemetry import otel


@pytest.fixture(autouse=True)
def _restore_worker_tracer_globals() -> Iterator[None]:
    """Snapshot the worker's tracer globals and put them back after every test."""
    config = otel._telemetry_config
    initialized = otel._PROVIDER_INITIALIZED
    provider = trace._TRACER_PROVIDER
    set_once = trace._TRACER_PROVIDER_SET_ONCE
    try:
        yield
    finally:
        otel._telemetry_config = config
        otel._PROVIDER_INITIALIZED = initialized
        trace._TRACER_PROVIDER = provider
        # Without restoring the guard as well, a later ``set_tracer_provider`` is
        # logged and ignored, so the next test silently keeps this one's provider.
        trace._TRACER_PROVIDER_SET_ONCE = set_once if provider is not None else Once()


@pytest.fixture(autouse=True)
def _fresh_visible_gpus() -> Iterator[None]:
    """Resolve each test's visible GPUs afresh, as each worker process does."""
    visible_gpus.cache_clear()
    yield
    visible_gpus.cache_clear()


@pytest.fixture(autouse=True)
def _restore_collective_env() -> Iterator[None]:
    """Undo the collective-transport variables a local engine sets in the process."""
    saved = {name: os.environ.get(name) for name in loopback_collective_env()}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
