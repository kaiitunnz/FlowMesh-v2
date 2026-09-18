"""Every process derives a workflow's trace id from one function object.

Producers agree on a workflow's trace id with no coordination only because they run the
same derivation. A second copy of it is correct on the day it is written and silently
splits the trace the day either copy changes -- with no failing test, because each copy
is internally consistent. These guard the property structurally rather than by comparing
outputs, which a byte-identical copy would satisfy.
"""

import re
from pathlib import Path

from shared.telemetry import ids
from worker.executors.mixins import _otel

_SRC = Path(__file__).resolve().parents[3] / "src"
_DEFINITION = re.compile(r"^def workflow_to_trace_id_int\b", re.MULTILINE)


def test_the_worker_derives_trace_ids_with_the_canonical_function() -> None:
    assert _otel.workflow_to_trace_id_int is ids.workflow_to_trace_id_int


def test_the_derivation_is_defined_exactly_once() -> None:
    defining = [
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob("*.py")
        if _DEFINITION.search(path.read_text(encoding="utf-8"))
    ]
    assert defining == ["shared/telemetry/ids.py"]
