"""Measure the v2 fabric's control-plane overhead against the v1 static-DAG path.

Submits one workflow body twice — once as ``flowmesh/v1``, once as ``flowmesh/v2`` —
over the same server, dispatcher and worker pool, so the difference between the two runs
is the v2 control plane. Reports the submit-window and queue-window deltas, and the v2
stage decomposition read from ``GET /api/v1/system/metrics``.

Needs ``SERVER_METRICS_CONTROL_PROFILING=1`` on the server; without it the run reports
the aggregate deltas and an empty breakdown.

    uv run scripts/dev/profile_v2_overhead.py examples/templates/<workflow>.yaml
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from flowmesh import FlowMesh

_V1_API_VERSION = "flowmesh/v1"
# Reported beside a window total rather than inside it: it runs within other stages.
_NESTED_STAGE = "ledger_snapshot"
_V2_API_VERSION = "flowmesh/v2"


def _as_mode(body: str, api_version: str) -> str:
    """The same body pinned to one execution track."""
    lines = [line for line in body.splitlines() if not line.startswith("apiVersion:")]
    return "\n".join([f"apiVersion: {api_version}", *lines])


def _queue_totals(metrics: dict[str, Any]) -> tuple[float, int]:
    overall = metrics.get("task_timings", {}).get("overall", {})
    return float(overall.get("queue_total_sec") or 0.0), int(
        overall.get("queue_count") or 0
    )


def _run(
    client: FlowMesh, body: str, api_version: str
) -> tuple[str, float, float | None]:
    """Submit one track and return its id, submit duration and mean queue time.

    The queue figure is this run's own: the recorder's aggregate is cumulative, so the
    run's mean is the difference in queue total over the difference in queue count.
    """
    before_total, before_count = _queue_totals(client.system.metrics())
    started = time.perf_counter()
    submitted = client.workflows.submit(_as_mode(body, api_version))
    submit_seconds = time.perf_counter() - started
    client.workflows.wait(submitted.workflow_id)
    after_total, after_count = _queue_totals(client.system.metrics())
    settled = after_count - before_count
    queue_seconds = (after_total - before_total) / settled if settled > 0 else None
    return submitted.workflow_id, submit_seconds, queue_seconds


def _breakdown(metrics: dict[str, Any], workflow_id: str) -> dict[str, Any]:
    workflows = metrics.get("v2_control_plane", {}).get("workflows", {})
    entry = workflows.get(workflow_id)
    return entry.get("windows", {}) if entry else {}


def _format_windows(windows: dict[str, Any]) -> str:
    if not windows:
        return "  (no stages recorded — is SERVER_METRICS_CONTROL_PROFILING set?)"
    lines = []
    for name, window in windows.items():
        stages = window.get("stages", {})
        if not stages:
            continue
        lines.append(f"  {name}: {window['total_sec'] * 1e3:.2f} ms")
        # The window total covers only its non-nested stages; the nested ones are
        # already inside them and print below as an "of which" line.
        counted = {
            stage: entry for stage, entry in stages.items() if stage != _NESTED_STAGE
        }
        for stage, entry in sorted(
            counted.items(), key=lambda item: -item[1]["total_sec"]
        ):
            lines.append(
                f"    {stage:<20} {entry['total_sec'] * 1e3:8.2f} ms"
                f"  x{entry['count']}"
            )
        if window.get("nested_sec"):
            lines.append(
                f"    (of which {_NESTED_STAGE} {window['nested_sec'] * 1e3:.2f} ms)"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path, help="Workflow YAML to submit")
    parser.add_argument(
        "--repeat", type=int, default=3, help="Runs per track (default 3)"
    )
    parser.add_argument("--json", action="store_true", help="Emit the raw breakdown")
    args = parser.parse_args()

    body = args.workflow.read_text()
    client = FlowMesh()

    submit_times: dict[str, list[float]] = {_V1_API_VERSION: [], _V2_API_VERSION: []}
    queue_times: dict[str, list[float]] = {_V1_API_VERSION: [], _V2_API_VERSION: []}
    last_v2_workflow = ""

    for _ in range(args.repeat):
        for api_version in (_V1_API_VERSION, _V2_API_VERSION):
            workflow_id, submit_seconds, queue_seconds = _run(client, body, api_version)
            submit_times[api_version].append(submit_seconds)
            if queue_seconds is not None:
                queue_times[api_version].append(queue_seconds)
            if api_version == _V2_API_VERSION:
                last_v2_workflow = workflow_id

    metrics = client.system.metrics()
    windows = _breakdown(metrics, last_v2_workflow)

    if args.json:
        json.dump(
            {
                "submit_seconds": submit_times,
                "queue_seconds": queue_times,
                "v2_windows": windows,
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    v1_submit = statistics.mean(submit_times[_V1_API_VERSION])
    v2_submit = statistics.mean(submit_times[_V2_API_VERSION])
    print(f"runs per track: {args.repeat}")
    print(
        f"submit  v1 {v1_submit * 1e3:8.2f} ms   v2 {v2_submit * 1e3:8.2f} ms"
        f"   delta {(v2_submit - v1_submit) * 1e3:+8.2f} ms"
    )
    if queue_times[_V1_API_VERSION] and queue_times[_V2_API_VERSION]:
        v1_queue = statistics.mean(queue_times[_V1_API_VERSION])
        v2_queue = statistics.mean(queue_times[_V2_API_VERSION])
        print(
            f"queue   v1 {v1_queue * 1e3:8.2f} ms   v2 {v2_queue * 1e3:8.2f} ms"
            f"   delta {(v2_queue - v1_queue) * 1e3:+8.2f} ms"
        )
    print(f"\nv2 control-plane stages ({last_v2_workflow}):")
    print(_format_windows(windows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
