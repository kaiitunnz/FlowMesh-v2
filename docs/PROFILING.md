# Control-plane profiling

A `flowmesh/v2` submission runs through the same process, dispatcher and worker pool as
a `flowmesh/v1` static DAG, branched by a single `apiVersion` field. The v2 track pays
control-plane cost the v1 track does not — template compilation, orchestration-ledger
drive and settle, ledger snapshot serialization, resident admission, mediated-boundary
permit minting, and relay establishment. Control-plane profiling times those stages and
reports them per workflow.

Enable it with `SERVER_METRICS_CONTROL_PROFILING=1` on the server. It is off by default,
and off it records nothing.

## Windows

Each measurement carries the window it fires in. The windows are separate aggregates
because they sit in different places relative to a task's lifetime.

| Window | When | Stages |
|--------|------|--------|
| `submit` | inside the submit request, before any task's queue window opens | `compile_lower`, `compile_assemble`, `compile_episodes`, `compile_validate`, `engine_build`, `ds_initial_advance` |
| `queue` | between a task's submission and its start | `dispatch` |
| `post_start` | mid-episode, after the task has started | `ds_drive`, `admission`, `permit`, `relay` |

`engine_build` covers opening the plan's roots, so `ds_initial_advance` is the cost of
admitting that advance into the queue rather than of computing it.

`ledger_snapshot` fires in every window — the ledger serializes during submission,
dispatch and outcome settlement alike. It runs inside another stage, so a window's
`total_sec` excludes it and reports it separately as `nested_sec`.

A window's `total_sec` is the sum of its non-nested stages, so it reconciles against the
aggregate it decomposes: `submit` against the submit request's duration, `queue` against
the task's recorded `queue_time`.

## Reading the breakdown

`GET /api/v1/system/metrics` carries a `v2_control_plane` section when profiling is on:

```json
{
  "v2_control_plane": {
    "workflows": {
      "wfl-...": {
        "windows": {
          "submit": {
            "total_sec": 0.031,
            "nested_sec": 0.004,
            "stages": {
              "compile_lower": {"count": 1, "total_sec": 0.012, "avg_sec": 0.012, "max_sec": 0.012}
            }
          }
        },
        "invocations": 0
      }
    }
  }
}
```

`invocations` counts the distinct invocations the post-start stages recorded against the
workflow.

The oldest workflow is evicted once the breakdown is tracking its cap, so it reports
recent runs rather than a server's whole history.

## Comparing v1 and v2

`scripts/dev/profile_v2_overhead.py` submits one workflow body on both tracks against a
running stack and prints the per-window deltas beside the v2 stage decomposition:

```bash
uv run scripts/dev/profile_v2_overhead.py examples/templates/<workflow>.yaml --repeat 5
```

Two things bound what the comparison means:

- **A delta is only meaningful for a task type both tracks run.** `agent` is v2-only, and
  resident and `serve` tasks have no v1 twin; their stages read as absolute timings.
- **The queue delta is only clean on a root task.** Every task of a workflow is stamped
  submitted at the same moment, so a downstream task's queue time spans its upstreams'
  execution. Point the comparison at a workflow whose tasks have no upstream — a
  single-task workflow gives an exact figure.

## Attribution

Every measurement is keyed by `workflow_id`; the post-start stages also carry
`invocation_id`. A gated `serve` request is driven by an external principal rather than a
workflow, so it owns no `workflow_id` and records no stage.
