# Telemetry

FlowMesh emits one OpenTelemetry trace per workflow, spanning every process the workflow
touches: the root server's control plane, each supervisor it dispatches through, and each
worker that runs a task. Spans, and the metrics that summarize them, export over OTLP to a
collector and are queried back through the server.

Telemetry is observation only. No span or metric is read by admission, credit release,
embodiment selection, dispatch, or recovery, and nothing it records enters the
orchestration ledger. Enable it with `SERVER_METRICS_TELEMETRY_LEVEL`; it is off by
default (`docs/ENV.md` lists the knobs).

## Configuration

`SERVER_METRICS_TELEMETRY_LEVEL` (`off` | `coarse` | `fine` | `full`) sets verbosity, and
`SERVER_METRICS_TRACES_ENABLED` / `SERVER_METRICS_METRICS_ENABLED` gate traces and metrics
independently within the selected level. `SERVER_METRICS_OTLP_ENDPOINT` names the
collector; leaving it unset builds no exporter at all, whatever the level.
`SERVER_METRICS_TRACE_SAMPLE_RATIO` sets the fraction of workflows traced,
`SERVER_METRICS_OTLP_TIMEOUT_SEC` bounds an export, and
`SERVER_METRICS_RESOURCE_SAMPLE_SEC` sets the worker-side resource-sampling interval.

All seven are read once at the config edge into a `TelemetryConfig` and reach every
supervisor and worker process through the worker-environment allowlist, so the root,
supervisors, and workers always agree on the level.

At `off` nothing is built: no `TracerProvider`, no `MeterProvider`, no OTLP exporter and no
exporter thread. A disabled tracer hands back one reused context manager on every call, so
an instrumentation site costs a single attribute read.

The worker's own tracer is the exception, and it is deliberate: the tracer that writes each
task's `spans.jsonl` for the governance analyzer predates this knob and runs
unconditionally. The level gates the OTLP export and the `flowmesh.*` spans, never the
JSONL sink — so `flowmesh trace analyze` and its `ProfileSummary` behave identically at
every level, including `off`.

## Identity

Every producer derives a workflow's `trace_id` from its `workflow_id` by the same pure
function, so the root, a supervisor, and a worker independently agree on it with no
coordination and a trace survives a server restart. A workflow's trace is therefore a
direct key lookup rather than an index scan, and `flowmesh trace tree <workflow_id>` needs
no workflow-to-trace mapping.

Five entity kinds outlive any single process — workflow, activation, work item, attempt,
and invocation. Each gets a deterministic, non-zero `span_id` derived from its durable id
and its kind, so a worker can name a parent span the server has not exported yet, and a
span for an entity that spans hours is built from the ledger's own recorded timestamps
rather than held open. Everything shorter-lived uses an ordinary random span id.

A gated `serve` request is driven by an external principal and owns no workflow, so it
roots its own trace keyed by its serve task and request instead.

An inbound `traceparent` from outside the fabric is recorded as a span **link**, never
adopted as the parent. Adopting it would take the caller's `trace_id` and a workflow's
trace would then differ between a CI-invoked and a CLI-invoked run.

## The span tree

A workflow's trace nests as the ledger does: the workflow span holds the control-plane
stages and the operator activations beneath it; an activation holds its episodes; an
episode holds its dispatch, its attempts, the worker-side task span, and the boundaries it
opens. Attributes are split into two namespaces — `flowmesh.logical.*` for operator,
activation, scope and result identity, and `flowmesh.physical.*` for work item, attempt,
worker, claim, permit and transport identity — so a consumer builds the logical view by
reading one prefix.

Spans carry ids and digests only. A prompt, a completion, a tool argument or result, a
credential, content bytes, a relay payload, or a header value never appears on a span or a
metric, and the collector drops any attribute outside the published set.

## Control-plane stages

A `flowmesh/v2` submission pays server-side control cost a `flowmesh/v1` static DAG does
not: template compilation, orchestration-ledger drive and settle, ledger snapshot
serialization, dispatch, resident admission, mediated-boundary permit minting, and relay
establishment. Each is timed as its own span under the workflow, named
`flowmesh.control.<stage>`.

Each stage span carries the window it fires in — `submit` inside the submit request,
`queue` between a task's submission and its start, `post_start` mid-episode — because where
a stage falls relative to a task's lifetime is what makes the v1-versus-v2 comparison
decomposable, and it is not recoverable from the shape of the tree. Since both tracks run
through one process, one dispatcher, and one worker pool, submitting the same workflow body
on each and comparing by window isolates the v2 control plane's cost.

## Metrics

Alongside the spans, a deployment with `SERVER_METRICS_METRICS_ENABLED` gets two
series families, sampled every `SERVER_METRICS_RESOURCE_SAMPLE_SEC` seconds and
carrying the same attribute keys the spans do, so a series joins the trace it
summarizes rather than being correlated by wall clock.

Each worker reports GPU utilization, memory, power and temperature per device, keyed
by worker and distinguished per device where a worker has several. A worker without an
accelerator reports nothing and logs nothing. The node a series belongs to is reached
by joining its worker id through the worker registry, which is where a worker's node is
recorded.

The root server reports the resident fleet per service family — replica count,
admission slots in use, and claim credit held — alongside the ready-queue depth.
A deployment with resident capacity disabled still reports queue depth.

`GET /api/v1/system/metrics` is unchanged.

## Storage

A deployment that wants the bundled store selects the `telemetry` Compose profile,
which adds an OpenTelemetry Collector and a ClickHouse instance beside the core stack
and leaves that stack otherwise untouched. The Collector receives OTLP on 4317 and
4318, writes spans and metrics to ClickHouse, and is the store's sole writer;
`TELEMETRY_CLICKHOUSE_DSN` points it at the bundled instance or at one the deployment
already runs. ClickHouse keeps its data in a named volume, so a stack restart does not
discard a trace.

The server's read path is configured separately, through `SERVER_METRICS_CLICKHOUSE_*`,
and never writes. The two halves commonly address the same instance but are never the
same configuration, so either one can be repointed or replaced without the other: the
collector and the read port never touch.

## Querying

`flowmesh trace tree <workflow-id>` renders a workflow's spans as an indented tree, one
line per span carrying its name, its duration, and the one or two ids that identify its
level; `--json` emits the same tree with the logical and physical attribute views intact
and separate. `flowmesh trace aggregate --metric <name> --group-by <attribute>` rolls one
metric up by one attribute key, with `--stat` selecting count, sum, avg, min, max, p50,
p95 or p99, `--kind` selecting the gauge or histogram table, and `--workflow-id`
restricting the aggregate to one workflow. The same two queries are `client.traces.tree()`
and `client.traces.aggregate()` on the SDK.

Both read through the server, which resolves them against the store behind its read port;
neither the CLI nor the SDK holds a store driver, so replacing the store is a collector
configuration change plus one adapter, with no client change. A workflow whose spans were
never recorded returns an empty tree, and a deployment with no store configured answers
that telemetry querying is unavailable there.

`flowmesh trace fetch` and `flowmesh trace analyze` are a different instrument on a
different pipeline — the worker-side `spans.jsonl` the governance analyzer reads — and
behave identically at every telemetry level.

## Propagation

Context crosses each process hop in the envelope's metadata half, never in an opaque
payload body: the HTTP `traceparent` at submission, a field inside each dispatch kind's own
payload container, a field on task and worker events, a field on the mediated-operation
permit, a field on every relay frame across all three transports, and the existing header
dict on worker-to-server HTTP. At `off` the field is omitted entirely, so a disabled
deployment puts zero extra bytes on any wire.
