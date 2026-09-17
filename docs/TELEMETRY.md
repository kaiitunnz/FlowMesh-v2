# Telemetry

FlowMesh emits OpenTelemetry traces and metrics from the server, supervisor, and worker
processes, controlled by one shared knob and gated to zero overhead when disabled.

## Configuration

`SERVER_METRICS_TELEMETRY_LEVEL` selects a verbosity level: `off` (default), `coarse`,
`fine`, or `full`. `SERVER_METRICS_TRACES_ENABLED` and `SERVER_METRICS_METRICS_ENABLED`
gate traces and metrics independently within that level. `SERVER_METRICS_OTLP_ENDPOINT`
points at an OTel Collector; when unset, no exporter is constructed regardless of level.
`SERVER_METRICS_TRACE_SAMPLE_RATIO` sets the fraction of workflows traced.
`SERVER_METRICS_OTLP_TIMEOUT_SEC` and `SERVER_METRICS_RESOURCE_SAMPLE_SEC` tune the
exporter timeout and the worker-side resource-sampling interval. See
[`ENV.md`](ENV.md) for defaults.

These vars are read once, at the config edge (`server.config.MetricsConfig.from_env`),
into a `shared.telemetry.config.TelemetryConfig`, and reach worker and supervisor
processes through the supervisor's worker-environment allowlist.

## `off` is free

At `off`, no `TracerProvider` or `MeterProvider` is constructed, and no OTLP exporter
thread starts. `shared.telemetry.provider.build_tracer` and `build_meter` return a null
twin instead of a real provider-backed tracer/meter: a disabled call site costs one
attribute read and no allocation, since the twin's span context manager is a single
reused instance rather than a fresh one per span.

## Identity

`shared.telemetry.ids.workflow_to_trace_id_int` derives a trace id from a workflow id
directly, so every producer computes the same trace id for a workflow independently,
with no coordination. `derived_span_id` similarly derives a deterministic, non-zero span
id for a long-lived entity (workflow, activation, work item, attempt, invocation) from
its durable id and a `SpanIdKind`; every other span uses an ordinary random id.

## Semantic conventions

`shared.telemetry.semconv` defines every `flowmesh.*` attribute key and span name used
by the substrate: the `flowmesh.logical.*` / `flowmesh.physical.*` attribute namespaces,
the resource attributes (`flowmesh.node_id`, `flowmesh.worker_id`, `flowmesh.role`), and
the control-plane stage (`ControlPlaneStage`) and window (`ControlPlaneWindow`) taxonomy
that attributes a control-plane span to where it fires relative to a task's lifetime.
