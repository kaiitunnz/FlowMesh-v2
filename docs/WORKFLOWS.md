# Workflow YAML format

Workflows are submitted as YAML (or JSON) to `POST /api/v1/workflows`
(see [`docs/API.md`](API.md)). The `examples/templates/` directory contains
runnable examples for each shape; this page documents the spec
hierarchy and the cross-cutting features.

## Single task

```yaml
apiVersion: flowmesh/v1
kind: InferenceTask
metadata:
  name: hello-inference
spec:
  taskType: inference
  resources:
    hardware: { gpu: { type: any, count: 1 } }
  model:
    source: { type: huggingface, identifier: TinyLlama/TinyLlama-1.1B-Chat-v1.0 }
    vllm: { gpu_memory_utilization: 0.5 }
  data:
    type: list
    items:
      - - role: user
          content: What is the capital of France?
  inference: { max_tokens: 64, temperature: 0.0 }
  output:
    destination: { type: http }
```

## Multi-stage DAG

```yaml
apiVersion: flowmesh/v1
kind: Workflow
spec:
  stages:
    - name: extract
      spec:
        taskType: inference
        ...
        data:
          type: list
          items:
            - - role: user
                content: "Extract entities from: {{input}}"
    - name: summarize
      dependsOn: [extract]
      spec:
        taskType: inference
        ...
        data:
          type: list
          items:
            - - role: user
                content: "Summarize: {{extract.output}}"
```

`spec.stages[].dependsOn` declares the DAG edges; the dispatcher
schedules each stage once all of its dependencies are `DONE`.
Substitutions like `{{extract.output}}` are resolved against the
upstream stage's result.

## Graph DAG

`spec.graph.nodes[]` — each node carries a `name`, an optional
`dependsOn`, and a task `spec` (with its own `taskType`). Dependencies
are explicit per node, and cycles are rejected. Multi-input prompts use
`spec.data.type: graph_template` on a downstream node to combine parent
outputs by node name and path; see
`src/worker/executors/utils/graph_templates.py` for the templating
contract.

## v2 execution mode (experimental)

`apiVersion: flowmesh/v2` selects the v2 representation track: the server
compiles the submission into versioned plan-time representations
(logical template and physical plan) and persists them alongside the
workflow. It is off by default — any other `apiVersion` keeps the v1 path.
See [`WORKFLOW_REPRESENTATIONS.md`](WORKFLOW_REPRESENTATIONS.md).

Existing single-task, `spec.stages`, and `spec.graph.nodes` forms compile
unchanged under v2. The constructs below are opt-in and provisional; they
require `apiVersion: flowmesh/v2` and are rejected under v1.

### Leaf declarations (`spec.v2`)

A task carries v2 declarations in a `spec.v2` sub-block, leaving legacy spec
keys untouched:

```yaml
- name: research
  spec:
    taskType: agent
    task: gather sources
    harness: { backend: scripted, version: v1 }
    v2:
      authority: { invoke: [web_search], delegate: [] }
      tools:
        - { name: web_search, interface: search/v1 }
      boundary: [invocation, external_effect, yield]
```

`authority`, `tools`, and `boundary` apply to `agent` leaves. Any leaf may
declare `provenance` (`pinned` | `live`) and `determinism` / `effect` /
`recovery` overrides, and a `result: { visibility: published }` to publish its
induced output.

An agent that runs code declares `sandbox.execute` in its invoke face, which
needs no `tools` entry — the fabric provides the interface. `spec.sandbox`
bounds one command and may be omitted for the defaults:

```yaml
- name: coder
  spec:
    taskType: agent
    task: build and test the project
    harness: { backend: codex, version: v1 }
    sandbox: { command_timeout_sec: 120, memory_bytes: 4294967296 }
    v2:
      authority: { invoke: ["sandbox.execute"], delegate: [] }
```

Commands run in the agent's own workspace on the worker that already holds its
private state, and the runtime denies network egress: reaching a model, tool, or
external effect takes the agent's mediated boundaries instead. The confinement
is a development posture for single-tenant workers, not multi-tenant isolation
(see [`EXECUTORS.md`](EXECUTORS.md)).

A workflow that needs its commands on the network declares the separate
`sandbox.egress` interface and opts the binding in:

```yaml
- name: fetcher
  spec:
    taskType: agent
    task: fetch and summarize the release notes
    harness: { backend: codex, version: v1 }
    sandbox: { network_egress: author_owned_at_least_once }
    v2:
      authority: { invoke: ["sandbox.execute", "sandbox.egress"], delegate: [] }
```

The deployment must also enable `AGENT_SANDBOX_EGRESS_ENABLED`, and a spawned
child gets the opt-in only if its parent delegates `sandbox.egress`. Access is
all-or-nothing IP networking — no destination or port policy — and an
egress-enabled command is an external effect the author owns: a failure before
the episode's next seal may run it again, and the fabric neither deduplicates
nor compensates it.

### Structured regions

In the graph form, a node carries a `region` instead of a `spec`. Regions wire
through `dependsOn` like tasks:

```yaml
spec:
  graph:
    nodes:
      - name: classify
        spec: { taskType: inference, ... }
      - name: route
        dependsOn: [classify]
        region: { kind: branch, selection: "{{classify.output.label}}", ports: [accept, revise] }
      - name: fanout
        dependsOn: [route]
        region: { kind: spawn, child: worker, authority: { invoke: [search] } }
      - name: collect
        dependsOn: [fanout]
        region: { kind: join, completion: all_settled, residual: cancel }
      - name: refine
        dependsOn: [collect]
        region:
          kind: loop
          coordinate: refine
          carried:
            - name: model
              kind: model_ref
              modelRef: { architecture: llama, version: base }
      - name: train
        dependsOn: [refine]
        spec: { taskType: lora_sft, ... }
        feedback: { to: refine, port: model }
```

Region kinds are `branch`, `merge`, `spawn`, `join`, `loop`, and `call`
(`call` normalizes to a `spawn`/`join` pair). A `feedback` edge is a structured
back-edge into a `loop` region; it is excluded from acyclic-topology checks, so
an unstructured `dependsOn` cycle is still rejected.

A `join` `completion` is `all_settled`, `all_succeed`, `any`, `first_k` (with
`k`), or `predicate` (with `predicate: { min_qualifiers, monotone }`). An early
completion (`any`/`first_k`/`predicate`) declares a `residual` policy
(`continue`, `drain`, `cancel`) for children still running when it releases, and
may set `no_winner_failure: true` to resolve a no-winner join as a failure rather
than empty. The winner is the lowest-`child_index` child that qualifies.

### Dry-run inspection

`POST /api/v1/workflows/validate` parses a workflow without executing it. For a
`flowmesh/v2` submission it also compiles the workflow and returns the logical
template, physical plan, and validation diagnostics in the `inspection` field;
compilation errors return `422` with readable source locations. For any other
`apiVersion` it returns the parsed task list with no `inspection`.

## data_retrieval: type lumid

`type: lumid` routes the retrieval through lumid-data-app (HTTP). Three
modes are supported; all require `lumid_data_url` and `lumid_data_token`.

`lumid_data_token` is the bearer forwarded to lumid-data-app (shared lum.id
auth). Set it to your lum.id PAT, or to a key from lumid-data-app's
`LUMID_API_KEYS` for local dev.

```yaml
# SQL mode — single rendered query per param row
data:
  type: lumid
  mode: sql
  lumid_data_url: "http://127.0.0.1:5101"
  lumid_data_token: "${LUMID_PAT}"   # your lum.id PAT, or a local dev key
  template: "SELECT symbol, close FROM demo.fact_ohlc_10m ORDER BY timestamp LIMIT 5"
  output_format: jsonl   # jsonl (default) or csv

# Agent mode — NL description dispatched to the data agent
data:
  type: lumid
  mode: agent
  lumid_data_url: "http://127.0.0.1:5101"
  lumid_data_token: "${LUMID_PAT}"
  description: "Retrieve the latest 10 OHLC rows for NVDA from the demo schema"
  schema_scope: demo
  max_steps: 20
  output_format: jsonl

# S3 Object mode — fetch raw blobs by key
data:
  type: lumid
  mode: s3
  lumid_data_url: "http://127.0.0.1:5101"
  lumid_data_token: "${LUMID_PAT}"
  template: "demo/unstructured/news-html/{slug}"
  params:
    - label: slug
      data:
        type: list
        items:
          - 2024-01-15-nvda-earnings.html
```

## Schedule hints

Workflows can declare scheduling preferences via
`metadata.annotations.schedule_hint`:

- `epoch_groups: [[<task_name>, ...], ...]` — epoch-ordered execution;
  tasks in epoch `n` only dispatch after every task in epoch `n-1`
  succeeds.
- `schedule_in_epoch_order: true` — for dependent DAGs, prefer
  position-in-epoch tie-breaks during dispatch.
