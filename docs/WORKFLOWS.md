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
                content: "Extract the entities from this report: ..."
    - name: summarize
      dependsOn: [extract]
      spec:
        taskType: inference
        ...
        data:
          type: list
          items:
            - - role: user
                content: "Summarize: ${extract.items.0.output}"
```

`spec.stages[].dependsOn` declares the DAG edges; the dispatcher
schedules each stage once all of its dependencies are `DONE`. A stage's
`dependsOn` names an earlier stage of the same workflow.
A placeholder `${stage.path}` reads into an upstream stage's result, which the
server renders before the stage is dispatched.

## Graph DAG

`spec.graph.nodes[]` — each node carries a `name`, an optional
`dependsOn`, and a task `spec` (with its own `taskType`). Dependencies
are explicit per node and name nodes of the same workflow, and cycles are
rejected. Multi-input prompts use
`spec.data.type: graph_template` on a downstream node to combine parent
outputs by node name and path; see
`src/worker/executors/utils/graph_templates.py` for the templating
contract.

## API task

`taskType: api` performs one HTTP request, to `spec.api.url` or, without one, to the
deployment's Nebula endpoint (`NEBULA_API_BASE_URL`, with `/v1/chat/completions`
appended).

`spec.api.retries` (default `0`, at most `10`) sets how many times a transient failure is retried before the task fails. A transient failure is a connection error or an HTTP status of 5xx, 408, or 429; other 4xx statuses are never retried. Retries back off exponentially: the first waits 1s and each later one doubles, capped at 60s. When a retryable response carries a `Retry-After` header (seconds or an HTTP date), that wait is used instead, also capped at 60s. Each retry logs a warning with the attempt count and the wait. A cancelled task stops retrying immediately.

## Inline credentials

A credential written inline in a task spec — a credential-named header, parameter, or
field such as `Authorization`, `api_key`, `password`, or `token`, or a URL carrying a
credential in its userinfo, query, fragment, or path parameters — is kept in its
workflow's credential vault. The stored task, its source, the task API, and the SDK show
`[REDACTED]` in its place, and only the task's own dispatch carries the value to its
worker. The credential lives until its workflow settles, across server restarts; a task
whose credential is no longer retained fails with `credential_not_retained`. Tasks
carrying the same credential merge only within one workflow.

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

`authority`, `tools`, and `boundary` apply to `agent` leaves. An agent input
`{ name, from: <agent>, region: <role> }` reads the aggregate of that agent's
child region; the named agent runs at the root, and is neither the reader nor
downstream of it. Any leaf may declare `provenance` (`pinned` | `live`) and
`determinism` / `effect` / `recovery` overrides, and a
`result: { visibility: published }` to publish its induced output. A spawn region
publishes its children's results as a collection keyed by child index within each
spawning scope:

```yaml
- name: fanout
  region: { kind: spawn, child: reviewer, result: { visibility: published } }
```

Clients read a published output by the name of the node that declares it.

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
`sandbox.egress` interface, which opts the agent's own commands in:

```yaml
- name: fetcher
  spec:
    taskType: agent
    task: fetch and summarize the release notes
    harness: { backend: codex, version: v1 }
    v2:
      authority: { invoke: ["sandbox.execute", "sandbox.egress"], delegate: [] }
```

An agent that declares the interface only to delegate it keeps its own commands
fenced with `sandbox: { network_egress: deny }`. The deployment must also enable
`AGENT_SANDBOX_EGRESS_ENABLED`, and a spawned child gets the opt-in only if its
parent delegates `sandbox.egress`. Access is all-or-nothing IP networking — no
destination or port policy — and an egress-enabled command is an external effect
the author owns: a failure before the episode's next seal may run it again, and
the fabric neither deduplicates nor compensates it.

### Structured regions

In the graph form, a node carries a `region` instead of a `spec`. Regions wire
through `dependsOn` like tasks:

```yaml
spec:
  graph:
    nodes:
      - name: plan
        spec: { taskType: inference, ... }
      - name: fanout
        dependsOn: [plan]
        region: { kind: spawn, child: worker, authority: { invoke: [search] } }
      - name: collect
        dependsOn: [fanout]
        region: { kind: join, completion: all_settled, residual: cancel }
      - name: report
        dependsOn: [collect]
        spec: { taskType: echo, ... }
```

Region kinds are `branch`, `merge`, `loop`, `spawn`, `join`, and `call` (`call`
normalizes to a `spawn`/`join` pair). A spawn or call fans out over its one
unnamed input: a task's result, a part of one, or a branch arm carrying one. A
join collects a spawn's children, so one of its inputs is a spawn. Only a join may
depend on a spawn, and a node that depends on a call reads the call's join. A
spawn's child task reads only the element it is spawned with; a child template
reads parent values through its `capture` inputs. A failed input fails the region
and everything downstream of it, as a failed dependency fails a task.

A `dependsOn` entry is a node name or a mapping
`{ node, port, input, project }`: `port` names the output it reads (a branch arm,
a loop's carried value, a call's return), `input` names the value for the
consumer, and `project` selects a part of it by field names and list indexes.

A `join` `completion` is `all_settled`, `all_succeed`, `any`, `first_k` (with
`k`), or `predicate` (with `predicate: { min_qualifiers, monotone }`). An early
completion (`any`/`first_k`/`predicate`) declares a `residual` policy
(`continue`, `drain`, `cancel`) for children still unsettled when it releases;
`cancel` cancels them, interrupting any already running, with everything a
cancelled agent child spawned. It may set `no_winner_failure: true` to resolve a
no-winner join as a failure rather than empty. The winner is the
lowest-`child_index` child that qualifies. An `all_succeed` join with a failed
child, or a no-winner join under `no_winner_failure`, resolves as a failure and
fails everything downstream of it.

#### Branches and merges

A `branch` routes its input to exactly one of its `outputs`. Its `selection` names
the `input` it reads, the `field` path to a string inside it, and optionally
`cases` mapping each value to a port; without `cases` the value names the port.
A consumer depends on the arm it takes:

```yaml
      - name: classify
        spec: { taskType: echo, ... }
      - name: route
        dependsOn: [{ node: classify, input: input }]
        region:
          kind: branch
          inputs: [{ name: input }]
          outputs: [{ name: accept }, { name: revise }]
          selection: { input: input, field: [items, 0, output], cases: { "ok": accept, "redo": revise } }
      - name: publish
        dependsOn: [{ node: route, port: accept }]
        spec: { taskType: echo, ... }
```

Case values are strings, so quote any that YAML would read otherwise. A value that
is not a string, or matches no case or port, fails the branch. Work on an arm the
branch did not take settles without running, and so does everything that needs
it. A `merge` joins arms back together: `combination: one_live` forwards the one
live arm's value, and `concat`, the default, collects every live input in
declared order. A branch selects on, and a spawn fans out over, a `one_live`
merge only when each of its arms carries one value rather than a join's or a
`concat` merge's aggregate.

#### Graph templates

`spec.graph.templates` declares named, finite subgraphs that loops run as their
body and spawns or calls run per child. A template declares its `inputs`, each
with a `role` (`carried` or `invariant` for a loop body, `param` for a child's
element or call argument, `capture` for a parent value a child reads), its
`returns` for a call, its `nodes`, and `edges` out of it. Inside a template a
node reads a template input by depending on `$ingress`, and an edge leaves
through `$feedback` (the next loop iteration), `$egress` (the loop's exit), or
`$return` (a call's result). An edge may carry a `project`.

#### Loops

A `loop` runs its `body_ref` template once per iteration. Its `carried` inputs
seed iteration 0 from its dependencies and are replaced by each `$feedback`;
its `invariants` bind once and stay readable at every iteration. The body
decides each iteration with a branch whose arms lead to `$feedback` or `$egress`;
here `revise` returns structured output carrying a `verdict` and the `text`:

```yaml
spec:
  graph:
    templates:
      - name: refine_body
        inputs: [{ name: draft, role: carried }]
        nodes:
          - name: revise
            dependsOn: [{ node: $ingress, port: draft, input: draft }]
            spec: { taskType: inference, ... }
          - name: judge
            dependsOn: [{ node: revise, input: input }]
            region:
              kind: branch
              inputs: [{ name: input }]
              outputs: [{ name: again }, { name: done }]
              selection: { input: input, field: [items, 0, output, verdict] }
        edges:
          - from: { node: judge, port: again }
            to: { node: $feedback, port: draft }
          - from: { node: judge, port: done }
            to: { node: $egress, port: draft }
    nodes:
      - name: first_draft
        spec: { taskType: inference, ... }
      - name: refine
        dependsOn: [{ node: first_draft, input: draft }]
        region:
          kind: loop
          body_ref: refine_body
          carried: [{ name: draft }]
      - name: publish
        dependsOn: [{ node: refine, port: draft, input: final }]
        spec: { taskType: echo, data: { type: list, items: ["${final.items.0.output.text}"] } }
```

A loop's value is the carried value its body exits with, and a consumer names
the carried port it reads. A loop declaring `result: { visibility: published }`
publishes one carried port, which `result.source_port` names when it has
several. An iteration starts as soon as the previous one feeds back, while work
of earlier iterations it does not depend on may still run, and the loop exits
only once all of it settles; dependencies alone order work across iterations.
`ORCHESTRATOR_MAX_LOOP_ITERATIONS` bounds how many iterations a loop runs, its
first included, and a body that feeds back past them fails the loop.
[`refine_loop_echo.yaml`](../examples/templates/refine_loop_echo.yaml) runs a
two-iteration loop with echo tasks.

#### Reading values

Every input a task reads is a value, and `${name.path}` reads into it: a whole
task result, the part a `project` selects, a fan-out element, a region's value,
a literal, or `null` for an empty value. An aggregate, as a join or a `concat`
merge delivers, reads as a list of `{key, outcome, value}`, where `value` is
`null` unless the member succeeded. A task reads an input by its `input` name, and
an upstream node by that node's name as the node's whole value; a node with
several output ports, as a branch or a loop carrying several values, is read by
the `input` name of a dependency naming one port. An SSH task's `inputs[].stage`
names a task's result, never an aggregate. Inside a
template, `${name}` reads a value whole, and `${name.task_id}` names the task
another node of the template ran as in the same iteration and child. A template
input, a projected input, or a region's value has no task, so `${name.task_id}`
on one is refused at submission.

Each task a template runs reports where it ran as `occurrence` in its task
information: the template member, the child it belongs to, and the iteration of
each loop around it, named by the loop's `loop_coordinate`, its node name unless
it declares one.

### Dry-run inspection

`POST /api/v1/workflows/validate` parses a workflow without executing it and lists
the tasks a submission registers. For a `flowmesh/v2` submission it also compiles
the workflow and returns the logical template, physical plan, and validation
diagnostics in the `inspection` field; compilation errors return `422` with
readable source locations. For any other `apiVersion` it returns the parsed task
list with no `inspection`.

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
