# Architecture

FlowMesh is a service fabric for running LLM agentic workflows on
distributed GPU workers. The server parses a workflow (YAML / JSON / n8n),
turns it into a DAG of tasks, dispatches each task to a worker, and
collects results and artifacts.

## Workspace layout

The codebase is a **uv workspace** with these packages:

| Package | Path | Purpose |
|---------|------|---------|
| `flowmesh` (root) | `pyproject.toml` | Lightweight PyPI metapackage |
| `flowmesh-sdk` | `sdk/` | Public Python SDK |
| `flowmesh-sdk-stack` | `sdk/stack/` | Stack/node helpers |
| `flowmesh-cli` | `cli/` | Typer CLI (`flowmesh ...`) |
| `flowmesh-cli-stack` | `cli/stack/` | Stack deployment commands |
| `flowmesh-hook` | `hook/` | Plugin hook protocol interfaces |
| Runtime source | `src/` | Server, Worker, shared runtime modules |

Only the SDK, CLI, stack helper, hook, and lightweight `flowmesh`
metapackage distributions are published to PyPI. The runtime source under
`src/` is copied into server and worker images directly and is not included in
the published `flowmesh` wheel.

## Topology

```
Client (CLI / SDK / API) ──▶ Server (FastAPI orchestrator, :8000)
                                │
                                ├── Redis (control + telemetry pub/sub, log streams)
                                │
                                └─▶ Supervisor (per-node) ──gRPC──▶ Worker (executor)
```

The runtime is two top-level processes:

1. **Server** (`src/server/`) — FastAPI orchestrator at `:8000`. Hosts
   workflow / task / dispatch logic and the **Supervisor subsystem**
   (`src/server/supervisor/`), which manages per-node worker lifecycle,
   runs the worker-facing gRPC server (`:50051`), and drives the
   Docker / Vast.ai worker adapters.
2. **Worker** (`src/worker/`) — stateless executor. Connects to a
   supervisor via gRPC, receives tasks, runs the matching executor,
   reports results.

## Communication

- **server ↔ supervisor (same node)** — `multiprocessing.Queue`.
- **server ↔ supervisor (across nodes)** — Redis pub/sub.
- **supervisor ↔ worker** — bidirectional gRPC. Proto stubs at
  `src/shared/grpc/supervisor/v1/`.
- **client ↔ server** — REST.

## Object IDs

3-char prefixes: `wfl-` workflows, `tsk-` tasks, `ssn-` SSH sessions,
`scn-` SSH connection rows, `cmd-` supervisor commands. The v2 orchestration
ledger adds `act-` activations, `scp-` scopes, `wki-` work items, `att-`
attempts, `inv-` invocations, `agr-` authority grants, and `idm-` idempotency
keys (the fabric-assigned dedupe authority for a mediated boundary). Resident-capacity
control adds `scl-` service claims, `rpl-` replica incarnations, and `lse-` allocation
leases. `msk-` is an unguessable ref for a workflow's vaulted model credential and `hnd-`
an unguessable claim-bound admission handoff token. The network plane adds `rog-` route
origins and `rly-` relay sessions. Always use `new_*_id()`
helpers in `src/shared/utils/ids.py`. Never use `uuid4()` or `secrets.token_hex`
for IDs.

## Task state machine

`PENDING → DISPATCHED → (DONE | FAILED | CANCELLED)`. Retried tasks
cycle back to `PENDING` until exhausted.

Retries are routed to a worker that has not already failed the task and
stop once every eligible worker has been tried or `max_attempts` is
reached; the terminal error is the executor's own message. Controlled
executor errors are not retried. A task that no worker can satisfy fails
after `TASK_NO_WORKER_GRACE_SEC`.

## Directory map

```
src/
  server/               FastAPI orchestrator
    auth/                 Helpers for calling plugins' auth and permission check hooks
    clients/              Client wrappers to connect to external services like Redis
    dispatcher/           Dispatch loop, worker selector, stage stickiness, context reuse
    governance/           Governance schemas and trace analysis
    hooks/                Plugin extension ABCs + registries
    main.py               Entrypoint, FLOWMESH_PLUGINS loader, EventMonitor wiring
    network/              Network plane: endpoint directory, reachability, resolver, relay
    orchestration/        Durable orchestration ledger (DS), engine, outcomes
    registries/           Worker / Node registries (Redis-backed)
    routers/v1/           workflows, tasks, results, workers, nodes, ssh, stack, system
    schemas/              REST API request and response schemas
    services/             monitoring, log streaming, ssh forwarding, runtime
    supervisor/           Per-node agent (gRPC server, adapters, lifecycle)
    task/                 parser, runtime, models, merge / epoch helpers
      v2/                   versioned representations, compiler
    utils/                concurrent, helpers, logging, misc, time
  shared/
    grpc/supervisor/v1/   Generated proto stubs (server + worker)
    schemas/              Cross-cutting schemas
    tasks/                Workflow/task spec models
    utils/                JSON, parsing, time, ids
  worker/
    docker/               Worker Dockerfiles (CPU + GPU)
    executors/            Executor implementations
      harness/              agent-episode harness backends (scripted, codex) + registry
      mixins/               data, governance, inference, training
      utils/                artifacts, checkpoints, data_utils, distributed,
                            graph_templates, huggingface, safe_eval
    runner.py             Task lifecycle (execute, write results, upload artifacts)
cli/                    Typer CLI (`flowmesh`)
hook/                   Plugin hook protocol interfaces
sdk/                    Public Python SDK
proto/                  gRPC service definition
examples/               Workflow YAMLs, sample configs, plugin examples
tests/{server,worker,shared,cli,sdk}/
scripts/dev/            compile_protos, sync_requirements, check_env_examples
```

## Key runtime behavior

- **v2 orchestration ledger (`DS`).** A `flowmesh/v2` submission compiles to a
  `PhysicalExecutionPlan` and runs through the durable orchestration engine
  (`src/server/orchestration/`), which owns semantic readiness: it turns settled
  records into ready work items over the acyclic plan and dispatches them through the
  `TaskRuntime`/dispatcher, so placement stays physical. Retries reuse the work item and
  its `invocation_id`; outputs publish idempotently to logical result slots. The snapshot
  persists at `workflow:{id}:ds` and `TaskRuntime.rehydrate` rebuilds it on restart; a v1
  submission keeps the static-DAG path.
- **Structured dynamic regions.** The engine executes the compiler's semi-static
  regions: control operators (`Branch`/`Merge`/`Spawn`/`Join`/`LoopContext`) settle
  in-ledger and never dispatch, while spawn children and loop iterations materialize
  incrementally by activation identity. A region closes on its child-init and loop-time
  capability account — sealed or revoked and drained — never on an observed-empty set.
  Every spawn site mints a monotonically attenuated `DelegatedAuthorityGrant`, and a
  denial records a durable `AuthorityDenied`/`PolicyDenied` that creates no child. An
  early join may release before full closure per its declared rule, with a residual
  policy governing children still running. Scope, loop, and activation budgets bound
  recursion. A spawn fans out to one child per element of its producer's result, and
  each child dispatches to a worker like any other task.
- **Cancellation.** A `flowmesh/v2` workflow cancels through the orchestration engine as
  a durable semantic event, so the ledger stays consistent with the task records and a
  cancelled workflow survives a restart without re-admitting cancelled work.
- **Physical episode lowering.** The compiler lowers a v2 template either transparently
  (one physical node per operator, the compatibility baseline) or into run-to-yield
  **episodes**: each node is annotated with the boundary that closes it (service issue,
  effect, durable checkpoint, continuation, region-blocking) and a chain of
  pure deterministic local leaves fuses into one episode. The two lowerings are
  contract-equivalent — an episode cut changes only where work yields, never a declared
  output, effect visibility, or progress closure. `ORCHESTRATOR_EPISODE_LOWERING=true`
  selects the episode-cut lowering.
- **Live-feasibility handoff.** A ready episode carries the lowerer's declared
  alternative; a feasibility check lets the scheduler defer an infeasible alternative,
  holding no worker, rather than dispatching it. It resolves no resident capacity.
- **Agent-harness substrate.** The logical `Agent` runs as a dispatchable run-to-yield
  episode that also owns a scope for its children, driven through a generic
  `HarnessAdapter` contract (`src/shared/harness/`). The engine validates each boundary
  the episode emits — a model or tool invocation, an effect, or a `spawn_agent` — against
  the operator's declared signature and authority before creating work; an undeclared
  request settles as a durable denial instead of running, and a re-driven boundary is
  deduplicated by its fabric-assigned idempotency key rather than repeating its effect. A
  `spawn_agent` creates one child activation, closed by a `SpawnSeal` or the agent's
  completion.
- **Agent-episode dispatch seam.** Every agent dispatches to the `AgentEpisodeExecutor`,
  which runs one adapter step per dispatch behind its resolved backend key (the built-in
  `scripted` backend or the `codex` app-server binding). A step resumes the agent's durable
  context and returns a completion, failure, cancellation, yield, or a typed boundary
  request; the server routes a boundary into the ledger and either re-dispatches the agent
  or suspends it until a durable outcome arrives, and a restart resumes with the same
  context. An agent's harness and managed-model binding are resolved at submission and
  pinned on its compiled operator; the backend comes from `spec.harness.backend` or the
  `AGENT_HARNESS_DEFAULT_BACKEND` default, and an agent with neither fails template
  validation. `agent` is a v2-only task type: a legacy v1 agent submission is rejected.
- **Agent-model gateway.** A managed model request an agent defers becomes a durable
  invocation the agent-model gateway settles off the agent's lane, injecting the result
  back at the originating call.
- **Resident-capacity control.** A `resident` model binding is served from reusable
  physical capacity rather than an external endpoint. Two control-plane actors — an
  Admission controller and a Lifecycle & scale manager — over durable control-state
  stores admit an invocation to a compatible model-serving replica, materializing one
  from zero on demand under policy. `ServiceClaim` facts are the sole credit authority;
  a credit releases only from a fenced ledger terminal consumed by `invocation_id`. When
  the network plane is also enabled, the invocation is carried data-direct over the fabric
  path: a per-replica claim-gated sidecar fronts the engine, and delivery is two-phase and
  server-driven — a bootstrap poke over the origin node's deputy obtains the engine
  acknowledgement, the Admission controller records `ACCEPTED` and issues the immutable
  `RouteAuthorization`, then a stream poke carries the authorized response over the
  data-direct deputy-to-sidecar channel while the server never carries the bytes; a lost
  or ambiguous delivery is `UNCERTAIN`, holds the credit, and re-drives, releasing only on a
  definite outcome, and a cancellation reaps both ends of the channel. Otherwise an in-server
  adapter relays the request as the claim-gated compatibility path. Enable with
  `RESIDENT_CAPACITY_ENABLED=true`. See [`RESIDENT_CAPACITY.md`](RESIDENT_CAPACITY.md).
- **Network-plane route substrate.** A topology-aware, control-resolved routing substrate
  turns trusted node endpoint advertisements and directional reachability evidence into an
  ordered route resolved by a pure resolver, carried by an origin-side deputy that never
  peer-discovers over the universal reverse-rendezvous `control_relay` — both ends attach
  outward to a root bridge, so neither needs an inbound connection — or a verified
  forward-dial `worker_direct` / `node_relay` offload for a reachable pair. The
  substrate holds no admission authority — it mints no `ServiceClaim` or `RouteAuthorization`
  and its transports carry only what a caller frames over them; resident-capacity control
  binds it to carry claim-gated resident invocation traffic. Enable with
  `NETWORK_PLANE_ENABLED=true`. See [`NETWORK_PLANE.md`](NETWORK_PLANE.md).
- **Task merging.** Compatible adjacent tasks in a DAG (same `taskType`,
  model, hardware shape, and merge key) coalesce into a single dispatch.
  Merged children ride on `WorkerTaskMessage.merged_children`; the worker
  writes per-child results into `result.children`; the dispatcher fans
  out synthetic `TASK_SUCCEEDED` / `TASK_FAILED` events. Disable with
  `ENABLE_TASK_MERGE=false`.
- **Stage stickiness** (`ENABLE_STAGE_WEIGHT_STICKINESS=true`) — the
  dispatcher pins stages that reference an upstream stage's checkpoint
  to the worker that produced it, falling back to normal selection when
  unavailable or stale. Mostly relevant for training pipelines reusing
  on-disk checkpoints.
- **Context reuse.** Workers report cached models/datasets in their
  `WorkerHardware`. The dispatcher's `_cached_worker_candidates` filters
  to workers whose cache covers the task's references; entries older
  than `WORKER_CACHE_TTL_SEC` are ignored.
- **Worker capabilities.** Beyond hardware fit, each worker advertises the set
  of task types it can service, and the dispatcher routes a task only to workers
  that advertise its type. A worker advertises a type only when its executor came
  up — e.g. SSH requires a reachable Docker daemon, and training or omni types
  require their (often GPU-only) dependencies — so a worker missing that executor
  isn't a candidate, rather than being handed a task it would fail.
- **Cursor pagination.** List endpoints accept `limit` and `before` /
  `after` cursors. The cursor is an opaque base64 of `(timestamp, id)`;
  do not parse client-side.
- **Redis channels.** The runtime uses three namespaces:
  - `flowmesh:control:*` — control plane (task assignments,
    cancellations, worker lifecycle).
  - `flowmesh:telemetry:*` — telemetry (heartbeats, status updates).
  - `flowmesh:logs:task:{task_id}` and
    `flowmesh:logs:workflow:{wfl_id}` — log streams, bounded by
    `LOG_STREAM_MAXLEN_TASK` / `LOG_STREAM_MAXLEN_WORKFLOW` and
    expired `LOG_STREAM_TTL_SEC` after close.

## Service restarts

Any Compose service can be recreated in place with `flowmesh stack restart
[SERVICE ...]`, without a full teardown. The root server survives its own
restart without losing in-flight work: scheduling state is persisted to Redis
and rebuilt on startup (`TaskRuntime.rehydrate`), and task events replay from a
durable stream. Rolling a new image across the cluster one node at a time is one
application. See [`SERVICE_RESTARTS.md`](SERVICE_RESTARTS.md).

## Plugin extension points

Server extension points are loaded via the `FLOWMESH_PLUGINS` env var.
Full contract, loader semantics, and a worked example live in
[`docs/PLUGINS.md`](PLUGINS.md).
