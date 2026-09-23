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
leases. `msk-` is an unguessable ref for a workflow's vaulted model credential, `hnd-`
an unguessable claim-bound admission handoff token, `chg-` an unguessable
cache-to-cache content-hydration grant, and `csg-` an unguessable content-store
access grant. Activation-private state adds
`aps-` state references, `sbm-` sealed-generation manifests, and `psa-` attachments.
The network plane adds `rog-` route
origins and `rly-` relay sessions. Worker-originated mediated boundaries add `mop-`
one-use mediated-operation permits.
Always use `new_*_id()`
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
    tools/                Fabric-served external-tool control authority (broker)
    utils/                concurrent, helpers, logging, misc, time
  shared/
    grpc/supervisor/v1/   Generated proto stubs (server + worker)
    schemas/              Cross-cutting schemas
    tasks/                Workflow/task spec models
    tools/                External-tool contract and search backends
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
  cancelled workflow survives a restart without re-admitting cancelled work. A worker
  success that races the cancel — an agent-episode step or input preparation the worker
  had already finished — settles the cancellation in place rather than re-admitting the
  task, and an episode suspended on a mediated boundary settles at the cancel that reaps
  it.
- **Physical episode lowering.** The compiler lowers a v2 template either transparently
  (one physical node per operator, the compatibility baseline) or into run-to-yield
  **episodes**: each node is annotated with the boundary that closes it (service issue,
  effect, durable checkpoint, continuation, region-blocking) and a chain of
  pure deterministic local leaves fuses into one episode. The two lowerings are
  contract-equivalent — an episode cut changes only where work yields, never a declared
  output, effect visibility, or progress closure. `ORCHESTRATOR_EPISODE_LOWERING=true`
  selects the episode-cut lowering.
- **Advisory lowering policy.** A deployment selects a compile-time policy at each
  lowering hook, refining a lowering the compiler has already found legal: it can keep a
  fusible operator out of its predecessor's episode, pick among the families compatible
  with a dependency's engine-batch key, and express warmth, reuse, affinity, or
  preemption preference. The hooks are selected independently, so a deployment composes
  the facets it wants. The compiler screens every answer, so fusion stays bounded to the
  pure, deterministic, local set, an episode only ever cuts more often, and a refinement
  holds the dependency's engine-batch key and isolation as well as its pinned family and
  requiredness. Choosing a worker, reserving capacity, minting a claim or attachment,
  and replacing a pinned resident binding belong to the fabric. A policy is
  deployment-global — a workflow submission selects none — and every hook defaults to a
  conservative policy that lowers identically to the compiler alone, so a workflow's
  declared outputs, effects, and recovery are the same whichever policies run. The
  fusion refinement applies to the episode-cut lowering; family and residency refinement
  applies to both lowerings. A residency refinement reaches an ordinary resident
  dependency: a `serve` node declares its own standing residency and an unresolved
  embodiment menu carries its intent per candidate, so neither consults the policy. A
  plan records the episode strategy and the policy effective at each hook, and the v2
  dry-run inspection compiles under the same configured set, so a validation report and
  a persisted submission describe the same physical decision. Select policies with
  `ORCHESTRATOR_FUSION_POLICY`, `ORCHESTRATOR_RESIDENCY_POLICY`, and
  `ORCHESTRATOR_SERVICE_FAMILY_POLICY`.
- **Inference-embodiment menus.** An inference leaf declares one model contract; whether
  resident capacity or a local executor serves it is the fabric's to decide. Such a leaf
  lowers to one physical node carrying the embodiments that contract admits — a
  resident-served candidate and a self-contained one — each with its own `EpisodeSpec`
  and either a local executor and resource envelope or a `ServiceFamilyRequirement` and a
  *conditional* `ResidencyIntent`. The compiler emits a menu only for leaves it proves run
  one declared contract, built from one effective engine request; any other leaf keeps the
  embodiment its source names, and `mode: resident` pins resident serving. An unresolved
  menu registers no residency demand, mints no claim, and is never read as one episode. At
  dispatch a scheduler-owned `EmbodimentSelector` reads a read-only feasibility snapshot
  and returns a selection or a defer, holding no capacity authority. The selection records
  to the ledger before the task is published and lands on the attempt, and the dispatcher
  materializes the task it implies. A leaf declaring several prompts is one batch,
  whether its embodiment is chosen from a menu or pinned to resident serving: it carries
  every conversation on one boundary under one `ServiceClaim` and one `invocation_id`,
  whose credit reserves an admission slot per conversation, and the replica issues each
  as its own concurrent engine request so the engine's continuous batching combines
  them. A batch past a replica's admission bound is a structural resident infeasibility
  the menu answers with its self-contained embodiment; a pinned leaf admits no other
  embodiment, so it fails at admission instead. A leaf whose request the compiler cannot
  project runs one prompt per invocation and is refused at submission when it declares
  more. See [`EXECUTORS.md`](EXECUTORS.md).
- **Upstream-resolved inference inputs.** An inference leaf names its prompts through one
  `CanonicalInferenceInputSource`: literal items, or one bounded projection of a declared
  direct upstream input, written as `data.expr` or `data.node` plus `data.path` and
  normalized to one descriptor. Compilation proves and fingerprints the descriptor and its
  declared envelope without reading a future upstream value, and candidate feasibility is
  screened against that envelope rather than an exact count. The origin worker resolves the
  source once against the upstream snapshot pinned to its task and records an
  `InputResolutionBinding` — source and resolver digests, ordered upstream content
  provenance, and the resolved request's digest and cardinality — before either embodiment
  reaches a model. Both embodiments then run that one request, so a replica receives
  prompts rather than a projection to interpret, and the raw upstream values never leave
  the origin worker. A resident claim reserves an admission slot per conversation the
  resolution materialized. A source that is missing, unprojectable, not a non-empty list of
  strings, or past its envelope fails the task before any model I/O and before admission,
  creating no claim and switching no embodiment. A re-driven task runs only when it
  re-resolves to its recorded binding.
- **Optional-envelope input preparation.** An upstream inference source may leave its item
  envelope (`max_items`) undeclared. Such a leaf is prepared before an embodiment is chosen
  for it: one bounded dispatch resolves the pinned source once on a worker and records the
  exact request it produces, so selection and resident admission are sized from the actual
  conversation count rather than a declared bound. The recorded request is immutable — a
  re-driven task hydrates and verifies it rather than resolving the source again, and a
  missing or mismatched request fails closed. The preparation reserves no capacity and pins
  no embodiment. `ORCHESTRATOR_MAX_PREPARED_INPUT_BYTES` optionally caps a prepared
  request's size; unset, it is uncapped.
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
- **Activation-private state.** An agent activation owns its mutable harness state
  under an opaque `ActivationPrivateStateReference`, whose lifecycle and recovery the
  orchestration ledger owns. A `PrivateStateBinding` names the private-state generation
  and recovery mode a continuation resumes on, an immutable `StateBundleManifest`
  describes one sealed private-state generation over registered typed components
  (`harness_home_fs`, `workspace_fs`), and a `PrivateStateAttachment` grants one worker
  incarnation the exclusive, write-epoch-fenced authority to materialize and write it. An
  attachment is physical execution authority over one activation's state, distinct from
  the capacity admission a `ServiceClaim` carries.
  Each dispatch mints a fresh write epoch, so a superseded holder can neither write nor
  seal. Components seal together at one quiescence fence, so a harness home never
  resumes beside a workspace from another private-state generation. While a generation is
  sealed local to the holder that produced it, that holder is a hard scheduler
  feasibility constraint resolved at dispatch: the episode lane yields as any other does,
  and an episode waits while its holder is busy. Owner loss, a worker-incarnation change,
  or a component that does not match its seal fails closed as a typed
  `PrivateStateUnavailable` rather than resuming against a fresh or partial home. One
  activation reaches another's state only by holding a valid binding and attachment for
  it, which the ledger's owner and write-epoch fences decide; the `0700` private root,
  keyed by the opaque reference, separates a holder's lineages from other users on its
  node,
  and a harness works inside its own components under its sandbox. Only opaque
  references cross into the ledger, control state, operation frames, logs, results, or
  artifacts. These fences are the activation's own — a private-state generation, its
  write epoch, and the ledger's owner fence — and are separate from the replica
  incarnation, listener generation, and admission epoch that fence a resident
  allocation. `WORKER_PRIVATE_STATE_DIR` sets the root.
- **Agent-local sandbox execution.** An agent that declares the `sandbox.execute`
  interface runs commands worker-locally in its own `workspace_fs`, on the worker its
  private-state attachment selected; an ordinary command makes no control-plane round
  trip or cross-worker hop. The runtime confines a command to its workspace, denies
  network egress, and bounds its resources; commands become durable together at the
  agent's ordinary boundary seal, and a loss before the seal fails closed as
  `PrivateStateUnavailable`. A child runs commands only where its parent delegated the
  interface. The feature is a single-trusted-tenant development posture, not
  multi-tenant isolation; an operator enables it for the fleet with
  `AGENT_SANDBOX_ENABLED`.
- **Author-owned sandbox egress.** An agent that declares the distinct `sandbox.egress`
  authority runs its commands with the network fence relaxed, where the deployment sets
  `AGENT_SANDBOX_EGRESS_ENABLED`; a ceiling that carries the interface only to delegate
  it opts its own commands back out with `sandbox.network_egress: deny`, and a child
  never inherits an egress its parent withheld. An egress command is an external effect:
  a failure before the next seal may repeat a command that already egressed, so the
  author owns idempotency and reconciliation.
- **Agent-model gateway.** A model boundary an agent defers with a `canned` or `echo`
  binding settles on the control plane off the agent's lane, injecting the result back at
  the originating call. The gateway resolves the activation's pinned binding and its
  vaulted credential; a `resident` binding admits through resident-capacity control, and
  an external (`openai`) binding egresses on the agent's own worker.
- **Harness egress-handoff modes.** A backend declares how it hands a mediated egress
  boundary to the worker egress lane. A `durable_pre_egress_yield` backend (the
  `scripted` binding) releases its episode lane at the boundary and resumes from the
  committed outcome: the worker captures the request, yields only its digest, and the
  boundary settles through the worker-originated path. A `synchronous_turn_only` backend
  (the `codex` binding) holds its own lane through one bounded same-worker egress within
  a turn, under a no-conflicting-capacity, deadline, and cancellation bound; the turn's
  durable anchors are its turn-completion boundaries, and recovery re-runs the whole turn
  from the last completion under a fresh permit.
- **External-model egress.** A managed external (`openai`) model turn egresses on the
  Agent's own worker through a worker-local Responses facade, bound to loopback and
  authenticated per episode so one episode drives only its own egress. Codex's model
  provider targets the facade: the facade translates each turn between the Responses wire
  and Chat Completions, injects the agent's pinned fabric facades, and runs the held
  egress — it proposes the request digest to control, awaits the one-use
  `MediatedOperationPermit` over the worker's attachment, and egresses synchronously
  through the `MediatedEgressSidecar`, returning the model's whole message inline. The
  per-workflow model credential rides the permit to the worker; a worker without one uses
  its deployment-global key. A fabric facade the model calls on the turn is captured into
  a `FacadeTurnGroup` reported to control, which records the group so the episode's next
  completion routes its members and the turn returns Codex a clean summary. The
  credential is kept out of the ledger, the control stores, and the logs.
- **Resident-capacity control.** A `resident` model binding is served from reusable
  physical capacity rather than an external endpoint. Two control-plane actors — an
  Admission controller and a Lifecycle & scale manager — over durable control-state
  stores admit an invocation to a compatible model-serving replica, materializing one
  from zero on demand under policy. `ServiceClaim` facts are the sole credit authority;
  a credit releases only from a fenced ledger terminal consumed by `invocation_id`. The
  invocation runs in the workers over the network plane (required), so control never
  constructs, parses, or carries engine traffic: the consuming episode's own worker — an
  agent or an inference/embedding leaf bound to a resident service — captures the
  boundary and holds the raw request worker-private, control admits the claim, binds the
  replica's claim-gated sidecar, and relays the claim-bound handoff to the origin worker,
  which carries the request to the replica worker over the reverse-rendezvous relay and
  drives the two-phase protocol. The origin worker reports the engine acknowledgement, at
  which the Admission controller records `ACCEPTED` and mints the immutable
  `RouteAuthorization` control relays back; the origin worker then streams the authorized
  response, materializes the completion into the content store, and reports the fenced
  outcome manifest. Root and supervisors relay opaque frames without decoding a body,
  cursor, or window; a lost or ambiguous delivery is `UNCERTAIN`, holds the credit, and
  re-drives from the materialized manifest, releasing only on the fenced terminal, and a
  cancellation reaps both ends. Enable with `RESIDENT_CAPACITY_ENABLED=true` (which
  requires `NETWORK_PLANE_ENABLED=true`). See [`RESIDENT_CAPACITY.md`](RESIDENT_CAPACITY.md).
- **Unified task-ID-gated resident serve surface.** Every public user-declared `serve`
  task is a resident-gated standing allocation reached only by its task ID, over one
  FlowMesh-authenticated, claim-gated endpoint
  (`/api/v1/serve/tasks/{task_id}/{upstream_path}`). The request relays to the task's
  standing replica unchanged and the engine's own response comes back unchanged, so an
  OpenAI-compatible client can drive any endpoint the engine serves. A serve task pins one
  gated exposure mode — `proxy`, the default root-local ingress reached at the task-qualified
  route above, or `forward`, a per-task public port on the root's public host reached at
  `http://<public_host>:<forward_port>/` with the engine's own paths. The root binds the
  port, authenticates and admits the request over the
  same gate as `proxy`, and relays it to the task's standing replica; a mode with no live
  exposure fails closed. At start the task is adopted as its own standing replica,
  validated under `RESIDENT_ALLOWED_MODELS`. Both modes carry traffic over `control_relay`;
  trusted direct target legs resolve behind the shared carriage seam. Available
  when `RESIDENT_CAPACITY_ENABLED=true` (which requires `NETWORK_PLANE_ENABLED=true`). See
  [`RESIDENT_CAPACITY.md`](RESIDENT_CAPACITY.md).
- **Network-plane route substrate.** A topology-aware, control-resolved routing substrate
  turns trusted node endpoint advertisements and directional reachability evidence into an
  ordered route resolved by a pure resolver, carried by an origin-side deputy that never
  peer-discovers over the universal reverse-rendezvous `control_relay` — both ends attach
  outward to a root bridge, so neither needs an inbound connection — or a verified
  forward-dial `worker_direct` / `node_relay` peer transport for a reachable pair. The
  substrate holds no admission authority — it mints no `ServiceClaim` or `RouteAuthorization`
  and its transports carry only what a caller frames over them; resident-capacity control
  binds it to carry claim-gated resident invocation traffic. Enable with
  `NETWORK_PLANE_ENABLED=true`. See [`NETWORK_PLANE.md`](NETWORK_PLANE.md).
- **Trusted peer transports.** Where a deployment declares the origin-to-target pair
  trusted, an admitted resident invocation leaves the reverse-rendezvous relay for a
  direct socket the route's own origin opens: `worker_direct` reaches the selected
  worker's claim-gated replica-sidecar listener, `node_relay` reaches the target node's
  purpose-scoped listener, which hands the session to its local sidecar uplink. The
  `RouteOrigin` is both the source identity and the dialer, so a workflow boundary's
  payload bypasses the root and the rendezvous for the whole request and response, while
  a root-sourced gated serve call has the root as its legitimate origin. Both carry the
  same frames, fences, windows, and cancellation as the relay, and the target sidecar's
  claim gate is the only authority over the traffic. Mutual TLS between the pair is the
  default. An untrusted, unreachable, or policy-ineligible pair is carried over
  `control_relay`, and a dial that fails before delivery falls back to it under the same
  held credit. Enable with `NETWORK_PLANE_PEER_ENABLED=true`. See
  [`NETWORK_PLANE.md`](NETWORK_PLANE.md).
- **Worker-originated mediated boundaries.** A fabric-served external tool (`search/v1`)
  or a managed external model turn egresses only in the Agent's assigned worker, never in
  the root or a supervisor. The worker captures the boundary, keeps the raw request in
  worker-private
  state, and yields the lane carrying only a canonical request digest — no raw arguments
  cross to the control plane. Central control mints a one-use, audience-bound
  `MediatedOperationPermit` and relays it to that worker as an ordinary control message on
  its authenticated attachment, never a dispatched task. The worker's
  `MediatedEgressSidecar` — a bounded worker-local egress lane, not a task, replica,
  endpoint, or authority — validates the permit fence and request digest, reads the
  provider credential only from its local environment, egresses, and reports a
  permit-fenced outcome that settles the boundary before the episode resumes. It retains
  the request non-destructively until the committed outcome is acknowledged. A fence
  rejection is a declared terminal boundary failure, never a retryable provider response;
  a lost outcome holds the boundary pending for a same-`idm-*` re-drive. The
  `FabricToolBroker` applies the tool's policy and correlation. See
  [`EXECUTORS.md`](EXECUTORS.md).
- **Reference-backed invocation outcomes.** A mediated boundary settles by reference: the
  producing worker materializes its result into the content-addressed `FabricContentStore`
  and reports a bounded `OutcomeManifest`, never the payload. The manifest commits to the
  ledger before the continuation re-readies or a linked `ServiceClaim` credit releases, and a
  resumed worker hydrates and digest-verifies the reference before injection. Root and
  supervisors relay opaque frames and hold only the manifest. Materialization is idempotent
  under `idm-*`. Outcome finalization and a prepared inference request are stored over one
  content-addressed object core under separate facades, so neither becomes a name for the
  other. The mediated-egress-sidecar tool path and the worker-materialized resident
  completion settle by reference; the model gateway settles inline. See
  [`EXECUTORS.md`](EXECUTORS.md).
- **Content references.** Every value the fabric stores immutably is named by one
  `ContentReference`: the authorization scope isolating it together with the digest and
  size a reader verifies it by, and nothing that says where it is or what it means. The
  control plane assigns the scope — a task's dispatch and a minted permit each carry the
  one their work writes under — so a worker carries a scope rather than asserting one. An
  outcome finalization, a prepared inference request, a task result, and any later
  consumer each keep their own binding to a reference, so an object is never a name for
  what a consumer calls it, and identical bytes in two scopes are two objects.
- **The shared content store.** Every content object lives in one shared durable store —
  an S3-compatible service such as the MinIO a default deployment co-locates on the root
  node, cloud S3, or a filesystem every node mounts — reached through the same
  `FabricObjectStore` contract and selected with `CONTENT_STORE_BACKEND`. A deployment
  that names no store runs the co-located one and points at it, so a fresh cluster stores
  content without being configured; naming `CONTENT_STORE_ENDPOINT_URL` moves the fabric
  onto real object storage and leaves the co-located store unstarted, which is the shape
  a production deployment takes. The root provisions the bucket it is pointed at where
  its credential allows, since the scoped session a worker reaches content under covers
  one scope's prefix rather than the bucket. It is a service
  beside the fabric, never the root process: the root and its supervisors hold no
  payload. A worker writes an object there before it reports the reference naming it, so
  a reference that reaches any binding names bytes that already outlive their producer,
  and a worker's death loses nothing. The outcome-finalization index lives on the control
  plane, binding an `idm-*` to a reference so a re-drive re-reports the first
  materialization rather than re-running a sampled producer; the store holds only bytes
  and never treats an idempotency key as a name. The scope that binding lands in is the
  one control assigned the work when it authorized the key, so the producer reporting a
  finalization is held to it rather than naming a scope of its own.
- **Store access.** A worker reaches the store only under a `csg-`
  `ContentStoreAccessGrant` the control plane mints for one dispatched task in one
  authorization scope, bound to the worker incarnation running it and expiring shortly
  after. The grant records what access was given and is not itself secret; the material
  that opens the backend travels beside it, relayed to that worker as a control message
  and delivered over its authenticated attachment, and is kept nowhere — no ledger,
  control store, manifest, frame, or log. A scope is the widest a task can reach, cut as a
  short-lived session over that scope's prefix, and the grant carries no list, delete, or
  binding operation — which references a task may use is still decided by the consumer
  bindings control checks. A fresh dispatch or recovery gets fresh access; expiry or a
  policy rotation fences what came before.
- **Worker content cache and granted hydration.** What a worker holds is a cache over that
  store, so a copy may be dropped at any time: a deployment can bound the cache by how
  long a copy goes unused and by disk, least recently used first, and leaves both
  unbounded by default. A read tries the local copy, then another worker's copy, then the
  store itself. For a peer's copy the control plane checks that the requesting worker is
  running the task and that the task is already bound to exactly that reference, resolves
  a live holder, and mints one short-lived `chg-` `ContentHydrationGrant` it hands to both
  ends: the holder serves only a grant it was handed, once, for that exact object, and the
  requester verifies the digest and size before anything reads the bytes. That transfer
  runs over the network plane's relay under its own namespace, so the root bridges opaque
  frames and never holds, assembles, or resolves the payload. A refused, expired, or
  replayed grant, an evicted copy, or a holder that is gone costs a read from the shared
  store rather than a failure. Enable the cache and its transfers with
  `CONTENT_HYDRATION_ENABLED=true` (which requires `NETWORK_PLANE_ENABLED=true`).
- **Task results.** A task's result lives in the shared content store: its worker
  writes the result envelope there under the task's store access before it reports
  success, and the success binds that reference once, at the commit that settles the
  task, so a retry, relocation, or duplicate success converges on the result already
  bound. A v2 task binds it into the ledger — its induced output slot, or for a spawned
  child or later loop iteration the value its work item settled with — and a v1 task onto
  its record. Every result the control plane reads — the result and bundle routes, stage
  references, conditions, fan-out, and an agent's accepted inputs — resolves that binding
  and reads the verified envelope from the store, holding no copy of its own; the results
  directory keeps only a task's logs and artifacts. A task that outlives its store access
  has it renewed while it still runs on the worker asking.
- **Task merging.** Ready v1 inference tasks of one org whose specs differ only in
  their inputs (`data` and `system_prompt`) coalesce into one dispatch, and each
  inference executor returns every merged child's own result. Merged children ride
  on `WorkerTaskMessage.merged_children` and come back in `result.children`. A child
  whose rendered spec differs from its parent's beyond its inputs leaves the merge
  and runs alone. A child the dispatch returns no result for, or whose parent is
  cancelled, returns to the queue and runs alone without spending an attempt, and a
  merged dispatch that fails or loses its worker returns its parent and every child
  still in it the same way. A batch the dispatcher releases before sending it keeps
  its children mergeable.
  Disable with `ENABLE_TASK_MERGE=false`.
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
- **Cluster telemetry.** A workflow emits one OpenTelemetry trace spanning the processes
  that act on it — the root server's control plane and each worker that runs a task,
  with a supervisor relaying frames it never decodes and so never records — with a
  `trace_id` every producer derives from the `workflow_id` by the same pure function, so
  they agree with no coordination and a trace survives a restart. Spans carry ids only,
  and telemetry is observation: nothing it records is read by admission, credit release,
  embodiment selection, dispatch or recovery, and it writes nothing to the orchestration
  ledger. Off by default; enable with `SERVER_METRICS_TELEMETRY_LEVEL`. See
  [`TELEMETRY.md`](TELEMETRY.md).
- **Workflow completion.** A workflow closes once, through one serialized finalizer:
  its log stream is sealed and its `flowmesh.workflow` span emitted when every task has
  settled. The span's end is the last durable finish among its tasks, so a workflow that
  closes again after a restart closes the same way it did the first time.
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
