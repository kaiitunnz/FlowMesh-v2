# Task types and executor registry

The worker resolves `spec.taskType` against an executor registry in
`src/worker/runner.py`. Built-in executors:

| `taskType` | Executor | Use case |
|-----------|----------|----------|
| `echo` | `EchoExecutor` | Echo input back as result (smoke tests) |
| `inference` | `VLLMExecutor` / `TransformersExecutor` | LLM inference |
| `embedding` | `VLLMEmbeddingExecutor` (text, when `model.vllm` is set) / `TransformersExecutor` (visual, `model.transformers.mode: visual-embedding`) | Text / visual embeddings |
| `diffusion` | `DiffusersExecutor` | Image / video diffusion models |
| `omni_text2{audio,image,speech,general}` | `Omni*Executor` | Multimodal generation |
| `training` | `SFTExecutor` / `LoRASFTExecutor` / `DPOExecutor` / `PPOExecutor` | LLM fine-tuning |
| `image_classification_training` | `ImageClassificationTrainingExecutor` | Vision classification fine-tuning (`AutoModelForImageClassification` + HF `Trainer`) |
| `rag` | `RAGExecutor` | Retrieval-augmented generation |
| `agent` | `AgentEpisodeExecutor` | Tool-using LLM agent run as a run-to-yield harness episode; requires a resolved harness binding (`spec.harness` or a deployment default) or fails validation |
| `data_profiling` | `DataProfilingExecutor` | DataFrame profiling |
| `data_retrieval` | `DataRetrievalExecutor` | DataFrame loading from sources (`type: sql`, `type: s3`, `type: lumid` with `mode: sql\|s3\|agent` via lumid-data-app; `type: lumid` (mode `sql`/`s3`/`agent`) requires `lumid_data_token`, the bearer forwarded to lumid-data-app) |
| `ssh` | `SSHExecutor` | Interactive SSH session or non-interactive container job |
| `serve` | `VLLMServeExecutor` | Persistent vLLM API server for a single model |
| `dev_model` | `DevModelExecutor` | GPU-free OpenAI-compatible endpoint; forwards to an upstream or returns canned responses |

Helper utilities live in `src/worker/executors/utils/` (`artifacts`,
`checkpoints`, `data_utils`, `distributed`, `graph_templates`,
`huggingface`, `safe_eval`). Cross-cutting behavior is in
`src/worker/executors/mixins/` (`data`, `governance`, `inference`,
`training`).

## Result schema

Every executor's `run()` returns an exact per-task-type subclass of
`BaseExecutorResult`, all defined in the shared `src/shared/schemas/result`
package. The base class carries two cross-cutting fields:

- `children: dict[str, BaseExecutorResult]` — per-child results when
  merged tasks share a dispatch.
- `artifacts: ArtifactContext | None` (wire key `_artifacts`) —
  resolution context for relative artifact refs.

Each subclass declares its exact fields (typed nested payloads) and tags itself
with a `task_type` discriminator — e.g. `InferenceResult`, `LoRAResult`,
`AgentResult`, `SSHResult`. The `AnyExecutorResult` discriminated union in
the same package deserializes a `results.json` back into its exact subclass
end-to-end (worker envelope → server ingest and `GET /results/{id}` → SDK).
Results without a `task_type` (legacy files, condition-skips) fall back to
the permissive base.

Artifact-bearing fields use `ArtifactRef` (`{"path": rel_path}`);
relative paths resolve against the producer's `_artifacts` context via
`artifact_to_source` / `_render_artifact_ref`.

## Agent-episode harness backends

Every agent runs through the dependency-light `AgentEpisodeExecutor`: one dispatch is one
run-to-yield step of the named `HarnessAdapter` binding, resumed from the durable capsule
and delivered outcomes the fabric ships. The executor advertises the `agent` capability on
any worker, so a CPU worker services agent episodes. A backend binding is imported lazily
from the worker adapter registry (`src/worker/executors/harness/`) only when its key is
selected.

Declared per agent under `spec.harness`:

- `backend` — the adapter binding: `scripted` (a deterministic backend that replays a
  declared step sequence from `params.script`) or `codex` (the version-pinned Codex
  app-server binding).
- `version` — pins the adapter/protocol so a capsule resumes only on a match.
- `params` — non-secret backend configuration. The `codex` backend takes an optional
  `codex_home` override (the rollout directory, by default isolated per workflow and agent
  under the results dir); its model upstream comes from `model_binding`, which it reaches
  through the worker-local Responses facade. A credential-bearing param is rejected; a
  model credential goes in `model_binding.api_key`.

## Per-workflow harness and model binding

An agent declares its harness and model binding in the workflow; both are pinned at
submission, so a retry or resume is unaffected by a later environment change.

`spec.harness.backend` (and optional `version`) selects the harness. A workflow that omits
it falls back to `AGENT_HARNESS_DEFAULT_BACKEND` / `AGENT_HARNESS_DEFAULT_VERSION`, and an
agent with neither fails validation.

`spec.model_binding` selects the managed model. Its modes are `canned` and `echo`
(deterministic, credential-free), `openai` (an external OpenAI-compatible upstream — `url`,
`model`, and the user's own inline `api_key`), and `resident` (a `service_model_ref` naming
a FlowMesh-served model, with no url or credential). Each field falls back to the
`AGENT_MODEL_GATEWAY_*` deployment default, then a `canned` default; an `openai` binding
requires a url and a `resident` binding requires a reference.

A model boundary the agent defers with a `canned` or `echo` binding settles on the control
plane, against that workflow's own binding; an `openai` binding egresses on the agent's own
worker (see [Managed external-model egress](#managed-external-model-egress)), and a
`resident` binding admits through resident-capacity control and runs in the workers over the
network plane (see [`RESIDENT_CAPACITY.md`](RESIDENT_CAPACITY.md)). The model credential is the
workflow's own inline `api_key`, vaulted server-side at submission so only a reference is
stored, resolved within its own workflow, and carried to the egressing worker on the one-use
permit — the raw key never persists in the source, template, ledger, or logs, and the
deployment key serves only as a fallback. A credential embedded in a `url` or a harness
param is rejected. The
`AGENT_MODEL_GATEWAY_*` defaults are in [`ENV.md`](ENV.md).

Vaulted credentials live in the Redis control store's trust boundary (ACL, auth, TLS) and are
not encrypted at rest, so the deployment operator owns that at-rest boundary.

## Fabric-served tools

Beyond the model and `spawn_agent`, an agent may declare a fabric-served tool it invokes
through a gateway-injected facade. The agent lists it under `spec.v2.tools` (`{name,
interface}`) and grants the interface in `authority.invoke`; only an agent's declared
facades are available to it.

`web_search` (`interface: search/v1`) is the built-in fabric tool. Its provider is keyless
DuckDuckGo by default, or a keyed provider via `WEB_SEARCH_*` ([`ENV.md`](ENV.md)); results
are snippets. An agent must declare `web_search` in its authority to use it, and a spawned
child must carry the interface in its child-region authority ceiling — an undeclared tool is
a compile error.

The control plane holds a search's authority and idempotency; the egress runs only in
the Agent's assigned worker. The agent's own worker
captures the `search/v1` boundary, records the raw request in worker-private state keyed
by its stable `(agent_task_id, call_correlation)` occurrence, and yields carrying only a
canonical request digest — the raw request never reaches the control plane. The engine
records the digest and mints a one-use `MediatedOperationPermit` (`mop-`) audience-bound
to that worker and its generation, and relays it to the worker as an ordinary control
message on the worker's authenticated attachment, never a dispatched task.

The worker's `MediatedEgressSidecar` — a bounded worker-local egress lane, not a task,
replica, endpoint, or authority — reads the request back from worker-private state,
validates the permit fence (audience, generation, interface, deadline, request digest)
and consumes the one-use permit, egresses through the local provider, and reports a
permit-fenced outcome the engine commits before the episode resumes. It reads its keyed
provider credential only from its own local worker environment (projected from
`WEB_SEARCH_*` through the supervisor worker-environment allowlist) — no credential
travels in a workflow, envelope, frame, message, or log. A fence rejection is a declared
terminal boundary failure, never a retryable provider response; a missing worker-private
request fails the boundary closed.

A successful provider result is materialized by reference: the sidecar writes the outcome
bytes to the content-addressed store under its `idm-*` and reports only an
`OutcomeManifest`; the result body never crosses the supervisor or root. A typed
control status (an unavailable provider) is a bounded inline datum. The request is
retained non-destructively until the engine acknowledges the committed outcome and reaps
custody; a same-`idm-*` re-drive finds the first materialization instead of re-sampling,
and a store-write or egress failure sends no outcome, holding the boundary pending for a
re-drive under the same `idm-*`. On resume the `AgentEpisodeExecutor` hydrates and
digest-verifies the manifest before injecting the value into the harness; a hydration
failure fails the step for a physical retry of the same reference, never a re-run. A
server restart re-mints the permit and re-relays it to the surviving worker, whose
in-memory request is intact; a genuine worker loss fails the boundary clean.

The `FabricToolBroker` applies a fabric tool's policy and correlation on the control
plane and terminalizes a server-captured boundary as an unavailable outcome.

## Harness egress-handoff modes

A backend declares how it hands a mediated egress boundary to the worker egress lane. A
`durable_pre_egress_yield` backend (`scripted`) releases its episode lane at the boundary
and resumes from the committed outcome: the worker captures the request, yields only its
digest, and the boundary settles through the worker-originated path above. A
`synchronous_turn_only` backend (`codex`) holds its own lane through one bounded
same-worker egress within a turn, under a no-conflicting-capacity, deadline, and
cancellation bound. The turn's durable anchors are its turn-completion boundaries;
recovery re-runs the whole turn from the last completion under a fresh permit, and a
re-run injects the same idempotency key so a settled effect never double-applies. A
backend advertises `durable_pre_egress_yield` only if it implements request-capsule
capture and outcome-reinjection recovery; the default is `synchronous_turn_only`.

## Managed external-model egress

An agent's managed external (`openai`) model turns egress on its own worker through a
worker-local Responses facade, bound to loopback and authenticated per episode so one
episode drives only its own egress. Codex's model provider targets the facade at
`/agent/{task_id}/v1/responses`, carrying the per-episode token the facade issued.

For each turn the facade translates the Responses request into a Chat Completions request,
injects the agent's pinned fabric facades, and runs the held egress: it proposes the
request digest to control, awaits the one-use `MediatedOperationPermit` over the worker's
attachment, and egresses synchronously through the `MediatedEgressSidecar`, returning the
model's whole message inline. The per-workflow model credential rides the permit to the worker; a worker
without one uses its deployment-global `AGENT_MODEL_API_KEY`. The `Authorization` header is
redacted in the facade's own logs, and the credential is kept out of the ledger, the
control stores, and the logs. A denial, a permit that never arrives within
`AGENT_MODEL_EGRESS_TIMEOUT_SEC`, or a fence rejection is a terminal turn failure.

A fabric facade the model calls on the turn is captured into a `FacadeTurnGroup`: each
search member carries the digest of its worker-private request and each spawn member
carries its args, ordered by emission with identities derived from the turn so a re-drive
recovers the same identities. The facade reports the group to control, which records it so
the episode's next completion routes the members, and returns Codex a clean summary in
place of the raw calls. A search member routes to the same worker egress by its digest; a
spawn member admits a child region.

## Resident service-backed leaves

An inference or embedding leaf declares a resident service binding with `spec.service`
(`{mode: resident}`, optionally `service_model_ref` and `isolation`) to consume a
FlowMesh-served model from resident capacity instead of loading one in the worker. The
binding normalizes to the same service dependency an Agent's resident model binding uses,
so both pin one plan-derived service-family requirement and a required residency intent,
and both raise the same control-admitted `ServiceClaim`. A resident inference leaf carries
no worker-local GPU requirement, since its model runs on the replica.

The leaf runs through the `ServiceLeafExecutor` as a run-to-yield episode with no harness:
its first step builds the model request from `spec.inference`/`spec.data` — a chat leaf
builds an explicit `messages` array or a prompt, an embedding leaf the `input` list of
texts — keeps it in worker-private resident custody, and yields one resident model boundary
carrying only its digest. The fabric admits the claim and drives the worker-originated
resident protocol to the replica exactly as for an Agent resident call; the replica reaches
its co-located engine on the route its family's interface selects (`/chat/completions` or
`/embeddings`), and the settled outcome — the completion text or the embedding vectors as
JSON — is injected on a resume and becomes the leaf's result. The resident-request capture
and reference-backed outcome hydration are the same caller-neutral substrate the
agent-episode executor uses. The leaf invocation never routes through the Agent model
gateway.

A service dependency's family folds the service interface, base model, and isolation
domain, and an adapter rides the admission profile's adapter slot. A shared base model and
interface reuse a warm replica; a differing interface, base model, adapter, or isolation
domain resolves to a distinct family and cannot share a batch or route on a matching model
name alone.
