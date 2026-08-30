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
- `params` — non-secret backend configuration. The `codex` backend reads `base_url` and
  `model` (the Responses gateway it drives) and an optional `codex_home` override (the
  rollout directory, by default isolated per workflow and agent under the results dir). A
  credential-bearing param is rejected; a model credential goes in `model_binding.api_key`.

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

A mediated model request the agent defers is settled by the agent-model gateway against that
workflow's own binding. The deployment holds no model key: the credential is the workflow's
inline `api_key`, vaulted server-side at submission so only a reference is stored and used on
the server-to-upstream path — the raw key never persists in the source, template, ledger, or
logs. A credential embedded in a `url` or a harness param is rejected. The
`AGENT_MODEL_GATEWAY_*` defaults are in [`ENV.md`](ENV.md).

Vaulted credentials live in the Redis control store's trust boundary (ACL, auth, TLS) and are
not encrypted at rest, so the deployment operator owns that at-rest boundary.

## Fabric-served tools

Beyond the model and `spawn_agent`, an agent may declare a fabric-served tool it invokes
through a gateway-injected facade. The agent lists it under `spec.v2.tools` (`{name,
interface}`) and grants the interface in `authority.invoke`; the compiler pins the facade
and the gateway injects only an agent's declared facades. The model's facade call is
captured server-side as an `invocation` boundary and executed off the agent's lane by the
`FabricToolBroker`, which returns a typed outcome (success with results and provenance, or
timeout / quota / unavailable) injected back at the call.

`web_search` (`interface: search/v1`) is the built-in fabric tool. Its provider is keyless
DuckDuckGo by default, or a keyed provider via `WEB_SEARCH_*` ([`ENV.md`](ENV.md)); results
are snippets only. A child region must declare the interface in its authority ceiling for a
spawned child to invoke it — a child that declares a tool the region omits is a compile
error.
