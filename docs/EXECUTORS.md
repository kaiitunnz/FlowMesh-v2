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
| `agent` | `AgentExecutor` / `AgentEpisodeExecutor` | Tool-using LLM agent — UTU / youtu-agent by default, or a run-to-yield harness episode when `spec.harness` names a backend |
| `data_profiling` | `DataProfilingExecutor` | DataFrame profiling |
| `data_retrieval` | `DataRetrievalExecutor` | DataFrame loading from sources (`type: sql`, `type: s3`, `type: lumid` with `mode: sql\|s3\|agent` via lumid-data-app; `type: lumid` (mode `sql`/`s3`/`agent`) requires `lumid_data_token`, the bearer forwarded to lumid-data-app) |
| `ssh` | `SSHExecutor` | Interactive SSH session or non-interactive container job |
| `serve` | `VLLMServeExecutor` | Persistent vLLM API server for a single model |

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

## Agent executor (utu / youtu-agent)

`AgentExecutor` requires the following env vars to run; the executor
asserts them at import time, so a worker without them fails the task
immediately:

- `UTU_LLM_TYPE` — provider kind (e.g. `chat.completions`).
- `UTU_LLM_MODEL` — model identifier.
- `UTU_LLM_BASE_URL` — LLM endpoint base URL.
- `UTU_LLM_API_KEY` — LLM API key.

Optional, for the search tools:

- `SERPER_API_KEY`
- `JINA_API_KEY`

## Agent-episode harness backends

When an agent's spec declares `harness.backend`, the worker runs it through the
dependency-light `AgentEpisodeExecutor` instead of the UTU path: one dispatch is one
run-to-yield step of the named `HarnessAdapter` binding, resumed from the durable capsule
and delivered outcomes the fabric ships. The executor advertises the `agent` capability
even on a worker that cannot import the UTU dependencies, so a CPU worker services agent
episodes. A backend binding is imported lazily from the worker adapter registry
(`src/worker/executors/harness/`) only when its key is selected.

Declared per agent under `spec.harness`:

- `backend` — the adapter binding: `scripted` (a deterministic backend that replays a
  declared step sequence from `params.script`) or `codex` (the version-pinned Codex
  app-server binding).
- `version` — pins the adapter/protocol so a capsule resumes only on a match.
- `params` — backend-specific configuration. The `codex` backend reads `base_url` and
  `model` (the Responses gateway it drives) and an optional `codex_home` override (the
  rollout directory, by default isolated per workflow and agent under the results dir).

The `codex` backend binds a live `codex app-server` through the `openai-codex` SDK
(`runtime-harness-codex` dependency group). It maps `thread/resume`, `thread/inject_items`,
and `turn/start` onto the adapter: a step runs against an on-disk rollout under
`CODEX_HOME`, and a settled outcome injects back as raw Responses items the next turn sees,
so a held facade call recovers across an app-server loss on the same rollout.
`tests/worker/test_codex_integration.py` exercises this against a live app-server and a
local Responses backend, including a `kill -9` recovery trace.

A mediated model request the agent defers is settled by the **agent-model gateway**, not a
raw model endpoint. The gateway implements the OpenAI Responses API (`POST /v1/responses`)
so a harness whose provider targets it crosses the same seam. Its upstream is configured
on the server under the `AGENT_MODEL_GATEWAY_*` env family (see
[`ENV.md`](ENV.md)); the default `canned` mode is deterministic and credential-free, while
`openai` forwards to an OpenAI-compatible endpoint, reusing the `UTU_LLM_*` provider path
when its own values are unset.
