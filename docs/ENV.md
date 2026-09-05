# Environment variables (curated)

The canonical declared set lives in
`cli/stack/src/flowmesh_cli_stack/env_schema.py` and is mirrored to
`cli/stack/src/flowmesh_cli_stack/assets/.env.example`. Run
`uv run scripts/dev/check_env_examples.py --write` after schema edits.

The tables below curate the knobs you actually tune. Anything not
listed here is in `.env.example`.

## Server

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ROLE` | `root` | `root` deploys local Redis; `worker` skips it and connects to the root's Redis via the URLs below |
| `REDIS_CONTROL_URL` | `redis://localhost:6379/0` | Redis control channel. On worker nodes, must point at the root node's reachable Redis endpoint |
| `REDIS_TELEMETRY_URL` | `redis://localhost:6380/0` | Redis telemetry channel. On worker nodes, must point at the root node's reachable Redis endpoint |
| `REDIS_RESIDENT_RELAY_URL` | (telemetry) | Redis endpoint for the resident relay; defaults to telemetry |
| `DATABASE_URL` | – | Postgres connection string |
| `RESULTS_DIR` | `./results` | Server-side results directory |
| `SERVER_RESULTS_DIR` | `flowmesh_results` | Host-side directory/docker volume to mount at `RESULTS_DIR` in the server container |
| `WORKER_RESULTS_DIR` | `flowmesh_results` | Server-side directory/docker volume to mount to worker containers |
| `SERVER_HTTP_PORT` | `8000` | Public HTTP port |
| `SERVER_GRPC_PORT` | `50051` | Supervisor gRPC port |
| `ORCHESTRATOR_DISPATCH_MODE` | `adaptive` | Scheduler mode |
| `ORCHESTRATOR_WORKER_SELECTION` | `best_fit` | `best_fit`, `first_fit`, `min_satisfying` |
| `SCHEDULER_LAMBDA_INFERENCE` | `0.4` | Inference task weight |
| `SCHEDULER_LAMBDA_TRAINING` | `0.8` | Training task weight |
| `SCHEDULER_LAMBDA_OTHER` | `0.5` | Other-task weight |
| `SCHEDULER_SELECTION_JITTER` | `1e-3` | Tie-break jitter |
| `ENABLE_TASK_MERGE` | `true` | DAG-level task coalescing |
| `TASK_MERGE_MAX_BATCH_SIZE` | `4` | Max merged tasks per dispatch |
| `ENABLE_CONTEXT_REUSE` | `true` | Bias toward workers with cached models |
| `WORKER_CACHE_TTL_SEC` | `3600` | Cache metadata TTL |
| `ENABLE_STAGE_WEIGHT_STICKINESS` | `false` | Pin stages to checkpoint-producing workers |
| `TASK_NO_WORKER_GRACE_SEC` | `60` | Grace before failing a task no worker can satisfy |
| `ORCHESTRATOR_MAX_SCOPE_DEPTH` | `64` | Max nested call/spawn/recursion depth for v2 dynamic regions |
| `ORCHESTRATOR_MAX_LOOP_ITERATIONS` | `1000` | Max loop iterations per v2 `LoopContext` |
| `ORCHESTRATOR_MAX_ACTIVATIONS` | `10000` | Max dynamic activations per v2 workflow instance |
| `ORCHESTRATOR_MAX_SPAWNS_PER_TURN` | `32` | Max spawn children admitted in one facade turn group |
| `ORCHESTRATOR_MAX_SPAWNS_PER_REGION` | `256` | Max spawn children admitted per agent child region |
| `ORCHESTRATOR_EPISODE_LOWERING` | `false` | Lower v2 templates into run-to-yield episodes |
| `ORCHESTRATOR_WORKER_ORIGINATED_BOUNDARIES` | `true` | Originate mediated tool boundaries from workers |
| `ORCHESTRATOR_AGENT_INPUT_BUDGET_BYTES` | `262144` | Max resolved first-turn input bytes per agent |
| `AGENT_HARNESS_DEFAULT_BACKEND` | – | Default agent harness backend when a workflow sets none |
| `AGENT_HARNESS_DEFAULT_VERSION` | – | Default agent harness backend version |
| `AGENT_MODEL_GATEWAY_MODE` | `canned` | Managed model upstream mode (`canned`/`echo`/`openai`/`proxy`) |
| `AGENT_MODEL_GATEWAY_URL` | – | Upstream base URL (openai/proxy modes) |
| `AGENT_MODEL_GATEWAY_MODEL` | – | Upstream model (openai/proxy modes) |
| `AGENT_MODEL_GATEWAY_TIMEOUT_SEC` | `60` | Upstream request timeout (seconds) |
| `AGENT_MODEL_SECRET_TTL_SEC` | `86400` | Expiry for a workflow's vaulted model credential |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` | Fabric web-search backend |
| `WEB_SEARCH_API_KEY` | – | Deployment key for a keyed search provider |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Results per search |
| `WEB_SEARCH_TIMEOUT_SEC` | `20` | Search request timeout (seconds) |
| `WEB_SEARCH_RESULT_CHAR_CAP` | `6000` | Injected result size cap |
| `WEB_SEARCH_MAX_CALLS` | `8` | Searches per episode |
| `WEB_SEARCH_MAX_PARALLEL_CALLS_PER_TURN` | `4` | Parallel searches per turn |
| `WEB_SEARCH_EGRESS_LOCALITY` | `server_relay` | Where a search egresses (`server_relay` or `worker_sidecar`) |
| `WEB_SEARCH_SIDECAR_REMOTE` | `false` | Carry a worker-sidecar search to a remote node |
| `WEB_SEARCH_SIDECAR_ROUTE` | `127.0.0.1:0` | Remote sidecar bind route |
| `WEB_SEARCH_SIDECAR_DIRECTLY_ROUTABLE` | `false` | Offer a direct dial to the remote sidecar |
| `CONTENT_STORE_ENABLED` | `true` | Serve the outcome content store |
| `CONTENT_STORE_ROOT` | – | Content-store root; under the data dir if empty |
| `RESIDENT_CAPACITY_ENABLED` | `false` | Serve resident model bindings via admission |
| `RESIDENT_INFERENCE_SUBSTRATE` | `serve` | Resident replica substrate (`serve` or `dev_model`) |
| `RESIDENT_SERVE_ACCESS_MODE` | `forward` | Materialized replica endpoint access mode |
| `RESIDENT_ADMISSION_SLOTS` | `8` | Conservative safe admission slots per replica |
| `RESIDENT_MAX_REPLICAS_PER_FAMILY` | `1` | Replica quota per service family |
| `RESIDENT_MAX_COLD_STARTS` | `1` | Concurrent cold starts |
| `RESIDENT_COLD_START_DEADLINE_SEC` | `300` | Cold-start / admission wait budget (seconds) |
| `RESIDENT_POLL_INTERVAL_SEC` | `1` | Admission wait poll interval (seconds) |
| `RESIDENT_REDRIVE_BACKOFF_SEC` | `0.5` | Backoff before re-driving a held resident invocation |
| `RESIDENT_MAX_TRANSIENT_REDRIVES` | `3` | Transient resident losses before a replica preempt |
| `RESIDENT_SERVE_TTL_SEC` | – | Materialized replica TTL (seconds) |
| `RESIDENT_ALLOWED_MODELS` | – | Comma-separated allowed model catalog; any if empty |
| `RESIDENT_FORWARD_API_KEY` | – | Credential the adapter presents to a keyless replica |
| `RESIDENT_SELECTION_STRATEGY` | `batch-aware-best-fit` | Per-family replica-selection strategy |
| `RESIDENT_IDLE_RETAIN_SEC` | `0` | Idle retain window before teardown; 0 disables |
| `RESIDENT_IDLE_SWEEP_INTERVAL_SEC` | `30` | Idle-teardown sweep interval (seconds) |
| `RESIDENT_SIDECAR_BIND_HOST` | `127.0.0.1` | Host a resident sidecar binds on the replica node |
| `RESIDENT_SIDECAR_DIRECTLY_ROUTABLE` | `false` | Advertise the resident sidecar as directly routable |
| `RESIDENT_RELAY_ONLY` | `false` | Mandate the reverse-relay for resident traffic |
| `NETWORK_PLANE_ENABLED` | `false` | Enable the route-discovery and relay substrate |
| `NETWORK_PLANE_ENDPOINT_URL` | – | Advertised node-relay endpoint (`host:port`) |
| `NETWORK_PLANE_SIDECAR_URL` | – | Node-local echo listener (`host:port`) |
| `NETWORK_PLANE_TRUST_DOMAIN` | `flowmesh` | Endpoint trust domain |
| `NETWORK_PLANE_REACHABILITY_CLASS` | `routable` | Endpoint reachability class |
| `NETWORK_PLANE_PROTOCOLS` | `echo` | Advertised transport protocols |
| `NETWORK_PLANE_POSITIVE_TTL_SEC` | `30` | Verified reachability TTL (seconds) |
| `NETWORK_PLANE_NEGATIVE_TTL_SEC` | `15` | Demoted reachability TTL (seconds) |
| `NETWORK_PLANE_BACKOFF_BASE_SEC` | `1` | Demotion retry backoff base (seconds) |
| `NETWORK_PLANE_BACKOFF_MAX_SEC` | `30` | Demotion retry backoff cap (seconds) |
| `NETWORK_PLANE_CONNECT_BUDGET_SEC` | `5` | Per-candidate optimistic connect budget (seconds) |
| `NETWORK_PLANE_ROUTE_TTL_SEC` | `30` | Resolved-route snapshot TTL (seconds) |
| `NETWORK_PLANE_RELAY_BUFFER_BYTES` | `65536` | Bounded relay-session in-flight buffer (bytes) |
| `NETWORK_PLANE_RELAY_WINDOW_BYTES` | `65536` | Reverse-relay per-direction in-flight window (bytes) |
| `ENABLE_WORKER_WATCHDOG` | `true` | Worker death detection |
| `WORKER_DEATH_GRACE_SEC` | `60` | Grace period before marking dead |
| `WORKER_REHYDRATION_GRACE_SEC` | `120` | Extra grace for a worker's rehydrated in-flight tasks after the root restarts, before the watchdog may reclaim them |
| `FLOWMESH_PLUGINS` | – | Comma-separated plugin module names |
| `FLOWMESH_PLUGIN_DATA_DIR` | `./plugin-data` | Writable mount at `/app/plugin-data` for plugin state. A path -> host bind-mount (auto-created); a bare name -> external Docker volume of that name. |
| `SERVER_CUDA_PROBE_IMAGE` | `nvidia/cuda:12.9.1-base-ubuntu24.04` | CUDA image the server runs briefly to query local GPU names/indices |
| `DOCKER_GPU_RUNTIME` | nvidia | Optional Docker runtime name for GPU probe/worker containers; leave empty unless the host requires a named runtime such as `nvidia` |
| `FLOWMESH_API_KEY` | – | Forwarded to spawned workers as their server-callback bearer |
| `ENABLE_PERSISTENT_PORT_FORWARD` | `true` | Keep port-forward listeners bound between task sessions; disable to bind listeners only for active sessions |
| `ENABLE_SERVER_SSH_PROXY` | `true` | Enable the WebSocket proxy for interactive SSH tasks |
| `ENABLE_SERVER_SERVE_PROXY` | `true` | Enable the HTTP reverse proxy for `serve` tasks |
| `LOG_LEVEL` | `INFO` | Server log level |

**Notes:**
- In Docker deployments, `SERVER_RESULTS_DIR` and `WORKER_RESULTS_DIR`
are the host directories or Docker volumes mounted into the server and
worker containers for storing and reading task results. For workflows
with a local output destination (`spec.output.destination.type="local"`)
that have downstream tasks, both variables must point to the same shared
directory or volume so the server can access the worker's task results.
Otherwise, downstream tasks that depend on upstream outputs will stall
in the dispatching loop indefinitely.
- When multiple deployments share one host, you can set `FLOWMESH_STACK_SUFFIX`
in `.env` to differentiate the deployments so that FlowMesh stack CLI does
not interfere with each other.
- `DOCKER_GPU_RUNTIME` defaults to `nvidia`. On hosts where Docker GPU access
works with `--gpus all` but fails with `--runtime=nvidia` (for example, DGX
Spark), set `DOCKER_GPU_RUNTIME=` in the stack env.

## Worker

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_TOKEN` | – | Auth token for supervisor gRPC |
| `SUPERVISOR_GRPC_TARGET` | – | Supervisor gRPC endpoint |
| `RESULTS_DIR` | `./results` | Task output directory |
| `WORKER_TAGS` | `` | Scheduler hints |
| `WORKER_COST_PER_HOUR` | `1.0` | Cost metadata |
| `WORKER_UPLOAD_RESULTS` | `false` | Upload results when no destination set |
| `WORKER_EXECUTOR_IDLE_CLEANUP_SEC` | `60` | Seconds a worker waits before unloading an idle executor to release the resources it holds; higher values avoid reload thrash between tasks but keep those resources reserved while idle |
| `HF_CACHE_DIR` | – | Shared HuggingFace cache mount |
| `HEARTBEAT_INTERVAL_SEC` | `30` | Heartbeat cadence |
| `SERVE_DEFAULT_TTL_SEC` | `3600` | Default vLLM serve session TTL when `spec.ttlSeconds` is unset |
| `SERVE_MAX_TTL_SEC` | `86400` | Upper bound on vLLM serve session TTL, regardless of `spec.ttlSeconds` |
| `WORKER_ENABLE_DEV_MODEL` | `false` | Advertise the GPU-free `dev_model` executor |
| `DEV_MODEL_FORWARD_URL` | – | Upstream URL `dev_model` forwards to; canned if unset |

## Supervisor

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_NAMESPACE` / `NODE_CLUSTER` / `NODE_ALIAS` | defaults | Identity |
| `NODE_TAGS` | `` | Scheduler hints (CSV) |
| `SUPERVISOR_GRPC_DISABLE_SERVER_TLS` | `false` | Local-only insecure gRPC |
| `SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS` | `true` | gRPC keepalive |
| `SUPERVISOR_GRPC_EXTERNAL_PORT` | – | External port (when port-forwarded) |
| `SERVER_GRPC_TLS_*` | – | TLS certificate files |

## SSH session resource caps

When `enable_ssh` is true on a Docker worker, these configured
ceilings bound every SSH session container spawned by that worker.
Unset values mean unbounded (host-wide access).

| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_MAX_CPU` | – | Max CPU cores per SSH container (float, e.g. `4` or `2.5`). Sets Docker `nano_cpus`. |
| `SSH_MAX_MEMORY` | – | Max memory per SSH container (e.g. `8Gi`, `512Mi`, or a byte count). Sets Docker `mem_limit`. |
| `SSH_MAX_PIDS` | – | Max PIDs per SSH container. Sets Docker `pids_limit`. Admin-only — not user-overridable. |
| `ENABLE_SSH_GPU_LIMIT` | `false` | When `true`, mount only the GPU subset matching the spec (`count` / `type` / `memory`); otherwise mount all worker GPUs. |

The effective CPU/memory limit is `min(spec.resources.hardware, worker
cap)`. A task that requests more than the worker cap is dispatched to
another worker if one has a larger cap; otherwise the dispatcher
follows its standard requeue/retry behavior. The worker logs a startup
warning if SSH is enabled with no cap configured.
