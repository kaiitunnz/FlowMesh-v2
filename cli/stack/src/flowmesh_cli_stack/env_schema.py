"""Stack env schema."""

from flowmesh.models.nodes import NodeRole
from flowmesh_stack.env_schema import (
    EnvSchema,
    EnvSection,
    EnvVar,
    EnvVarType,
    require_all_or_none,
    require_if_true,
)

STACK_ENV_SCHEMA = EnvSchema(
    name="stack",
    header=[
        "# FlowMesh Stack Configuration",
        "# Copy to .env and adjust as needed",
    ],
    sections=[
        EnvSection(
            title="Image Source",
            vars=[
                EnvVar("FLOWMESH_REGISTRY", "ghcr.io/mlsys-io", required=True),
                EnvVar("FLOWMESH_VERSION", "dev", required=True),
                EnvVar(
                    "FLOWMESH_CACHE_VERSION",
                    description=[
                        "Optional registry cache lineage for stack push.",
                        "Leave empty to use the default stable cache scope.",
                    ],
                ),
                EnvVar("FLOWMESH_BUILD_REF", "local"),
            ],
        ),
        EnvSection(
            title="Node Identity",
            vars=[
                EnvVar(
                    "FLOWMESH_STACK_SUFFIX",
                    description=[
                        "Optional suffix appended to stack-managed Docker object "
                        "names.",
                        "Use a distinct suffix per local stack on shared hosts.",
                        "This isolates container, network, and volume names,",
                        "but each stack still needs unique ports.",
                    ],
                ),
                EnvVar(
                    "NODE_ROLE",
                    NodeRole.ROOT.value,
                    var_type=EnvVarType.ENUM,
                    choices=NodeRole,
                ),
                EnvVar("NODE_NAMESPACE", "flowmesh"),
                EnvVar("NODE_CLUSTER", "dev"),
                EnvVar("NODE_ALIAS", "node"),
                EnvVar("NODE_TAGS", var_type=EnvVarType.CSV),
                EnvVar("ENABLE_SUPERVISOR", "true", var_type=EnvVarType.BOOL),
                EnvVar("SERVER_HOST", "localhost", required=True),
                EnvVar(
                    "SERVER_HTTP_PORT",
                    "8000",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
                EnvVar(
                    "SERVER_GRPC_PORT",
                    "50051",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
                EnvVar(
                    "SERVER_LOG_LEVEL",
                    "INFO",
                    var_type=EnvVarType.LOG_LEVEL,
                ),
            ],
        ),
        EnvSection(
            title="Server gRPC TLS",
            description=["Leave empty to disable"],
            vars=[
                EnvVar(
                    "SERVER_TLS_DIR",
                    "./secrets/tls/server",
                    var_type=EnvVarType.DIR_PATH,
                    use_default=True,
                    ensure_path="create",
                ),
                EnvVar(
                    "SERVER_GRPC_TLS_CA_FILE",
                    "/etc/ssl/server/server-ca.pem",
                    var_type=EnvVarType.FILE_PATH,
                ),
                EnvVar(
                    "SERVER_GRPC_TLS_CERT_FILE",
                    "/etc/ssl/server/server.pem",
                    var_type=EnvVarType.FILE_PATH,
                ),
                EnvVar(
                    "SERVER_GRPC_TLS_KEY_FILE",
                    "/etc/ssl/server/server.key",
                    var_type=EnvVarType.FILE_PATH,
                ),
            ],
        ),
        EnvSection(
            title="Supervisor gRPC",
            description=[
                "Tuning for the supervisor's gRPC server and worker connections.",
                "Leave SUPERVISOR_GRPC_EXTERNAL_PORT empty unless workers connect",
                "through a port-forwarded / proxied address.",
            ],
            vars=[
                EnvVar(
                    "SUPERVISOR_GRPC_DISABLE_SERVER_TLS",
                    "false",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "SUPERVISOR_GRPC_EXTERNAL_PORT",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS",
                    "true",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "SUPERVISOR_GRPC_MIN_RECV_PING_INTERVAL_MS",
                    "60000",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar(
                    "SUPERVISOR_GRPC_KEEPALIVE_TIME_MS",
                    "300000",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar(
                    "SUPERVISOR_GRPC_KEEPALIVE_TIMEOUT_MS",
                    "10000",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
            ],
        ),
        EnvSection(
            title="Redis Connectivity",
            vars=[
                EnvVar(
                    "REDIS_CONTROL_URL",
                    "redis://localhost:6379/0",
                    var_type=EnvVarType.URL,
                    required=True,
                    url_schemes={"redis", "rediss"},
                ),
                EnvVar(
                    "REDIS_TELEMETRY_URL",
                    "redis://localhost:6380/0",
                    var_type=EnvVarType.URL,
                    required=True,
                    url_schemes={"redis", "rediss"},
                ),
                EnvVar(
                    "REDIS_RESIDENT_RELAY_URL",
                    "",
                    description=(
                        "Redis endpoint for the resident relay; defaults to telemetry."
                    ),
                    var_type=EnvVarType.URL,
                    url_schemes={"redis", "rediss"},
                ),
            ],
        ),
        EnvSection(
            title="Core Ports",
            vars=[
                EnvVar(
                    "REDIS_CONTROL_PORT",
                    "6379",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
                EnvVar(
                    "REDIS_TELEMETRY_PORT",
                    "6380",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
            ],
        ),
        EnvSection(
            title="Log Streams (Redis)",
            description=["Caps Redis Streams for per-task and per-workflow logs."],
            vars=[
                EnvVar(
                    "LOG_STREAM_MAXLEN_TASK",
                    "50000",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "LOG_STREAM_MAXLEN_WORKFLOW",
                    "200000",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "LOG_STREAM_TTL_SEC",
                    "3600",
                    description="Expire log stream keys after close (0 disables).",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar(
                    "TASK_LOG_ARCHIVE_FLUSH_INTERVAL_SEC",
                    "5",
                    description="Flush archived task logs at most every N seconds.",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.1,
                ),
                EnvVar(
                    "TASK_LOG_ARCHIVE_FLUSH_MAX_ENTRIES",
                    "100",
                    description="Flush archived task logs after buffering N entries.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
            ],
        ),
        EnvSection(
            title="v2 Orchestration Guardrails",
            description=["Backstops bounding structured dynamic regions."],
            vars=[
                EnvVar(
                    "ORCHESTRATOR_MAX_SCOPE_DEPTH",
                    "64",
                    description="Max nested call/spawn/recursion scope depth.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "ORCHESTRATOR_MAX_LOOP_ITERATIONS",
                    "1000",
                    description="Max loop-time iterations per LoopContext activation.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "ORCHESTRATOR_MAX_ACTIVATIONS",
                    "10000",
                    description="Max dynamic activations per workflow instance.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "ORCHESTRATOR_MAX_SPAWNS_PER_TURN",
                    "32",
                    description="Max spawn children admitted in one facade turn group.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "ORCHESTRATOR_MAX_SPAWNS_PER_REGION",
                    "256",
                    description="Max spawn children admitted per agent child region.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "ORCHESTRATOR_EPISODE_LOWERING",
                    "0",
                    description="Lower v2 templates into run-to-yield episodes.",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "ORCHESTRATOR_WORKER_ORIGINATED_BOUNDARIES",
                    "0",
                    description="Originate mediated tool boundaries from workers.",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "ORCHESTRATOR_AGENT_INPUT_BUDGET_BYTES",
                    "262144",
                    description="Max resolved first-turn input bytes per agent.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
            ],
        ),
        EnvSection(
            title="Agent Harness & Model Binding",
            description=["Deployment defaults for agent harnesses and their models."],
            vars=[
                EnvVar(
                    "AGENT_HARNESS_DEFAULT_BACKEND",
                    "",
                    description=(
                        "Default agent harness backend when a workflow sets none."
                    ),
                ),
                EnvVar(
                    "AGENT_HARNESS_DEFAULT_VERSION",
                    "",
                    description="Default agent harness backend version.",
                ),
                EnvVar(
                    "AGENT_MODEL_GATEWAY_MODE",
                    "canned",
                    description="Managed model upstream mode.",
                    choices=("canned", "echo", "openai", "proxy"),
                ),
                EnvVar(
                    "AGENT_MODEL_GATEWAY_URL",
                    "",
                    description="Upstream base URL (openai/proxy modes).",
                    var_type=EnvVarType.URL,
                ),
                EnvVar(
                    "AGENT_MODEL_GATEWAY_MODEL",
                    "",
                    description="Upstream model (openai/proxy modes).",
                ),
                EnvVar(
                    "AGENT_MODEL_GATEWAY_TIMEOUT_SEC",
                    "60",
                    description="Upstream request timeout (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "AGENT_MODEL_SECRET_TTL_SEC",
                    "86400",
                    description="Expiry for a workflow's vaulted model credential.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
            ],
        ),
        EnvSection(
            title="Web Search",
            description=["Fabric-mediated web_search tool for agents."],
            vars=[
                EnvVar(
                    "WEB_SEARCH_PROVIDER",
                    "duckduckgo",
                    description="Fabric web-search backend.",
                ),
                EnvVar(
                    "WEB_SEARCH_API_KEY",
                    "",
                    description="Deployment key for a keyed search provider.",
                ),
                EnvVar(
                    "WEB_SEARCH_MAX_RESULTS",
                    "5",
                    description="Results per search.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "WEB_SEARCH_TIMEOUT_SEC",
                    "20",
                    description="Search request timeout (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "WEB_SEARCH_RESULT_CHAR_CAP",
                    "6000",
                    description="Injected result size cap.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "WEB_SEARCH_MAX_CALLS",
                    "8",
                    description="Searches per episode.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "WEB_SEARCH_MAX_PARALLEL_CALLS_PER_TURN",
                    "4",
                    description="Parallel searches per turn.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "WEB_SEARCH_EGRESS_LOCALITY",
                    "server_relay",
                    description="Where a search egresses.",
                    choices=["server_relay", "worker_sidecar"],
                ),
                EnvVar(
                    "WEB_SEARCH_SIDECAR_REMOTE",
                    "false",
                    description="Carry a worker-sidecar search to a remote node.",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "WEB_SEARCH_SIDECAR_ROUTE",
                    "127.0.0.1:0",
                    description="Remote sidecar bind route.",
                ),
                EnvVar(
                    "WEB_SEARCH_SIDECAR_DIRECTLY_ROUTABLE",
                    "false",
                    description="Offer a direct dial to the remote sidecar.",
                    var_type=EnvVarType.BOOL,
                ),
            ],
        ),
        EnvSection(
            title="Resident-capacity control",
            vars=[
                EnvVar(
                    "RESIDENT_CAPACITY_ENABLED",
                    "false",
                    description="Serve resident model bindings via admission.",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "RESIDENT_INFERENCE_SUBSTRATE",
                    "serve",
                    description="Resident replica substrate (serve or dev_model).",
                    choices=["serve", "dev_model"],
                ),
                EnvVar(
                    "RESIDENT_SERVE_ACCESS_MODE",
                    "forward",
                    description="Materialized replica endpoint access mode.",
                    choices=["forward", "proxy", "direct"],
                ),
                EnvVar(
                    "RESIDENT_ADMISSION_SLOTS",
                    "8",
                    description="Conservative safe admission slots per replica.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "RESIDENT_MAX_REPLICAS_PER_FAMILY",
                    "1",
                    description="Replica quota per service family.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "RESIDENT_MAX_COLD_STARTS",
                    "1",
                    description="Concurrent cold starts.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "RESIDENT_COLD_START_DEADLINE_SEC",
                    "300",
                    description="Cold-start / admission wait budget (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "RESIDENT_POLL_INTERVAL_SEC",
                    "1",
                    description="Admission wait poll interval (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "RESIDENT_REDRIVE_BACKOFF_SEC",
                    "0.5",
                    description="Backoff before re-driving a held resident invocation.",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                ),
                EnvVar(
                    "RESIDENT_MAX_TRANSIENT_REDRIVES",
                    "3",
                    description="Transient resident losses before a replica preempt.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "RESIDENT_SERVE_TTL_SEC",
                    "",
                    description="Materialized replica TTL (seconds).",
                ),
                EnvVar(
                    "RESIDENT_ALLOWED_MODELS",
                    "",
                    description="Comma-separated allowed model catalog; any if empty.",
                ),
                EnvVar(
                    "RESIDENT_FORWARD_API_KEY",
                    "",
                    description="Credential the adapter presents to a keyless replica.",
                ),
                EnvVar(
                    "RESIDENT_SELECTION_STRATEGY",
                    "batch-aware-best-fit",
                    description="Per-family replica-selection strategy.",
                    choices=["batch-aware-best-fit", "least-load", "round-robin"],
                ),
                EnvVar(
                    "RESIDENT_IDLE_RETAIN_SEC",
                    "0",
                    description="Idle retain window before teardown; 0 disables.",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                ),
                EnvVar(
                    "RESIDENT_IDLE_SWEEP_INTERVAL_SEC",
                    "30",
                    description="Idle-teardown sweep interval (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "RESIDENT_SIDECAR_BIND_HOST",
                    "127.0.0.1",
                    description="Host a resident sidecar binds on the replica node.",
                ),
                EnvVar(
                    "RESIDENT_SIDECAR_DIRECTLY_ROUTABLE",
                    "false",
                    description="Advertise the resident sidecar as directly routable.",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "RESIDENT_RELAY_ONLY",
                    "false",
                    description="Mandate the reverse-relay for resident traffic.",
                    var_type=EnvVarType.BOOL,
                ),
            ],
        ),
        EnvSection(
            title="Reference-backed outcomes",
            vars=[
                EnvVar(
                    "CONTENT_STORE_ENABLED",
                    "true",
                    description="Serve the outcome content store.",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "CONTENT_STORE_ROOT",
                    "",
                    description="Content-store root; under the data dir if empty.",
                ),
            ],
        ),
        EnvSection(
            title="Network plane",
            vars=[
                EnvVar(
                    "NETWORK_PLANE_ENABLED",
                    "false",
                    description="Enable the route-discovery and relay substrate.",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "NETWORK_PLANE_ENDPOINT_URL",
                    "",
                    description="Advertised node-relay endpoint (host:port).",
                ),
                EnvVar(
                    "NETWORK_PLANE_SIDECAR_URL",
                    "",
                    description="Node-local echo listener (host:port).",
                ),
                EnvVar(
                    "NETWORK_PLANE_TRUST_DOMAIN",
                    "flowmesh",
                    description="Endpoint trust domain.",
                ),
                EnvVar(
                    "NETWORK_PLANE_REACHABILITY_CLASS",
                    "routable",
                    description="Endpoint reachability class.",
                    choices=["same_node", "same_cluster", "routable"],
                ),
                EnvVar(
                    "NETWORK_PLANE_PROTOCOLS",
                    "echo",
                    description="Advertised transport protocols.",
                    var_type=EnvVarType.CSV,
                ),
                EnvVar(
                    "NETWORK_PLANE_POSITIVE_TTL_SEC",
                    "30",
                    description="Verified reachability TTL (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "NETWORK_PLANE_NEGATIVE_TTL_SEC",
                    "15",
                    description="Demoted reachability TTL (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "NETWORK_PLANE_BACKOFF_BASE_SEC",
                    "1",
                    description="Demotion retry backoff base (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "NETWORK_PLANE_BACKOFF_MAX_SEC",
                    "30",
                    description="Demotion retry backoff cap (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "NETWORK_PLANE_CONNECT_BUDGET_SEC",
                    "5",
                    description="Per-candidate optimistic connect budget (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "NETWORK_PLANE_ROUTE_TTL_SEC",
                    "30",
                    description="Resolved-route snapshot TTL (seconds).",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar(
                    "NETWORK_PLANE_RELAY_BUFFER_BYTES",
                    "65536",
                    description="Bounded relay-session in-flight buffer (bytes).",
                    var_type=EnvVarType.INT,
                    min_value=1024,
                ),
                EnvVar(
                    "NETWORK_PLANE_RELAY_WINDOW_BYTES",
                    "65536",
                    description="Reverse-relay per-direction in-flight window (bytes).",
                    var_type=EnvVarType.INT,
                    min_value=1024,
                ),
            ],
        ),
        EnvSection(
            title="Redis Access",
            vars=[
                EnvVar("REDIS_ACL_ENABLED", "1", var_type=EnvVarType.BOOL),
                EnvVar("REDIS_USERNAME", "admin"),
                EnvVar("REDIS_PASSWORD", "very-strong-password"),
            ],
        ),
        EnvSection(
            title="Redis TLS",
            description=["Leave empty to disable"],
            vars=[
                EnvVar(
                    "REDIS_TLS_DIR",
                    "./secrets/tls/redis",
                    var_type=EnvVarType.DIR_PATH,
                    use_default=True,
                    ensure_path="create",
                ),
                EnvVar(
                    "REDIS_TLS_CA_FILE",
                    "/etc/ssl/redis/redis-ca.pem",
                    var_type=EnvVarType.FILE_PATH,
                ),
                EnvVar(
                    "REDIS_TLS_CERT_FILE",
                    "/etc/ssl/redis/redis-server.pem",
                    var_type=EnvVarType.FILE_PATH,
                ),
                EnvVar(
                    "REDIS_TLS_KEY_FILE",
                    "/etc/ssl/redis/redis-server.key",
                    var_type=EnvVarType.FILE_PATH,
                ),
            ],
        ),
        EnvSection(
            title="Port Forward Support",
            vars=[
                EnvVar("ENABLE_SERVER_PORT_FORWARD", "true", var_type=EnvVarType.BOOL),
                EnvVar(
                    "ENABLE_PERSISTENT_PORT_FORWARD", "true", var_type=EnvVarType.BOOL
                ),
                EnvVar("SERVER_PORT_FORWARD_BIND_HOST", "0.0.0.0", required=True),
                EnvVar("SERVER_PORT_FORWARD_PUBLIC_HOST", "localhost", required=True),
                EnvVar(
                    "SERVER_PORT_FORWARD_PORT_START",
                    "32000",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
                EnvVar(
                    "SERVER_PORT_FORWARD_PORT_END",
                    "32100",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
            ],
        ),
        EnvSection(
            title="SSH Task Support",
            vars=[
                EnvVar("ENABLE_SERVER_SSH_PROXY", "true", var_type=EnvVarType.BOOL),
                EnvVar(
                    "ENABLE_SERVER_SSH_CONNECTION_AUDIT",
                    "true",
                    var_type=EnvVarType.BOOL,
                ),
            ],
        ),
        EnvSection(
            title="Serve Task Support",
            vars=[
                EnvVar("ENABLE_SERVER_SERVE_PROXY", "true", var_type=EnvVarType.BOOL),
            ],
        ),
        EnvSection(
            title="SSH Worker Defaults",
            vars=[
                EnvVar("ENABLE_SSH_BY_DEFAULT", "true", var_type=EnvVarType.BOOL),
                EnvVar("SSH_DEFAULT_IMAGE"),
                EnvVar("SSH_DEFAULT_USER"),
                EnvVar("SSH_DEFAULT_TTL_SEC", var_type=EnvVarType.FLOAT, min_value=0),
                EnvVar("SSH_DEFAULT_IDLE_SEC", var_type=EnvVarType.FLOAT, min_value=0),
                EnvVar("SSH_MAX_TTL_SEC", var_type=EnvVarType.FLOAT, min_value=0),
                EnvVar("SSH_POLL_INTERVAL_SEC", var_type=EnvVarType.FLOAT, min_value=0),
                EnvVar("SSH_STOP_TIMEOUT_SEC", var_type=EnvVarType.FLOAT, min_value=0),
                EnvVar(
                    "SSH_MAX_CPU",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                    min_inclusive=False,
                ),
                EnvVar("SSH_MAX_MEMORY"),
                EnvVar("SSH_MAX_PIDS", var_type=EnvVarType.INT, min_value=1),
                EnvVar(
                    "ENABLE_SSH_GPU_LIMIT",
                    "false",
                    var_type=EnvVarType.BOOL,
                    description=[
                        "Whether to apply requested GPU limits to SSH tasks.",
                        "If false, SSH tasks are allocated all available GPUs",
                        "regardless of their resource requests.",
                    ],
                ),
            ],
        ),
        EnvSection(
            title="General Settings",
            vars=[
                EnvVar("TZ", "Asia/Singapore", required=True),
                EnvVar(
                    "LOG_LEVEL", "INFO", var_type=EnvVarType.LOG_LEVEL, required=True
                ),
            ],
        ),
        EnvSection(
            title="Orchestrator Settings",
            vars=[
                EnvVar(
                    "ORCHESTRATOR_DISPATCH_MODE",
                    "adaptive",
                    var_type=EnvVarType.ENUM,
                    choices={"adaptive"},
                ),
                EnvVar(
                    "ORCHESTRATOR_WORKER_SELECTION",
                    "best_fit",
                    var_type=EnvVarType.ENUM,
                    choices={"best_fit", "first_fit", "min_satisfying"},
                ),
                EnvVar(
                    "SCHEDULER_SELECTION_JITTER",
                    "0.001",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.0,
                ),
                EnvVar(
                    "SCHEDULER_LAMBDA_INFERENCE",
                    "0.4",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.0,
                ),
                EnvVar(
                    "SCHEDULER_LAMBDA_TRAINING",
                    "0.8",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.0,
                ),
                EnvVar(
                    "SCHEDULER_LAMBDA_OTHER",
                    "0.5",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.0,
                ),
                EnvVar("ENABLE_TASK_MERGE", "true", var_type=EnvVarType.BOOL),
                EnvVar(
                    "TASK_MERGE_MAX_BATCH_SIZE",
                    "4",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar("ENABLE_CONTEXT_REUSE", "true", var_type=EnvVarType.BOOL),
                EnvVar(
                    "WORKER_CACHE_TTL_SEC",
                    "3600",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar(
                    "ENABLE_STAGE_WEIGHT_STICKINESS",
                    "false",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "TASK_NO_WORKER_GRACE_SEC",
                    "60",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar("ENABLE_WORKER_WATCHDOG", "true", var_type=EnvVarType.BOOL),
                EnvVar(
                    "WORKER_DEATH_CHECK_INTERVAL",
                    "30",
                    var_type=EnvVarType.INT,
                    min_value=5,
                ),
                EnvVar(
                    "WORKER_DEATH_GRACE_SEC",
                    "60",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar(
                    "WORKER_REHYDRATION_GRACE_SEC",
                    "120",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
            ],
        ),
        EnvSection(
            title="Server Heartbeat",
            vars=[
                EnvVar(
                    "SERVER_HEARTBEAT_INTERVAL",
                    "30",
                    var_type=EnvVarType.INT,
                    min_value=1,
                )
            ],
        ),
        EnvSection(
            title="Vast.ai Configuration",
            vars=[
                EnvVar("VAST_SEARCH_LIMIT", var_type=EnvVarType.INT, min_value=0),
                EnvVar("VAST_MAX_RETRIES", var_type=EnvVarType.INT, min_value=0),
            ],
        ),
        EnvSection(
            title="Worker Parameters",
            vars=[
                EnvVar("WORKER_LOG_LEVEL", "INFO", var_type=EnvVarType.LOG_LEVEL),
                EnvVar(
                    "HEARTBEAT_INTERVAL_SEC", "30", var_type=EnvVarType.INT, min_value=1
                ),
                EnvVar(
                    "WORKER_COST_PER_HOUR",
                    "1.0",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.0,
                ),
                EnvVar(
                    "SERVER_RESULTS_DIR",
                    var_type=EnvVarType.DIR_PATH,
                    description=[
                        "Directory/Docker volume for the server to look up task "
                        "results after worker completion.",
                        "Set to the same value as WORKER_RESULTS_DIR so the server "
                        "can access worker results.",
                        "For workflows with a local output destination "
                        '(`spec.output.destination.type="local"`),',
                        "`SERVER_RESULTS_DIR` and `WORKER_RESULTS_DIR` must point to "
                        "the same shared directory",
                        "or volume; otherwise, the server cannot read the worker's "
                        "outputs and downstream tasks",
                        "will stall in the dispatching loop.",
                        "Defaults to the stack-scoped results volume when empty.",
                    ],
                ),
                EnvVar(
                    "WORKER_RESULTS_DIR",
                    var_type=EnvVarType.DIR_PATH,
                    description=[
                        "Defaults to the stack-scoped results volume when empty."
                    ],
                ),
                EnvVar("HF_CACHE_DIR", var_type=EnvVarType.DIR_PATH),
                EnvVar(
                    "WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar("WORKER_TAGS", var_type=EnvVarType.CSV),
                EnvVar("WORKER_HB_DIR", var_type=EnvVarType.DIR_PATH),
                EnvVar(
                    "FLOWMESH_BASE_URL",
                    "http://localhost:8000",
                    var_type=EnvVarType.URL,
                    required=True,
                    url_schemes={"http", "https"},
                ),
                EnvVar(
                    "FLOWMESH_API_KEY",
                    description="Supplier API key for worker authentication with "
                    "the server",
                ),
                EnvVar(
                    "NEBULA_API_BASE_URL",
                    var_type=EnvVarType.URL,
                    url_schemes={"http", "https"},
                ),
                EnvVar(
                    "SERVER_CUDA_PROBE_IMAGE",
                    "nvidia/cuda:12.9.1-base-ubuntu24.04",
                    description="Server-side CUDA image used to probe local GPUs.",
                ),
                EnvVar(
                    "DOCKER_GPU_RUNTIME",
                    "nvidia",
                    description="Optional Docker runtime name for GPU containers.",
                ),
                EnvVar(
                    "CUDA_VISIBLE_DEVICES", "all", var_type=EnvVarType.CSV_INTS_OR_ALL
                ),
                EnvVar("WORKER_UPLOAD_RESULTS", "false", var_type=EnvVarType.BOOL),
                EnvVar(
                    "WORKER_EXECUTOR_IDLE_CLEANUP_SEC",
                    "60",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                ),
                EnvVar(
                    "MODEL_CLEANUP_AFTER_UPLOAD",
                    "0",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar(
                    "SERVE_DEFAULT_TTL_SEC",
                    "3600",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                ),
                EnvVar(
                    "SERVE_MAX_TTL_SEC",
                    "86400",
                    var_type=EnvVarType.FLOAT,
                    min_value=0,
                ),
                EnvVar(
                    "WORKER_ENABLE_DEV_MODEL",
                    "false",
                    var_type=EnvVarType.BOOL,
                    description="Advertise the GPU-free dev_model executor.",
                ),
                EnvVar(
                    "DEV_MODEL_FORWARD_URL",
                    var_type=EnvVarType.URL,
                    url_schemes={"http", "https"},
                    description="Upstream URL dev_model forwards to; canned if unset.",
                ),
            ],
        ),
        EnvSection(
            title="Model Pre-downloading",
            description=[
                "Comma-separated list of models to pre-download during worker startup",
                "Leave empty to disable model pre-downloading",
                "Example: meta-llama/Llama-3.2-1B-Instruct,"
                "meta-llama/Llama-3.2-3B-Instruct",
            ],
            vars=[EnvVar("PREDOWNLOAD_MODEL_LIST", var_type=EnvVarType.CSV)],
        ),
        EnvSection(
            title="API Keys injected into workers (optional)",
            vars=[
                EnvVar("OPENAI_API_KEY"),
                EnvVar("GOOGLE_API_KEY"),
                EnvVar("VAST_API_KEY"),
                EnvVar("HF_TOKEN"),
                EnvVar("NEBULA_API_TOKEN"),
            ],
        ),
        EnvSection(
            title="External Plugins",
            description=[
                "Plugins are Python packages dropped under FLOWMESH_PLUGIN_DIR ",
                "(read-only at /app/plugins) and selected by FLOWMESH_PLUGINS as ",
                "a comma-separated list of top-level module names. Each must ",
                "expose `install()` returning a `HookBindings`. ",
                "FLOWMESH_PLUGIN_DATA_DIR is writable at /app/plugin-data for ",
                "plugin state. Leave all empty unless you ship a plugin.",
            ],
            vars=[
                EnvVar(
                    "FLOWMESH_PLUGIN_DIR",
                    "./plugins",
                    var_type=EnvVarType.DIR_PATH,
                    use_default=True,
                    ensure_path="create",
                ),
                EnvVar(
                    "FLOWMESH_PLUGIN_DATA_DIR",
                    "./plugin-data",
                    use_default=True,
                    description=[
                        "A path (`./x`, `/abs/x`) -> host bind-mount (auto-created).",
                        "A bare name -> external Docker volume of that name.",
                    ],
                ),
                EnvVar("FLOWMESH_PLUGINS", ""),
            ],
        ),
        EnvSection(
            title="n8n Integration",
            vars=[
                EnvVar(
                    "N8N_CREDENTIAL_AES_PASSWORD",
                    description="AES-GCM key to decrypt encrypted n8n credentials.",
                    warn_if_empty=True,
                )
            ],
        ),
        EnvSection(
            title="Worker launch config (optional)",
            vars=[
                EnvVar(
                    "SERVER_WORKER_CONFIG",
                    "./configs/worker_config.yaml",
                    var_type=EnvVarType.FILE_PATH,
                    use_default=True,
                    ensure_path="create",
                )
            ],
        ),
        EnvSection(
            title="Logging",
            vars=[
                EnvVar(
                    "LOG_MAX_BYTES",
                    "5242880",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "LOG_BACKUP_COUNT",
                    "5",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar("SERVER_APP_RELOAD", "0", var_type=EnvVarType.BOOL),
                EnvVar("SERVER_APP_LOG_LEVEL", "info", var_type=EnvVarType.LOG_LEVEL),
            ],
        ),
    ],
    validators=[
        lambda env, errors, warnings: require_if_true(
            env, "REDIS_ACL_ENABLED", ["REDIS_USERNAME", "REDIS_PASSWORD"], errors
        ),
        lambda env, errors, warnings: require_all_or_none(
            env,
            [
                "SERVER_GRPC_TLS_CA_FILE",
                "SERVER_GRPC_TLS_CERT_FILE",
                "SERVER_GRPC_TLS_KEY_FILE",
            ],
            errors,
        ),
    ],
)


# Schema-default overrides applied when rendering a worker-role .env.
# Unused vars are blanked out to avoid confusion and misconfiguration.
WORKER_ROLE_OVERRIDES = {
    "NODE_ROLE": NodeRole.WORKER.value,
    "REDIS_TLS_CERT_FILE": "",
    "REDIS_TLS_KEY_FILE": "",
}


def role_overrides(role: NodeRole) -> dict[str, str]:
    """Return the schema-default overrides for a given role's rendered .env."""
    return WORKER_ROLE_OVERRIDES.copy() if role == NodeRole.WORKER else {}


def deploy_overrides(deploy: bool, version: str | None = None) -> dict[str, str]:
    """Return the schema-default overrides for a deploy-shaped rendered .env."""
    if not (deploy and version):
        return {}
    return {"FLOWMESH_VERSION": version}
