# Network plane

The network plane is the topology-aware, control-resolved routing substrate. It turns
trusted endpoint advertisements and directional reachability evidence into an ordered,
expiry-bounded route between a caller and a target listener, and it carries bytes over one
of three transports. It is control-resolved and data-direct: the resolver runs in the
control plane and the origin-side deputy executes only the resolved candidate ladder —
workers never scan addresses or discover peers.

The plane is a routing substrate: it resolves and carries what a caller frames over it, and
authority stays with the caller. It mints no `ServiceClaim`, releases no credit, and issues
no `RouteAuthorization`; a route resolves to a resident-facing sidecar listener, never to an
engine. A route observation is network evidence; it can never promote, release, or overwrite
a capacity credit. Enable it with `NETWORK_PLANE_ENABLED=true` (`docs/ENV.md` lists the
knobs).

## The four facts

Route resolution is derived from four separate facts, each with its own owner and fence.

- **`NetworkEndpointAdvertisement`** — an operator-configured, identity/TLS-bound node (or
  registered ingress edge) network-plane endpoint: a purpose-scoped URL, an endpoint
  `generation`, trust domain, reachability class, and protocol capability. It rides on
  node registration and is carried on the node record, so the node registry is its durable
  carrier. It is not a worker-supplied arbitrary URL and not the generic server-management
  endpoint. It also carries the node's non-secret outbound relay-attachment identity —
  attach eligibility for the universal `control_relay`, not an inbound address. A node may
  advertise with an empty URL, the outbound-only shape, and the server derives its endpoint
  and attachment identity from the node id. Re-registration mints a fresh endpoint
  generation, the fence that invalidates route evidence bound to a superseded one.
- **`ReplicaListenerAdvertisement`** — a non-secret listener in the replica directory,
  fenced by replica incarnation and listener generation. It names the resident-facing
  sidecar/adapter capability and its route endpoint(s); it never names a raw engine
  listener or credential.
- **`NetworkReachabilityView`** — a derived, demand-paged, directional view over
  classified `RouteObservation`s, keyed by `(RouteOrigin, target node, listener
  generation, transport)`. Heartbeats carry only a node's own advertisement and liveness;
  there is no background pairwise probing.
- **`RouteOrigin`** — a trusted caller origin, bound by control to a registered source
  endpoint and network/policy class. It is the execution-network source, never the logical
  caller, and it owns no admission policy.

The **route resolver** is a pure function of a `RouteOrigin`, the target listener, the
target node's endpoint advertisement, and a reachability snapshot. It returns an ordered,
expiry-bounded `ResolvedRoute` candidate ladder with explicit hops and a route epoch. It
issues no authority, chooses no capacity, and mutates nothing.

## Reachability state machine

Each directional key moves through `UNKNOWN → OPTIMISTIC → VERIFIED → DEMOTED`:

- A resolve marks the candidates it will try `OPTIMISTIC`.
- A `verified` observation sets `VERIFIED` with a positive TTL; past the TTL the key
  returns `UNKNOWN` for re-verification.
- A network-path failure — DNS, connect, TLS, route, or timeout — sets `DEMOTED` with a
  negative TTL and bounded, exponential retry backoff; once the backoff cools the key
  returns `UNKNOWN` for an optimistic retry.
- Authority, tenant, fence, application, and engine failures are not path evidence and
  leave the state unchanged — they never demote.

Keying observations by listener generation fences a stale advertisement: a new generation
gets fresh `UNKNOWN` entries, and node re-registration invalidates a target's entries.

## Transports

The resolver orders three transport candidates. `control_relay` is the universal base,
carried whenever the origin and target both advertise an outbound relay attachment;
policy may rank a verified `worker_direct` or `node_relay` offload ahead of it, and a
demoted offload drops out until its backoff cools.

- **`worker_direct`** — the dialer to the listener. A forward-dial offload legal only for an
  explicitly directly routable listener whose endpoint class the dialer's network class can
  reach, under a bounded optimistic connect budget. Shared-node placement alone does not
  make it legal.
- **`node_relay`** — the dialer to the target node's announced endpoint, which uplinks over
  an authenticated node-local relay session to the target listener the route names. A
  forward-dial offload; the initial same-node path as well as the normal cross-node path.
- **`control_relay`** — the universal reverse-rendezvous base. Its descriptor names the
  origin and target reverse attachments by node (the delivery routes by node id) and the
  target's node-local sidecar delivery, not a chain of dialable addresses. It is feasible
  whenever both ends hold a live outbound attachment, so it resolves for an outbound-only
  node where the forward-dial offloads do not.

A route has two origins. The logical `RouteOrigin` — a worker-side deputy or the root's
gated serve ingress — grades `control_relay` and is the route's identity. The two
forward-dial candidates are opened by the root, so they are graded against the root's own
origin and their reachability evidence accumulates once for the root rather than separately
under every caller. A reachability probe passes its own origin for both and admits no
offload.

## Trusted target-leg offloads

A resident invocation's target leg leaves the reverse-rendezvous relay for a direct socket
only where the deployment declares the root-to-target pair trusted. The resolver marks a
forward-dial candidate as admitted when the target node advertises the required trust
domain, one of the trusted reachability classes, and the mutually authenticated transport,
under its current endpoint and listener generation, with root-to-target reachability
evidence that has not demoted; a `worker_direct` candidate additionally needs the listener
itself to advertise that transport, and a `node_relay` candidate the node's purpose-scoped
target-leg listener. Anything else is carried over `control_relay`. Enable the policy with
`NETWORK_PLANE_TARGET_LEG_ENABLED=true`.

Both ends authenticate with mutual TLS from the configured CA bundle, and the target pins
the configured root identity, so a certificate the CA signed for another party is refused
before any frame is read. The `worker_direct` target is the replica worker's own claim-gated
listener, whose port the worker reports at registration and control composes with the node's
advertised host. The `node_relay` target is a node listener that resolves each frame's
control-minted relay session to that node's local sidecar uplink — the dialer supplies no
host, port, or engine endpoint.

An offloaded leg carries the same frames as the relay: the claim-bound handoff before
acceptance, the route authorization after it, the invocation and request identities,
descriptor and generation fences, byte windows, cursors, and cancellation. The root writes
frames through and reads no payload, engine token, cursor, or window, and the target
sidecar's claim gate remains the only authority over what reaches an engine.

A dial that fails before any frame is delivered records a classified path observation and
carries the attempt over `control_relay` under the same claim, request identity, and held
credit. Once a frame has been written the leg never switches transport: a loss from there on
leaves the outcome ambiguous, which the origin reports as uncertain with its credit held.
Fence, tenant, descriptor, application, and engine rejections are authorization failures, so
they neither demote a path nor select a fallback.

Only the target leg moves. A workflow origin keeps carrying its source-to-root leg over its
own outbound attachment; a root-sourced gated serve call is its own origin, so both of its
legs leave the rendezvous. `GET /api/v1/network/legs` counts resident frames and payload
bytes per leg and transport, so the two read apart.

## Reverse-rendezvous relay

`control_relay` carries a resident invocation without either end accepting an inbound
connection — the shape a flat outbound-only fleet needs. Each node attaches outward to the
root by writing its per-node `up` stream and reading its per-node `down` stream on a
dedicated relay Redis endpoint; the root **rendezvous bridge** is the only party that moves
a frame between two nodes, reading a node's up stream from a durable cursor and forwarding
each frame to the peer node's down stream chosen by the session's routing record. The
bridge reads only routing and flow-control fields and forwards the payload opaquely; it
holds no admission, credit, or engine authority. Draining is fair — priority control frames
first, then data round-robined across sessions so one busy session cannot starve another.

A `rly-*` **session** frames a direction with a stable sequence and a receiver-owned
acknowledged cursor under a per-invocation byte window. The sender admits a data frame only
within the receiver's granted window and blocks once it is full, so a slow receiver
backpressures a fast producer and never holds more than its advertised window in flight; a
completion larger than the window is chunked under it rather than framed whole. Control
frames — open, acknowledgement, window grant, cancel, terminal — ride the priority lane and
bypass the data window, so a cancel is never deadlocked behind a full window. A cumulative
acknowledgement advances the window and releases only relay-window byte credit, never a
capacity credit. Recovery is a durable cursor lease rather than a consumer group: each leg
has one logical receiver that resumes from its stored cursor, and a restarted receiver
reclaims an owner-fenced lease. Unacknowledged frames are never trimmed — a stream is
trimmed only at or below the acknowledged id. The relay uses its own traffic namespace and
Redis endpoint, distinct from the event/log relay and the legacy proxy streams.

## Echo seam

A feature-gated, SYSTEM/ADMIN echo proves the substrate end to end without any resident
traffic. The control plane resolves a route to a target listener, delivers the plan to the
origin node's deputy over the trusted node-command seam, and folds the deputy's classified
observations back into the reachability view. The deputy round-trips a small payload over
the selected transport against a bounded echo sidecar — never a resident engine. See the
`Network` section of [`API.md`](API.md).

## Reuse without resident contracts

Remote external-tool carriage reuses this substrate — the pure resolver, the
`worker_direct` / `node_relay` / reverse-rendezvous transports, and the bounded frame,
cursor, and lease mechanics — over a distinct claim-free session namespace and a
control-issued nonresident sidecar target, without any resident contract: it mints no
`ServiceClaim` or `RouteAuthorization`, reuses no resident relay session, and carries only
the opaque tool operation and its result. See [`EXECUTORS.md`](EXECUTORS.md).
