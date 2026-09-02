# Network plane

The network plane is the topology-aware, control-resolved routing substrate. It turns
trusted endpoint advertisements and directional reachability evidence into an ordered,
expiry-bounded route between a caller and a target listener, and it carries bytes over one
of three transports. It is control-resolved and data-direct: the resolver runs in the
control plane and the origin-side deputy executes only the resolved candidate ladder —
workers never scan addresses or discover peers.

The plane is a routing substrate only. It carries no resident traffic, exposes no resident
engine listener, and never mints a `ServiceClaim`, releases a credit, or issues a
`RouteAuthorization`. A route observation is network evidence; it can never promote,
release, or overwrite a capacity credit. Enable it with `NETWORK_PLANE_ENABLED=true`
(`docs/ENV.md` lists the knobs).

## The four facts

Route resolution is derived from four separate facts, each with its own owner and fence.

- **`NetworkEndpointAdvertisement`** — an operator-configured, identity/TLS-bound node (or
  registered ingress edge) network-plane endpoint: a purpose-scoped URL, an endpoint
  `generation`, trust domain, reachability class, and protocol capability. It rides on
  node registration and is carried on the node record, so the node registry is its durable
  carrier. It is not a worker-supplied arbitrary URL and not the generic server-management
  endpoint. Re-registration mints a fresh generation, and route evidence bound to a
  superseded generation is invalidated.
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

The resolver orders three transport candidates; the deputy tries them in order until one
round-trips, and a demoted direct or node path drops out until its backoff cools.

- **`worker_direct`** — caller to the listener. Legal only for an explicitly directly
  routable listener whose endpoint class the origin's network class can reach, under a
  bounded optimistic connect budget. Shared-node placement alone does not make it legal.
- **`node_relay`** — caller to the target node's announced endpoint, which uplinks over an
  authenticated node-local relay session to the target listener the route names. It is the
  initial same-node path as well as the normal cross-node path.
- **`control_relay`** — the controlled fallback: a bounded relay session through the root's
  relay endpoint and, when the target node advertises one, its node endpoint to the target
  listener.

Each relay hop is target-addressed: it reads one leading frame naming its next hop, dials
it, and byte-relays the rest, so a route reaches whichever listener its hops name and a
multi-hop ladder chains through the relays. The origin deputy writes a frame per hop after
the first; a direct route writes none.

A `RelaySession` bridges two stream pairs with a bounded in-flight buffer, so a slow
consumer backpressures a fast producer; it is cancellable and self-cleaning. It is a
dedicated network-plane mechanism, distinct from the event/log relay.

## Echo seam

A feature-gated, SYSTEM/ADMIN echo proves the substrate end to end without any resident
traffic. The control plane resolves a route to a target listener, delivers the plan to the
origin node's deputy over the trusted node-command seam, and folds the deputy's classified
observations back into the reachability view. The deputy round-trips a small payload over
the selected transport against a bounded echo sidecar — never a resident engine. See the
`Network` section of [`API.md`](API.md).
