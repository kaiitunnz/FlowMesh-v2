#!/usr/bin/env bash
# Issue the CA and node identities a trusted direct offload is carried over.
#
# Each node both dials an offload target and serves one, so its identity carries
# clientAuth and serverAuth. The cluster gRPC CA cannot be reused as-is: it issues
# server-only identities, which authenticate a listener but never prove which node is
# dialing it.
#
# A dialing origin verifies that its target's certificate covers the host it dialed, so
# pass the node's advertised offload address (NETWORK_PLANE_OFFLOAD_NODE_LISTENER_URL's
# host, and the worker listener host) as extra SANs — a certificate that omits it fails
# the handshake and the offload falls back to the relay.
#
# Usage: generate_offload_tls_certs.sh <node-name> [extra-san ...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <node-name> [extra-san ...]" >&2
  exit 2
fi

TLS_DIR="${TLS_DIR:-secrets/tls/offload}"
TLS_DAYS="${TLS_DAYS:-3650}"
NODE_NAME="$1"
shift

declare -a SAN_ENTRIES=()

add_san_entry() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    return
  fi
  if [[ "${value}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    SAN_ENTRIES+=("IP:${value}")
  else
    SAN_ENTRIES+=("DNS:${value}")
  fi
}

# The node name is the identity a peer matches, so it is always a SAN.
add_san_entry "${NODE_NAME}"
for entry in "$@"; do
  add_san_entry "${entry}"
done
add_san_entry "localhost"
add_san_entry "127.0.0.1"

mkdir -p "${TLS_DIR}"

CA_KEY="${TLS_DIR}/offload-ca.key"
CA_CERT="${TLS_DIR}/offload-ca.pem"
NODE_KEY="${TLS_DIR}/${NODE_NAME}.key"
NODE_CSR="${TLS_DIR}/${NODE_NAME}.csr"
NODE_CERT="${TLS_DIR}/${NODE_NAME}.pem"
NODE_EXT="${TLS_DIR}/${NODE_NAME}.ext"

rm -f "${TLS_DIR}/offload-ca.srl"

# Reuse the CA across nodes so every identity it issues verifies against one bundle.
if [[ ! -f "${CA_CERT}" ]]; then
  openssl genrsa -out "${CA_KEY}" 4096
  openssl req -x509 -new -nodes \
    -key "${CA_KEY}" \
    -sha256 -days "${TLS_DAYS}" \
    -subj "/CN=FlowMesh Offload CA" \
    -out "${CA_CERT}"
  chmod 600 "${CA_KEY}"
  chmod 644 "${CA_CERT}"
fi

openssl genrsa -out "${NODE_KEY}" 2048
openssl req -new -key "${NODE_KEY}" \
  -subj "/CN=${NODE_NAME}" \
  -out "${NODE_CSR}"

SAN_CSV=$(IFS=,; echo "${SAN_ENTRIES[*]}")
cat > "${NODE_EXT}" <<EOF
subjectAltName=${SAN_CSV}
extendedKeyUsage=clientAuth,serverAuth
EOF

openssl x509 -req \
  -in "${NODE_CSR}" \
  -CA "${CA_CERT}" -CAkey "${CA_KEY}" \
  -CAcreateserial \
  -out "${NODE_CERT}" \
  -days "${TLS_DAYS}" -sha256 \
  -extfile "${NODE_EXT}"

chmod 600 "${NODE_KEY}"
chmod 644 "${NODE_CERT}"

echo "Generated CA: ${CA_CERT}"
echo "Generated node cert/key: ${NODE_CERT} ${NODE_KEY}"
echo "Export these on ${NODE_NAME}:"
echo "NETWORK_PLANE_OFFLOAD_TLS_CA_FILE=${CA_CERT}"
echo "NETWORK_PLANE_OFFLOAD_TLS_CERT_FILE=${NODE_CERT}"
echo "NETWORK_PLANE_OFFLOAD_TLS_KEY_FILE=${NODE_KEY}"
