#!/usr/bin/env bash
set -euo pipefail

TLS_DIR="${TLS_DIR:-secrets/tls/server}"
TLS_DAYS="${TLS_DAYS:-3650}"

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

for entry in "$@"; do
  add_san_entry "${entry}"
done

add_san_entry "localhost"
add_san_entry "127.0.0.1"

mkdir -p "${TLS_DIR}"

# File layout for CA and server gRPC certs.
CA_KEY="${TLS_DIR}/server-ca.key"
CA_CERT="${TLS_DIR}/server-ca.pem"
SERVER_KEY="${TLS_DIR}/server.key"
SERVER_CSR="${TLS_DIR}/server.csr"
SERVER_CERT="${TLS_DIR}/server.pem"
SERVER_EXT="${TLS_DIR}/server.ext"

# Drop stale serial to avoid reusing CA serial state.
rm -f "${TLS_DIR}/server.srl"

# Generate CA key/cert (self-signed).
openssl genrsa -out "${CA_KEY}" 4096
openssl req -x509 -new -nodes \
  -key "${CA_KEY}" \
  -sha256 -days "${TLS_DAYS}" \
  -subj "/CN=FlowMesh Internal CA" \
  -out "${CA_CERT}"

# Generate server gRPC key + CSR.
openssl genrsa -out "${SERVER_KEY}" 2048
CN_HOST="${1:-localhost}"
openssl req -new -key "${SERVER_KEY}" \
  -subj "/CN=${CN_HOST}" \
  -out "${SERVER_CSR}"

# Add SANs for localhost and any provided hosts.
SAN_CSV=$(IFS=,; echo "${SAN_ENTRIES[*]}")
cat > "${SERVER_EXT}" <<EOF
subjectAltName=${SAN_CSV}
extendedKeyUsage=serverAuth
EOF

# Sign server CSR with the internal CA.
openssl x509 -req \
  -in "${SERVER_CSR}" \
  -CA "${CA_CERT}" -CAkey "${CA_KEY}" \
  -CAcreateserial \
  -out "${SERVER_CERT}" \
  -days "${TLS_DAYS}" -sha256 \
  -extfile "${SERVER_EXT}"

# Restrict private keys; keep certs readable.
chmod 600 "${CA_KEY}" "${SERVER_KEY}"
chmod 644 "${CA_CERT}" "${SERVER_CERT}"

echo "Generated CA: ${CA_CERT}"
echo "Generated server cert/key: ${SERVER_CERT} ${SERVER_KEY}"
echo "Export this for the server; the container reads the mounted paths the"
echo "SERVER_GRPC_TLS_*_FILE defaults already name:"
echo "SERVER_TLS_DIR=${TLS_DIR}"
