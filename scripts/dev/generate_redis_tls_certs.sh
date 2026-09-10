#!/usr/bin/env bash
set -euo pipefail

TLS_DIR="${TLS_DIR:-secrets/tls/redis}"
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

CA_KEY="${TLS_DIR}/redis-ca.key"
CA_CERT="${TLS_DIR}/redis-ca.pem"
SERVER_KEY="${TLS_DIR}/redis-server.key"
SERVER_CSR="${TLS_DIR}/redis-server.csr"
SERVER_CERT="${TLS_DIR}/redis-server.pem"
SERVER_EXT="${TLS_DIR}/redis-server.ext"

rm -f "${TLS_DIR}/redis-ca.srl"

openssl genrsa -out "${CA_KEY}" 4096
openssl req -x509 -new -nodes \
  -key "${CA_KEY}" \
  -sha256 -days "${TLS_DAYS}" \
  -subj "/CN=FlowMesh Redis CA" \
  -out "${CA_CERT}"

openssl genrsa -out "${SERVER_KEY}" 2048
CN_HOST="${1:-localhost}"
openssl req -new -key "${SERVER_KEY}" \
  -subj "/CN=${CN_HOST}" \
  -out "${SERVER_CSR}"

SAN_CSV=$(IFS=,; echo "${SAN_ENTRIES[*]}")
cat > "${SERVER_EXT}" <<EOF
subjectAltName=${SAN_CSV}
extendedKeyUsage=serverAuth
EOF

openssl x509 -req \
  -in "${SERVER_CSR}" \
  -CA "${CA_CERT}" -CAkey "${CA_KEY}" \
  -CAcreateserial \
  -out "${SERVER_CERT}" \
  -days "${TLS_DAYS}" -sha256 \
  -extfile "${SERVER_EXT}"

chmod 600 "${CA_KEY}" "${SERVER_KEY}"
chmod 644 "${CA_CERT}" "${SERVER_CERT}"

echo "Generated CA: ${CA_CERT}"
echo "Generated Redis cert/key: ${SERVER_CERT} ${SERVER_KEY}"
echo "Export this for the stack; the container reads the mounted paths the"
echo "REDIS_TLS_*_FILE defaults already name:"
echo "REDIS_TLS_DIR=${TLS_DIR}"
