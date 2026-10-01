#!/bin/sh
# FlowMesh SSH session entrypoint.
#
# Reads SSH_USER and AUTHORIZED_KEYS from the environment, sets up the user
# and authorized_keys file, then starts sshd in the foreground.
# FLOWMESH_STAGED_INPUT_SPECS entries are newline-separated
# "mount_path<TAB>target_path" pairs used to materialize staged read-only
# input views into the container.
# FLOWMESH_CREATE_DIRS is a newline-separated list of directories to create.
# FLOWMESH_PERMIT_ENV is a comma-separated list of the env names a login receives.
set -e

# SSH host keys
ssh-keygen -A -q

# Session user
SSH_USER="${SSH_USER:-flowmesh}"
SSH_UID="${SSH_UID:-10001}"
SSH_GID="${SSH_GID:-10001}"
FLOWMESH_FINISH_SENTINEL="${FLOWMESH_FINISH_SENTINEL:-/tmp/.flowmesh_finish}"

# Create group and user if they don't exist
if ! getent group "$SSH_GID" >/dev/null 2>&1; then
    groupadd -g "$SSH_GID" "$SSH_USER"
fi

if ! id "$SSH_USER" >/dev/null 2>&1; then
    useradd -m -u "$SSH_UID" -g "$SSH_GID" -s /bin/bash "$SSH_USER"
fi

HOME_DIR="$(getent passwd "$SSH_USER" | cut -d: -f6)"

# Authorized keys
mkdir -p "$HOME_DIR/.ssh"
# Always write fresh — containers are ephemeral so no stale keys should exist,
# but writing from scratch avoids any confusion if the base image has leftovers.
if [ -n "$AUTHORIZED_KEYS" ]; then
    printf '%s\n' "$AUTHORIZED_KEYS" > "$HOME_DIR/.ssh/authorized_keys"
else
    : > "$HOME_DIR/.ssh/authorized_keys"
fi
chmod 700 "$HOME_DIR/.ssh"
chmod 600 "$HOME_DIR/.ssh/authorized_keys"
chown -R "$SSH_USER:$SSH_USER" "$HOME_DIR/.ssh"

# Optional FlowMesh mount helpers
if [ -n "${FLOWMESH_STAGED_INPUT_SPECS:-}" ]; then
    printf '%s\n' "$FLOWMESH_STAGED_INPUT_SPECS" | while IFS="$(printf '\t')" read -r mount_path target_path; do
        [ -n "$mount_path" ] || continue
        [ -n "$target_path" ] || continue
        if [ -e "$mount_path" ] && [ ! -d "$mount_path" ]; then
            echo "Refusing to replace existing non-directory path: $mount_path" >&2
            exit 1
        fi
        mkdir -p "$mount_path"
        cp -a "$target_path"/. "$mount_path"/
        chown -R root:root "$mount_path"
        chmod -R a-w "$mount_path"
    done
fi

if [ -n "${FLOWMESH_CREATE_DIRS:-}" ]; then
    printf '%s\n' "$FLOWMESH_CREATE_DIRS" | while IFS= read -r dir_path; do
        [ -n "$dir_path" ] || continue
        mkdir -p "$dir_path"
        chown -R "$SSH_USER:$SSH_GID" "$dir_path"
    done
fi

# In-session helper to finish the SSH task successfully.
cat > /usr/local/bin/flowmesh-finish << EOF
#!/bin/sh
set -e
touch "$FLOWMESH_FINISH_SENTINEL"
echo "FlowMesh finish requested; the SSH session will close shortly."
EOF
chmod 755 /usr/local/bin/flowmesh-finish

# sshd config
cat > /etc/ssh/sshd_config.d/flowmesh.conf << 'EOF'
PasswordAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
PermitRootLogin no
PrintMotd no
AllowTcpForwarding no
X11Forwarding no
AllowAgentForwarding no
GatewayPorts no
ClientAliveInterval 60
ClientAliveCountMax 3
EOF

# The task's env rides authorized_keys as environment= options; sshd honours only
# the names listed here, which the worker validated as plain identifiers.
if [ -n "${FLOWMESH_PERMIT_ENV:-}" ]; then
    case "$FLOWMESH_PERMIT_ENV" in
        *[!A-Za-z0-9_,]*)
            echo "Refusing session env names: $FLOWMESH_PERMIT_ENV" >&2
            exit 1
            ;;
    esac
    echo "PermitUserEnvironment $FLOWMESH_PERMIT_ENV" >> /etc/ssh/sshd_config.d/flowmesh.conf
fi

# Start sshd
exec /usr/sbin/sshd -D -e
