"""Fence this process, then become the sandboxed command.

Run as ``python _launcher.py <spec-json> <program> [args...]``. The fence installs in a
freshly started, single-threaded interpreter and is inherited across ``execv``, which is
why it is a separate entry point rather than a pre-exec hook: installing a seccomp
filter between fork and exec can deadlock a threaded worker.

The fence is: a Landlock ruleset that denies every path outside the activation's own
workspace and the read-only runtime, a seccomp filter that denies IP sockets and
io_uring, and the resource limits the episode's envelope declares. Each layer fails
closed — an error installing one aborts before the command runs. A spec whose ``egress``
is set relaxes the two network layers, and only those: it is installed for a command
whose capability was minted with the opt-in, never chosen here.
"""

import ctypes
import json
import os
import resource
import struct
import sys

_NR = {
    "x86_64": {
        "audit": 0xC000003E,
        "socket": 41,
        "seccomp": 317,
        "io_uring_setup": 425,
        "landlock_create": 444,
        "landlock_add": 445,
        "landlock_restrict": 446,
    },
    "aarch64": {
        "audit": 0xC00000B7,
        "socket": 198,
        "seccomp": 277,
        "io_uring_setup": 425,
        "landlock_create": 444,
        "landlock_add": 445,
        "landlock_restrict": 446,
    },
}

_PR_SET_NO_NEW_PRIVS = 38
_LANDLOCK_RULE_PATH_BENEATH = 1
# Read, list, and execute; a read-only root grants exactly these.
_FS_READ = 0x1 | 0x4 | 0x8
# Read and write one device node, which an interpreter needs to seed and to discard.
_FS_DEVICE = 0x2 | 0x4
# Every filesystem right each ABI knows, so anything not granted below is denied.
_FS_HANDLED = {1: 0x1FFF, 2: 0x3FFF, 3: 0x7FFF, 4: 0x7FFF, 5: 0xFFFF}
_NET_HANDLED = 0x3  # bind and connect over TCP


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def _fail(message: str) -> None:
    raise OSError(ctypes.get_errno(), message)


def _landlock(libc: ctypes.CDLL, nr: dict[str, int], spec: dict) -> None:
    """Restrict the filesystem to the workspace plus a read-only runtime.

    Networking is handled by omission: declaring the TCP rights without granting a rule
    denies them. An egress-authorized command declares no network rights at all, which
    leaves the filesystem confinement untouched.
    """
    abi = int(spec["landlock_abi"])
    if abi < 1:
        return
    handled_fs = _FS_HANDLED[min(abi, max(_FS_HANDLED))]
    if abi >= 4 and not spec["egress"]:
        attr = struct.pack("QQ", handled_fs, _NET_HANDLED)
    else:
        attr = struct.pack("Q", handled_fs)
    buf = ctypes.create_string_buffer(attr, len(attr))
    ctypes.set_errno(0)
    fd = libc.syscall(nr["landlock_create"], ctypes.byref(buf), len(attr), 0)
    if fd < 0:
        _fail("cannot create the sandbox filesystem ruleset")
    try:
        for path, access in (
            *((p, handled_fs) for p in spec["rw"]),
            *((p, _FS_READ) for p in spec["ro"]),
            *((p, _FS_DEVICE) for p in spec["devices"]),
        ):
            try:
                parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue  # a root this image does not have grants nothing
            try:
                rule = ctypes.create_string_buffer(
                    struct.pack("=Qi", access & handled_fs, parent), 12
                )
                ctypes.set_errno(0)
                if libc.syscall(
                    nr["landlock_add"],
                    ctypes.c_int(fd),
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(rule),
                    0,
                ):
                    _fail(f"cannot grant the sandbox access to {path}")
            finally:
                os.close(parent)
        ctypes.set_errno(0)
        if libc.syscall(nr["landlock_restrict"], ctypes.c_int(fd), 0):
            _fail("cannot restrict the sandbox to its workspace")
    finally:
        os.close(fd)


def _filter_program(nr: dict[str, int], egress: bool) -> bytes:
    """The classic-BPF seccomp program: deny io_uring, and IP sockets unless authorized.

    A ring is denied in both modes because its operations run in kernel context and are
    not re-checked against this filter. Unix sockets and the filesystem are untouched.

    Denying by socket domain is the whole of the network fence, so lifting it lifts
    every destination at once: seccomp cannot dereference a ``sockaddr``, and no later
    syscall carries the address in a register. An egress-authorized command therefore
    reaches whatever the worker and kernel reach.
    """
    ld, jeq, ret = 0x20, 0x15, 0x06
    deny, allow = 0x00050000 | 13, 0x7FFF0000

    def stmt(code: int, k: int) -> bytes:
        return struct.pack("HBBI", code, 0, 0, k)

    def jump(k: int, jt: int, jf: int) -> bytes:
        return struct.pack("HBBI", jeq, jt, jf, k)

    socket_filter = (
        []
        if egress
        else [
            jump(nr["socket"], 0, 4),  # anything but socket() runs
            stmt(ld, 16),  # socket() domain
            jump(2, 1, 0),  # AF_INET
            jump(10, 0, 1),  # AF_INET6
        ]
    )
    # Every jump below targets the deny that ends the body, so the distances move with
    # the socket filter. Its false branch falls through to that filter when there is one
    # and must otherwise step over the deny, or an egress-authorized command would be
    # denied every syscall it makes.
    body = [
        stmt(ld, 0),  # seccomp_data.nr
        jump(nr["io_uring_setup"], len(socket_filter), 0 if socket_filter else 1),
        *socket_filter,
        stmt(ret, deny),
        stmt(ret, allow),
    ]
    return b"".join(
        [
            stmt(ld, 4),  # seccomp_data.arch
            jump(nr["audit"], 0, len(body) - 2),  # a foreign arch denies everything
            *body,
        ]
    )


def _seccomp(libc: ctypes.CDLL, nr: dict[str, int], egress: bool) -> None:
    """Install the syscall filter this command runs under."""
    prog = _filter_program(nr, egress)

    class _Prog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    buf = ctypes.create_string_buffer(prog, len(prog))
    fprog = _Prog(len(prog) // 8, ctypes.cast(buf, ctypes.c_void_p))
    libc.syscall.argtypes = [
        ctypes.c_long,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    ctypes.set_errno(0)
    if libc.syscall(nr["seccomp"], 1, 0, ctypes.byref(fprog)):
        _fail("cannot install the sandbox syscall filter")


def _limits(spec: dict) -> None:
    # No RLIMIT_NPROC: it bounds the whole uid rather than this command, and every
    # activation on a worker shares one uid, so a per-command process cap needs the
    # cgroup delegation an unprivileged container does not have.
    for limit, value in (
        (resource.RLIMIT_AS, spec["memory_bytes"]),
        (resource.RLIMIT_CPU, spec["cpu_seconds"]),
        (resource.RLIMIT_FSIZE, spec["file_size_bytes"]),
        (resource.RLIMIT_NOFILE, spec["open_files"]),
    ):
        resource.setrlimit(limit, (value, value))


def main() -> None:
    spec = json.loads(sys.argv[1])
    program, argv = sys.argv[2], sys.argv[2:]
    machine = os.uname().machine
    if (nr := _NR.get(machine)) is None:
        raise OSError(f"the sandbox does not know the syscalls of {machine}")
    libc = _libc()
    # Both Landlock and seccomp refuse an unprivileged caller that can still gain
    # privilege through a setuid exec, so this comes first or neither installs.
    ctypes.set_errno(0)
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0):
        _fail("cannot drop privilege escalation")
    _landlock(libc, nr, spec)
    _seccomp(libc, nr, spec["egress"])
    _limits(spec)
    os.execv(program, argv)  # nosec B606 - argv list, no shell, absolute program


if __name__ == "__main__":
    main()
