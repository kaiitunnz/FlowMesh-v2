"""Sealing a component tree into the content identity a manifest records."""

import hashlib
from pathlib import Path

from .attachment import PrivateStateUnavailable, PrivateStateUnavailableReason
from .manifest import SealedComponent, StateComponentKind, component_spec

_CHUNK = 1 << 20


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def seal_component(
    kind: StateComponentKind, root: Path, *, reference_id: str
) -> SealedComponent:
    """Seal a component tree, deriving its content identity from the tree itself.

    The digest covers every regular file's relative path and contents in canonical
    order, so the same tree seals identically on any holder and any edit after the seal
    is detectable. A symlink is never followed, so a link planted in the tree cannot
    draw state the reference does not own into the seal; replacing a sealed file with
    one still changes the digest.
    """
    tree = hashlib.sha256()
    total = 0
    entries = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).parts):
        if path.is_symlink() or not path.is_file():
            continue
        digest, size = _file_digest(path)
        tree.update(f"{path.relative_to(root).as_posix()}\0{size}\0{digest}\n".encode())
        total += size
        entries += 1
    return SealedComponent(
        kind=kind,
        schema_version=component_spec(kind).schema_version,
        content_digest=tree.hexdigest(),
        size_bytes=total,
        entry_count=entries,
    )


def verify_component(sealed: SealedComponent, root: Path, *, reference_id: str) -> None:
    """Fail closed unless a tree carries exactly the sealed component."""
    if not root.is_dir():
        raise PrivateStateUnavailable(
            PrivateStateUnavailableReason.COMPONENT_MISSING,
            f"{sealed.kind.value} is not materialized",
            reference_id=reference_id,
        )
    observed = seal_component(sealed.kind, root, reference_id=reference_id)
    if observed.content_digest != sealed.content_digest:
        raise PrivateStateUnavailable(
            PrivateStateUnavailableReason.COMPONENT_MISMATCH,
            f"{sealed.kind.value} does not match its sealed generation",
            reference_id=reference_id,
        )
