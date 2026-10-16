"""Custody helpers for tests that stage artifact files."""

from pathlib import Path

from shared.content import ContentReference
from worker.content.custody import ArtifactCustody


def stage(
    custody: ArtifactCustody, scope: str, dispatch_id: str, source: Path
) -> ContentReference:
    """Stage the file at ``source`` in a dispatch's custody."""
    with source.open("rb") as opened:
        return custody.stage(scope, dispatch_id, opened)
