"""Shared utilities for extractors."""

import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from uridx.record import Record

T = TypeVar("T")


class MissingExtractorDependency(Exception):
    """Raised by an extractor when an optional dependency or service is unavailable.

    The `extract` command wrapper turns this into a friendly hint + exit; the `add`
    command catches it per-bucket and warns+skips so one missing dep/service doesn't
    abort a whole multi-type run.
    """


def file_uri(path: Path) -> str:
    """Canonical file:// URI for a local path."""
    return f"file://{path.resolve()}"


def get_file_mtime(path: Path) -> str:
    """Get file modification time as ISO8601 string."""
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def output(record: Record) -> None:
    """Print a record as one JSONL line (omitting unset/None fields)."""
    print(record.model_dump_json(exclude_none=True), flush=True)


def resolve_paths(paths: list[Path], extensions: set[str]) -> list[Path]:
    """Resolve a list of paths to matching files."""
    if not paths:
        paths = [Path.cwd()]

    files = []
    for p in paths:
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(f for f in p.rglob("*") if f.suffix.lower() in extensions and f.is_file())
    return files


def drop_indexed(candidates: dict[str, T], force: bool, label: Callable[[T], str] = str) -> dict[str, T]:
    """Return the candidates whose source_uri isn't indexed yet, reporting each skip.

    Keyed by source_uri, valued with whatever the extractor needs to do the work (a
    path, a database row), so it can filter before paying for extraction. `label`
    renders a value for the skip message. No-op when force.
    """
    if force or not candidates:
        return candidates
    from uridx.db.engine import init_db
    from uridx.db.operations import get_existing_source_uris

    init_db()
    indexed = get_existing_source_uris(list(candidates))
    for uri in indexed:
        print(f"Skipping {label(candidates[uri])} (already ingested)", file=sys.stderr)
    return {uri: value for uri, value in candidates.items() if uri not in indexed}


def drop_indexed_files(files: list[Path], force: bool) -> list[Path]:
    """Drop already-indexed files from a resolved file list (no-op when force)."""
    return list(drop_indexed({file_uri(f): f for f in files}, force).values())


def prepare_files(paths: list[Path], extensions: set[str], force: bool) -> list[Path]:
    """Resolve paths to matching files, dropping already-indexed ones (unless force)."""
    return drop_indexed_files(resolve_paths(paths, extensions), force)
