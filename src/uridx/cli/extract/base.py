"""Shared utilities for extractors."""

import sys
from datetime import datetime, timezone
from pathlib import Path

from uridx.record import Record


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
    print(record.model_dump_json(exclude_none=True))


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


def filter_existing(source_uri_map: dict, force: bool, label=lambda v: v) -> None:
    """Drop already-ingested URIs from source_uri_map in place (no-op when force)."""
    if force or not source_uri_map:
        return
    from uridx.db.engine import init_db
    from uridx.db.operations import get_existing_source_uris

    init_db()
    for uri in get_existing_source_uris(list(source_uri_map.keys())):
        print(f"Skipping {label(source_uri_map[uri])} (already ingested)", file=sys.stderr)
        del source_uri_map[uri]


def filter_existing_files(files: list[Path], force: bool) -> list[Path]:
    """Drop already-ingested files from a resolved file list (no-op when force)."""
    source_map = {file_uri(f): f for f in files}
    filter_existing(source_map, force)
    return list(source_map.values())


def prepare_files(paths: list[Path], extensions: set[str], force: bool) -> list[Path]:
    """Resolve paths to matching files, dropping already-ingested ones (unless force)."""
    return filter_existing_files(resolve_paths(paths, extensions), force)
