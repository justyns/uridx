"""Shared utilities for extractors."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_file_mtime(path: Path) -> str:
    """Get file modification time as ISO8601 string."""
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def output(record: dict) -> None:
    """Output a single JSONL record to stdout."""
    print(json.dumps(record))


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
