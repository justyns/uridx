"""Shared utilities for extractors."""

import json
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
