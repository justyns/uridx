"""Shared utilities for extractors."""

import hashlib
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeVar

import typer

if TYPE_CHECKING:
    from uridx.record import Record

T = TypeVar("T")

NoGitignore = Annotated[bool, typer.Option("--no-gitignore", help="Also index files git ignores")]
NoDefaultExcludes = Annotated[
    bool,
    typer.Option("--no-default-excludes", help="Also index node_modules, build and the other built-in excludes"),
]


def load_rules(defaults: bool) -> list[str]:
    """Stored exclude rules, plus the built-in excludes as globs when `defaults`."""
    from uridx.db.engine import get_session, init_db
    from uridx.db.operations import load_excludes
    from uridx.excludes import DEFAULT_EXCLUDES

    init_db()
    with get_session() as session:
        rules = load_excludes(session)
    return rules + ["**/" + d for d in DEFAULT_EXCLUDES] if defaults else rules


class MissingExtractorDependency(Exception):
    """Raised by an extractor when an optional dependency or service is unavailable.

    The `extract` command wrapper turns this into a friendly hint + exit; the `add`
    command catches it per-bucket and warns+skips so one missing dep/service doesn't
    abort a whole multi-type run.
    """


def file_uri(path: Path) -> str:
    """Canonical file:// URI for a local path."""
    return f"file://{path.resolve()}"


def file_hash(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get_file_mtime(path: Path) -> str:
    """Get file modification time as ISO8601 string."""
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def output(record: "Record") -> None:
    """Print a record as one JSONL line (omitting unset/None fields)."""
    print(record.model_dump_json(exclude_none=True), flush=True)


def resolve_paths(
    paths: list[Path], extensions: set[str], force: bool, no_gitignore: bool, no_default_excludes: bool
) -> list[Path]:
    """Resolve a list of paths to matching files, dropping what the exclude rules cover.

    `force` lets a file named in `paths` through the rules.
    """
    from uridx.excludes import is_excluded, iter_files

    rules = load_rules(not no_default_excludes)
    files = []
    for p in paths or [Path.cwd()]:
        if p.is_file():
            if force or not is_excluded(str(p.resolve()), rules):
                files.append(p)
        elif p.is_dir():
            files.extend(f for f in iter_files(p, rules, gitignore=not no_gitignore) if f.suffix.lower() in extensions)
    return files


def drop_indexed(
    candidates: dict[str, T],
    force: bool,
    label: Callable[[T], str] = str,
    version: Callable[[T], str] | None = None,
) -> dict[str, T]:
    """Return the candidates that need (re-)extracting, reporting each skip.

    Keyed by source_uri, valued with whatever the extractor needs to do the work (a
    path, a database row), so it can filter before paying for extraction. `label`
    renders a value for the skip message. With `version`, an indexed candidate is kept
    when its stored source_hash differs from that version. A NULL source_hash (an item
    indexed before uridx stored one) counts as current. No-op when force.
    """
    if force or not candidates:
        return candidates
    from uridx.db.engine import init_db
    from uridx.db.operations import get_source_hashes

    init_db()
    stored = get_source_hashes(list(candidates))
    fresh = {}
    for uri, value in candidates.items():
        if uri in stored and (version is None or stored[uri] is None or stored[uri] == version(value)):
            print(f"Skipping {label(value)} (already ingested)", file=sys.stderr)
        else:
            fresh[uri] = value
    return fresh


def drop_indexed_files(files: list[Path], force: bool) -> list[Path]:
    """Drop files already indexed with the same bytes (no-op when force)."""
    return list(drop_indexed({file_uri(f): f for f in files}, force, version=file_hash).values())


def prepare_files(
    paths: list[Path], extensions: set[str], force: bool, no_gitignore: bool, no_default_excludes: bool
) -> list[Path]:
    """Resolve paths to matching files, dropping already-indexed ones (unless force)."""
    return drop_indexed_files(resolve_paths(paths, extensions, force, no_gitignore, no_default_excludes), force)
