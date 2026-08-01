"""Extract email messages from a local Maildir (e.g. one synced by mbsync/isync)."""

import multiprocessing
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer

from ._mail import JunkMode, message_record, parse_bytes, parse_duration
from .base import as_utc

_worker_cfg = None


def _iter_files(root: Path, folder_filter: Optional[str], cutoff_mtime: Optional[float]) -> Iterator[tuple[str, str]]:
    """Yield (path, folder) for every Maildir message file under root's cur/ and new/."""
    # Walk cur/new ourselves rather than mailbox.Maildir: its folder API assumes Maildir++
    # (.dot-prefixed) names and misses mbsync's verbatim nested folders, and it yields legacy
    # message objects instead of letting us parse with email.policy.default.
    base = root / folder_filter if folder_filter and folder_filter != "INBOX" else root
    if not base.is_dir():
        print(f"Folder not found: {base}", file=sys.stderr)
        return
    for f in base.rglob("*"):
        if not f.is_file() or f.parent.name not in ("cur", "new"):
            continue
        if cutoff_mtime is not None and f.stat().st_mtime < cutoff_mtime:
            continue
        rel = f.parent.parent.relative_to(root)
        folder = "INBOX" if str(rel) == "." else str(rel)
        if folder_filter and folder != folder_filter:
            continue
        yield str(f), folder


def _init_worker(account: str, junk_mode: JunkMode, since, user_tags: list[str]) -> None:
    global _worker_cfg
    _worker_cfg = (account, junk_mode, since, user_tags)


def _extract_one(item: tuple[str, str]) -> str | None:
    """File path -> Record JSONL line (or None if unreadable/filtered). Runs in workers."""
    path, folder = item
    account, junk_mode, since, user_tags = _worker_cfg
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return None
    msg = parse_bytes(data, path)
    if msg is None:
        return None
    rec = message_record(msg, folder, account=account, junk_mode=junk_mode, since=since, user_tags=user_tags)
    return rec.model_dump_json(exclude_none=True) if rec is not None else None


def extract(
    paths: Annotated[list[Path], typer.Argument(help="Maildir root directories")],
    account: Annotated[
        Optional[str], typer.Option("--account", "-a", help="Account label (default: directory name)")
    ] = None,
    folder: Annotated[
        Optional[str], typer.Option("--folder", help="Only this folder, relative to the Maildir root")
    ] = None,
    junk: Annotated[JunkMode, typer.Option("--junk", help="Junk handling")] = JunkMode.skip,
    since: Annotated[
        Optional[datetime],
        typer.Option("--since", "-s", formats=["%Y-%m-%d"], help="Only mail dated on/after this date"),
    ] = None,
    newer_than: Annotated[
        Optional[str],
        typer.Option(
            "--newer-than", help="Only files modified within this window (e.g. 30m, 2h, 7d) for incremental sync"
        ),
    ] = None,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
    jobs: Annotated[int, typer.Option("--jobs", "-j", help="Parallel parse workers")] = 1,
):
    """Extract email messages from a local Maildir."""
    since_dt = as_utc(since)
    try:
        window = parse_duration(newer_than)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(1)
    cutoff_mtime = time.time() - window if window is not None else None

    for path in paths:
        if not path.is_dir():
            print(f"Maildir not found: {path}", file=sys.stderr)
            continue
        cfg = (account or path.name, junk, since_dt, tag or [])
        files = _iter_files(path, folder, cutoff_mtime)
        if jobs > 1:
            with multiprocessing.Pool(jobs, initializer=_init_worker, initargs=cfg) as pool:
                lines = pool.imap_unordered(_extract_one, files, chunksize=32)
                for line in lines:
                    if line:
                        print(line, flush=True)
        else:
            _init_worker(*cfg)
            for item in files:
                if line := _extract_one(item):
                    print(line, flush=True)
