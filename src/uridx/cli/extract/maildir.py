"""Extract email messages from a local Maildir (e.g. one synced by mbsync/isync)."""

import multiprocessing
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer

from ._mail import JunkMode, message_record, parse_bytes, parse_duration
from uridx.timeutil import to_aware_utc


@dataclass(frozen=True)
class _Config:
    """What turning a message file into a record needs, beyond the file itself."""

    account: str
    junk_mode: JunkMode
    since: Optional[datetime]
    user_tags: list[str]


_worker_cfg: Optional[_Config] = None


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


def _to_line(item: tuple[str, str], cfg: _Config) -> str | None:
    """File path -> Record JSONL line (or None if unreadable/filtered).

    Returns the serialized line rather than a Record so pool workers ship a string
    back across the process boundary instead of pickling a model.
    """
    path, folder = item
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return None
    msg = parse_bytes(data, path)
    if msg is None:
        return None
    rec = message_record(
        msg, folder, account=cfg.account, junk_mode=cfg.junk_mode, since=cfg.since, user_tags=cfg.user_tags
    )
    return rec.model_dump_json(exclude_none=True) if rec is not None else None


def _init_worker(cfg: _Config) -> None:
    global _worker_cfg
    _worker_cfg = cfg


def _worker(item: tuple[str, str]) -> str | None:
    """Pool entry point: the config arrives once via the initializer, not per item."""
    return _to_line(item, _worker_cfg)


def _emit(lines: Iterable[str | None]) -> None:
    for line in lines:
        if line:
            print(line, flush=True)


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
    since_dt = to_aware_utc(since)
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
        cfg = _Config(account or path.name, junk, since_dt, tag or [])
        files = _iter_files(path, folder, cutoff_mtime)
        if jobs > 1:
            with multiprocessing.Pool(jobs, initializer=_init_worker, initargs=(cfg,)) as pool:
                _emit(pool.imap_unordered(_worker, files, chunksize=32))
        else:
            _emit(_to_line(item, cfg) for item in files)
