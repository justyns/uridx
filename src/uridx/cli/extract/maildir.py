"""Extract email messages from a local Maildir (e.g. one synced by mbsync/isync)."""

import sys
import time
from collections.abc import Iterator
from email.message import EmailMessage
from pathlib import Path
from typing import Annotated, Optional

import typer

from ._mail import JunkMode, iter_message_records, parse_bytes, parse_duration, parse_since
from .base import output


def _iter_messages(
    root: Path, folder_filter: Optional[str], cutoff_mtime: Optional[float]
) -> Iterator[tuple[EmailMessage, str]]:
    """Yield (message, folder) for every Maildir message under root's cur/ and new/."""
    # Walk cur/new ourselves rather than mailbox.Maildir: its folder API assumes Maildir++
    # (.dot-prefixed) names and misses mbsync's verbatim nested folders, and it yields legacy
    # message objects instead of letting us parse with email.policy.default.
    for f in root.rglob("*"):
        if not f.is_file() or f.parent.name not in ("cur", "new"):
            continue
        if cutoff_mtime is not None and f.stat().st_mtime < cutoff_mtime:
            continue
        rel = f.parent.parent.relative_to(root)
        folder = "INBOX" if str(rel) == "." else str(rel)
        if folder_filter and folder != folder_filter:
            continue
        try:
            data = f.read_bytes()
        except OSError as e:
            print(f"Error reading {f}: {e}", file=sys.stderr)
            continue
        if (msg := parse_bytes(data, str(f))) is not None:
            yield msg, folder


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
        Optional[str], typer.Option("--since", "-s", help="Only mail dated on/after this date (YYYY-MM-DD)")
    ] = None,
    newer_than: Annotated[
        Optional[str],
        typer.Option(
            "--newer-than", help="Only files modified within this window (e.g. 30m, 2h, 7d) for incremental sync"
        ),
    ] = None,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
):
    """Extract email messages from a local Maildir."""
    try:
        since_dt = parse_since(since)
        window = parse_duration(newer_than)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(1)
    cutoff_mtime = time.time() - window if window is not None else None

    for path in paths:
        if not path.is_dir():
            print(f"Maildir not found: {path}", file=sys.stderr)
            continue
        for rec in iter_message_records(
            _iter_messages(path, folder, cutoff_mtime),
            account=account or path.name,
            junk_mode=junk,
            since=since_dt,
            user_tags=tag,
        ):
            output(rec)
