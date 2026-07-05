"""Extract email messages from mbox archives (Gmail Takeout, Thunderbird, list archives)."""

import mailbox
import sys
from collections.abc import Iterator
from email.message import EmailMessage
from pathlib import Path
from typing import Annotated, Optional

import typer

from ._mail import JunkMode, iter_message_records, parse_bytes, parse_since
from .base import output


def _iter_messages(path: Path) -> Iterator[tuple[EmailMessage, str]]:
    """Yield (message, folder) for every message in an mbox file."""
    box = mailbox.mbox(str(path))
    try:
        for key in box.iterkeys():
            try:
                data = box.get_bytes(key)
            except Exception as e:
                print(f"Error reading message {key} in {path}: {e}", file=sys.stderr)
                continue
            if (msg := parse_bytes(data, f"message {key} in {path}")) is not None:
                yield msg, path.stem
    finally:
        box.close()


def extract(
    paths: Annotated[list[Path], typer.Argument(help="mbox files")],
    account: Annotated[
        Optional[str], typer.Option("--account", "-a", help="Account label (default: filename stem)")
    ] = None,
    junk: Annotated[JunkMode, typer.Option("--junk", help="Junk handling")] = JunkMode.skip,
    since: Annotated[
        Optional[str], typer.Option("--since", "-s", help="Only mail on/after this date (YYYY-MM-DD)")
    ] = None,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
):
    """Extract email messages from one or more mbox files."""
    try:
        since_dt = parse_since(since)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(1)

    for path in paths:
        if not path.is_file():
            print(f"mbox file not found: {path}", file=sys.stderr)
            continue
        for rec in iter_message_records(
            _iter_messages(path),
            account=account or path.stem,
            junk_mode=junk,
            since=since_dt,
            user_tags=tag,
        ):
            output(rec)
