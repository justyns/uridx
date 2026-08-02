"""Extract Tsugite chat history from its SQLite backend (history.db).

Tsugite stores sessions in a single SQLite database (event-sourced: a `sessions`
row plus an `events` stream). We read it directly so uridx stays standalone, and
reconstruct User/Assistant turns from `user_input` (data.text) and `model_response`
(data.raw_content) events the same way tsugite renders a conversation.
"""

import json
import os
import sqlite3
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer

from uridx.record import Record

from ._turns import Role, build_turn_chunks
from .base import drop_indexed, output

DEFAULT_DB = Path.home() / ".local" / "share" / "tsugite" / "history" / "history.db"


def _resolve_db(path: Optional[Path]) -> Path:
    """Resolve the history.db path: explicit arg (a .db file or its dir), env, or XDG default."""
    if path is not None:
        return path / "history.db" if path.is_dir() else path
    env = os.getenv("TSUGITE_HISTORY_DB")
    return Path(env) if env else DEFAULT_DB


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _iter_turns(conn: sqlite3.Connection, session_id: str) -> Iterator[tuple[Role, str]]:
    """Reduce a session's event stream to the (role, text) pairs build_turn_chunks consumes.

    Ordered by event id, which is how tsugite itself renders a conversation.
    """
    rows = conn.execute(
        "SELECT type, data FROM events WHERE session_id=? AND type IN ('user_input', 'model_response') ORDER BY id",
        (session_id,),
    )
    for row in rows:
        data = json.loads(row["data"])
        if row["type"] == "user_input":
            yield "user", (data.get("text") or "").strip()
        else:  # model_response
            yield "assistant", (data.get("raw_content") or "").strip()


def extract(
    path: Annotated[
        Optional[Path], typer.Argument(help="history.db file or its directory (default: XDG tsugite history)")
    ] = None,
    agent: Annotated[Optional[str], typer.Option("--agent", "-a", help="Filter by agent name")] = None,
    since: Annotated[
        Optional[datetime],
        typer.Option("--since", "-s", formats=["%Y-%m-%d"], help="Only sessions created on/after this date"),
    ] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all sessions")] = False,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
):
    """Extract Tsugite chat history from its SQLite backend (history.db)."""
    db_path = _resolve_db(path)
    if not db_path.is_file():
        print(f"Tsugite history database not found: {db_path}", file=sys.stderr)
        raise typer.Exit(1)

    conn = _connect_ro(db_path)

    clauses, params = [], []
    if agent:
        clauses.append("agent = ?")
        params.append(agent)
    if since:
        clauses.append("created_at >= ?")  # created_at is ISO-8601 UTC, so string compare is chronological
        params.append(since.strftime("%Y-%m-%d"))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sessions = conn.execute(
        f"SELECT session_id, agent, model, workspace, created_at, ended_at, status FROM sessions{where} "
        "ORDER BY created_at DESC",
        params,
    ).fetchall()

    source_uri_map: dict[str, sqlite3.Row] = {}
    for s in sessions:
        source_uri_map[f"tsugite://{s['agent'] or 'unknown'}/{s['session_id']}"] = s
    source_uri_map = drop_indexed(source_uri_map, force, label=lambda s: s["session_id"])
    if not source_uri_map:
        return

    for source_uri, s in source_uri_map.items():
        chunks = build_turn_chunks(_iter_turns(conn, s["session_id"]))
        if not chunks:
            continue

        agent_name = s["agent"] or "unknown"
        metadata = {
            "agent": s["agent"],
            "model": s["model"],
            "workspace": s["workspace"],
            "created_at": s["created_at"],
            "ended_at": s["ended_at"],
            "status": s["status"],
        }
        output(
            Record(
                source_uri=source_uri,
                chunks=chunks,
                tags=[agent_name, "conversation", "tsugite"] + (tag or []),
                title=f"{agent_name}: {s['session_id']}",
                source_type="tsugite",
                context=json.dumps(metadata),
                created_at=s["created_at"],
            )
        )
