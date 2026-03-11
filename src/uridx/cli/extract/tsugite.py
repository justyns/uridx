"""Extract Tsugite chat history from ~/.local/share/tsugite/history/"""

import json
import re
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from uridx.config import URIDX_API_URL
from uridx.db.operations import get_existing_source_uris

from .base import output

# Patterns for system/tool content to skip in user messages
_SKIP_PATTERNS = (
    "<tsugite_execution_result",
    "<scheduled_task",
    "<context_update",
)


def _extract_text(content) -> str:
    """Extract text from message content (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)
    return ""


def _is_system_content(text: str) -> bool:
    """Check if content is system/tool output that should be skipped."""
    stripped = text.lstrip()
    return any(stripped.startswith(pat) for pat in _SKIP_PATTERNS)


def _build_turns(messages: list[dict]) -> list[dict]:
    """Build turn chunks from a list of turn records."""
    turns = []

    for record in messages:
        if record.get("type") != "turn":
            continue

        msgs = record.get("messages", [])
        user_parts = []
        assistant_parts = []

        for msg in msgs:
            role = msg.get("role", "")
            text = _extract_text(msg.get("content", ""))
            if not text.strip():
                continue

            if role == "user":
                if not _is_system_content(text):
                    user_parts.append(text)
            elif role == "assistant":
                assistant_parts.append(text)

        # Build the chunk text
        text_parts = []
        if user_parts:
            text_parts.append(f"User: {' '.join(user_parts)}")
        if assistant_parts:
            text_parts.append(f"Assistant: {' '.join(assistant_parts)}")

        if text_parts:
            turns.append(
                {
                    "text": "\n\n".join(text_parts),
                    "key": f"turn-{len(turns)}",
                    "meta": {"turn_index": len(turns)},
                }
            )

    return turns


def _parse_session(jsonl_path: Path) -> dict | None:
    """Parse a Tsugite session JSONL file into chunks."""
    records = []
    meta = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")
            if rtype == "session_meta":
                meta = {
                    "agent": record.get("agent", "unknown"),
                    "model": record.get("model"),
                    "machine": record.get("machine"),
                    "created_at": record.get("created_at"),
                    "compacted_from": record.get("compacted_from"),
                }
            elif rtype == "turn":
                records.append(record)

    if not records:
        return None

    chunks = _build_turns(records)
    if not chunks:
        return None

    agent = meta.get("agent", "unknown")
    session_id = jsonl_path.stem
    title = f"{agent}: {session_id}"

    return {
        "chunks": chunks,
        "metadata": meta,
        "title": title,
        "agent": agent,
        "session_id": session_id,
    }


def _extract_agent_from_filename(name: str) -> str | None:
    """Try to extract agent name from filename like 20260310_084034_odyn_f30fa0."""
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[2]
    return None


def _extract_date_from_filename(name: str) -> str | None:
    """Extract date string (YYYYMMDD) from filename."""
    parts = name.split("_")
    if parts and re.match(r"^\d{8}$", parts[0]):
        return parts[0]
    return None


def extract(
    path: Annotated[Optional[Path], typer.Argument(help="History directory")] = None,
    agent: Annotated[Optional[str], typer.Option("--agent", "-a", help="Filter by agent name")] = None,
    since: Annotated[
        Optional[str], typer.Option("--since", "-s", help="Only sessions after this date (YYYY-MM-DD)")
    ] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files")] = False,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
):
    """Extract Tsugite chat history from session JSONL files."""
    history_dir = path or (Path.home() / ".local" / "share" / "tsugite" / "history")

    if not history_dir.exists():
        print(f"History directory not found: {history_dir}", file=sys.stderr)
        raise typer.Exit(1)

    # Parse since date filter
    since_date = None
    if since:
        try:
            since_date = since.replace("-", "")  # YYYY-MM-DD -> YYYYMMDD
        except Exception:
            print(f"Invalid date format: {since} (expected YYYY-MM-DD)", file=sys.stderr)
            raise typer.Exit(1)

    # Collect candidate files with pre-filtering
    source_uri_map: dict[str, Path] = {}
    for jsonl_file in history_dir.glob("*.jsonl"):
        if jsonl_file.stat().st_size == 0:
            continue

        stem = jsonl_file.stem
        file_agent = _extract_agent_from_filename(stem)

        # Fast agent filter from filename (avoids parsing the file)
        if agent and file_agent and file_agent != agent:
            continue

        # Fast date filter from filename
        if since_date:
            file_date = _extract_date_from_filename(stem)
            if file_date and file_date < since_date:
                continue

        uri = f"tsugite://{file_agent or 'unknown'}/{stem}"
        source_uri_map[uri] = jsonl_file

    # Dedup: skip already-indexed sessions
    if not force and source_uri_map:
        if not URIDX_API_URL:
            from uridx.db.engine import init_db

            init_db()
        existing = get_existing_source_uris(list(source_uri_map.keys()))
        for uri in existing:
            print(f"Skipping {source_uri_map[uri].name} (already ingested)", file=sys.stderr)
            del source_uri_map[uri]

    if not source_uri_map:
        return

    for source_uri, jsonl_file in source_uri_map.items():
        try:
            result = _parse_session(jsonl_file)
        except Exception as e:
            print(f"Error parsing {jsonl_file.name}: {e}", file=sys.stderr)
            continue

        if not result or not result["chunks"]:
            continue

        # If --agent was specified but the file's meta agent doesn't match, skip
        if agent and result["agent"] != agent:
            continue

        session_agent = result["agent"]
        session_id = result["session_id"]
        base_uri = f"tsugite://{session_agent}/{session_id}"

        auto_tags = [session_agent, "conversation", "tsugite"]
        all_tags = auto_tags + (tag or [])

        output(
            {
                "source_uri": base_uri,
                "chunks": result["chunks"],
                "tags": all_tags,
                "title": result["title"],
                "source_type": "tsugite",
                "context": json.dumps(result["metadata"]),
                "replace": True,
            }
        )
