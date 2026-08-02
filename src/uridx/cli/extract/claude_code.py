"""Extract Claude Code conversations from ~/.claude/projects/"""

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer

from uridx.record import ChunkInput, Record
from uridx.timeutil import to_aware_utc

from ._turns import Role, build_turn_chunks
from .base import drop_indexed, output


@dataclass
class ParsedConversation:
    """A parsed Claude Code conversation: turn chunks, title, metadata, and last activity."""

    chunks: list[ChunkInput]
    title: str
    metadata: dict
    started_at: str | None
    last_active: datetime | None


def _extract_content(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    texts.append(f"[Tool: {block.get('name')}]")
        return "\n".join(texts)
    return ""


def _is_tool_result(msg: dict) -> bool:
    content = msg.get("message", {}).get("content", [])
    if isinstance(content, list) and content:
        first = content[0] if content else {}
        return isinstance(first, dict) and first.get("type") == "tool_result"
    return False


def _iter_turns(messages: list[dict]) -> Iterator[tuple[Role, str]]:
    """Reduce transcript lines to the (role, text) stream build_turn_chunks consumes.

    Tool results arrive as user messages but are tool output, not something the user
    said, so they're dropped rather than starting a new turn.
    """
    for msg in messages:
        role = msg.get("type")
        if role not in ("user", "assistant"):
            continue
        if role == "user" and _is_tool_result(msg):
            continue
        yield role, _extract_content(msg.get("message", {}))


def _parse_conversation(jsonl_path: Path) -> ParsedConversation | None:
    messages = []
    metadata = {}
    first_timestamp = None
    last_timestamp = None
    ai_title = None

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                messages.append(msg)
                ts = msg.get("timestamp")
                if ts:
                    if first_timestamp is None:
                        first_timestamp = ts
                    last_timestamp = ts
                if msg.get("type") == "ai-title" and msg.get("aiTitle"):
                    ai_title = msg["aiTitle"]
                if not metadata and msg.get("cwd"):
                    metadata = {
                        "project_path": msg.get("cwd"),
                        "agent_id": msg.get("agentId"),
                        "session_id": msg.get("sessionId"),
                        "git_branch": msg.get("gitBranch"),
                        "slug": msg.get("slug"),
                    }
            except json.JSONDecodeError:
                continue

    if not messages:
        return None

    chunks = build_turn_chunks(_iter_turns(messages))
    if not chunks:
        return None

    title = ai_title or metadata.get("slug") or jsonl_path.stem
    metadata["started_at"] = first_timestamp
    metadata["ended_at"] = last_timestamp

    return ParsedConversation(
        chunks=chunks,
        title=title,
        metadata=metadata,
        started_at=first_timestamp,
        last_active=to_aware_utc(last_timestamp or first_timestamp),
    )


def extract(
    path: Annotated[Optional[Path], typer.Argument(help="Projects directory")] = None,
    since: Annotated[
        Optional[datetime],
        typer.Option(
            "--since",
            "-s",
            formats=["%Y-%m-%d"],
            help="Only conversations with activity on/after this date (uses file mtime to skip old files)",
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files even if already ingested")] = False,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
):
    """Extract Claude Code conversations from ~/.claude/projects/"""
    projects_dir = path or (Path.home() / ".claude" / "projects")

    if not projects_dir.exists():
        print(f"Projects directory not found: {projects_dir}", file=sys.stderr)
        raise typer.Exit(1)

    since_dt = to_aware_utc(since)
    cutoff_mtime = since_dt.timestamp() if since_dt else None

    # Build list of source_uris that will be generated
    source_uri_map: dict[str, tuple[Path, str]] = {}  # uri -> (jsonl_file, project_hash)
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        project_hash = project_dir.name
        for jsonl_file in project_dir.glob("*.jsonl"):
            st = jsonl_file.stat()
            if st.st_size == 0:
                continue
            # cheap prefilter: a file written normally has mtime >= its last message
            if cutoff_mtime is not None and st.st_mtime < cutoff_mtime:
                continue
            uri = f"claude-code://{project_hash}/{jsonl_file.stem}"
            source_uri_map[uri] = (jsonl_file, project_hash)

    source_uri_map = drop_indexed(source_uri_map, force, label=lambda v: str(v[0]))
    if not source_uri_map:
        return

    for source_uri, (jsonl_file, project_hash) in source_uri_map.items():
        try:
            result = _parse_conversation(jsonl_file)
        except Exception as e:
            print(f"Error parsing {jsonl_file}: {e}", file=sys.stderr)
            continue

        if not result or not result.chunks:
            continue

        # a fresh mtime can outlive the conversation itself (copies, later rewrites)
        if since_dt and (result.last_active is None or result.last_active < since_dt):
            continue

        output(
            Record(
                source_uri=source_uri,
                chunks=result.chunks,
                tags=["claude-code", "conversation"] + (tag or []),
                title=result.title,
                source_type="claude-code",
                context=json.dumps(result.metadata),
                # When the conversation happened, not when it was indexed: this drives
                # the recency boost and `search --after`, as it does for tsugite.
                created_at=result.started_at,
            )
        )
