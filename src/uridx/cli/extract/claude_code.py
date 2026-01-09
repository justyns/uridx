"""Extract Claude Code conversations from ~/.claude/projects/"""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from uridx.config import URIDX_API_URL
from uridx.db.operations import get_existing_source_uris

from .base import output


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


def _build_turns(messages: list[dict]) -> list[dict]:
    turns = []
    current_user = None
    current_assistant = []
    turn_index = 0

    for msg in messages:
        msg_type = msg.get("type")
        if msg_type not in ("user", "assistant"):
            continue

        content = _extract_content(msg.get("message", {}))
        if not content:
            continue

        if msg_type == "user" and not _is_tool_result(msg):
            if current_user or current_assistant:
                text_parts = []
                if current_user:
                    text_parts.append(f"User: {current_user}")
                if current_assistant:
                    text_parts.append(f"Assistant: {' '.join(current_assistant)}")
                if text_parts:
                    turns.append(
                        {
                            "text": "\n\n".join(text_parts),
                            "key": f"turn-{turn_index}",
                            "meta": {"turn_index": turn_index},
                        }
                    )
                    turn_index += 1
            current_user = content
            current_assistant = []
        elif msg_type == "assistant":
            current_assistant.append(content)

    if current_user or current_assistant:
        text_parts = []
        if current_user:
            text_parts.append(f"User: {current_user}")
        if current_assistant:
            text_parts.append(f"Assistant: {' '.join(current_assistant)}")
        if text_parts:
            turns.append(
                {
                    "text": "\n\n".join(text_parts),
                    "key": f"turn-{turn_index}",
                    "meta": {"turn_index": turn_index},
                }
            )

    return turns


def _parse_conversation(jsonl_path: Path) -> dict | None:
    messages = []
    metadata = {}
    first_timestamp = None
    last_timestamp = None

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                messages.append(msg)
                if first_timestamp is None:
                    first_timestamp = msg.get("timestamp")
                last_timestamp = msg.get("timestamp")
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

    chunks = _build_turns(messages)
    if not chunks:
        return None

    title = metadata.get("slug") or jsonl_path.stem
    metadata["started_at"] = first_timestamp
    metadata["ended_at"] = last_timestamp

    return {"chunks": chunks, "metadata": metadata, "title": title}


def extract(
    path: Annotated[Optional[Path], typer.Argument(help="Projects directory")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files even if already ingested")] = False,
):
    """Extract Claude Code conversations from ~/.claude/projects/"""
    projects_dir = path or (Path.home() / ".claude" / "projects")

    if not projects_dir.exists():
        print(f"Projects directory not found: {projects_dir}", file=sys.stderr)
        raise typer.Exit(1)

    # Build list of source_uris that will be generated
    source_uri_map: dict[str, tuple[Path, str]] = {}  # uri -> (jsonl_file, project_hash)
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        project_hash = project_dir.name
        for jsonl_file in project_dir.glob("*.jsonl"):
            if jsonl_file.stat().st_size == 0:
                continue
            uri = f"claude-code://{project_hash}/{jsonl_file.stem}"
            source_uri_map[uri] = (jsonl_file, project_hash)

    # Check which already exist (unless --force)
    if not force and source_uri_map:
        if not URIDX_API_URL:
            from uridx.db.engine import init_db

            init_db()
        existing = get_existing_source_uris(list(source_uri_map.keys()))
        for uri in existing:
            print(f"Skipping {source_uri_map[uri][0]} (already ingested)", file=sys.stderr)
            del source_uri_map[uri]

    if not source_uri_map:
        return

    for source_uri, (jsonl_file, project_hash) in source_uri_map.items():
        try:
            result = _parse_conversation(jsonl_file)
        except Exception as e:
            print(f"Error parsing {jsonl_file}: {e}", file=sys.stderr)
            continue

        if not result or not result["chunks"]:
            continue

        output(
            {
                "source_uri": source_uri,
                "chunks": result["chunks"],
                "tags": ["claude-code", "conversation"],
                "title": result["title"],
                "source_type": "claude-code",
                "context": json.dumps(result["metadata"]),
                "replace": True,
            }
        )
