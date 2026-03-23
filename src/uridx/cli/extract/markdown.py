"""Extract markdown files, splitting by headings."""

import json
import re
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from uridx.db.operations import get_existing_source_uris

from .base import get_file_mtime, output, resolve_paths

EXTENSIONS = {".md", ".markdown"}
MIN_CHUNK_SIZE = 100


def _heading_level(heading: str | None) -> int:
    if not heading:
        return 0
    match = re.match(r"^(#{1,6})\s+", heading)
    return len(match.group(1)) if match else 0


def _merge_small_chunks(chunks: list[dict]) -> list[dict]:
    merged = []
    carry = None
    for chunk in chunks:
        if carry:
            chunk = {**chunk, "text": carry["text"] + "\n\n" + chunk["text"]}
            carry = None
        if len(chunk["text"]) < MIN_CHUNK_SIZE:
            carry = chunk
        else:
            merged.append(chunk)
    if carry:
        if merged:
            merged[-1] = {**merged[-1], "text": merged[-1]["text"] + "\n\n" + carry["text"]}
        else:
            merged.append(carry)
    return merged


def _slugify(text: str) -> str:
    if not text:
        return "untitled"
    text = re.sub(r"^#+\s*", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:50] or "untitled"


def _parse(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    heading_pattern = r"^(#{1,6}\s+.+)$"
    parts = re.split(heading_pattern, content, flags=re.MULTILINE)

    chunks = []
    current_heading = None
    current_content = []
    # Track parent headings by level: {level: heading_text}
    parent_headings: dict[int, str] = {}

    def _emit_chunk():
        if not current_heading and not current_content:
            return
        text_parts = []
        # Prepend ancestor headings for context
        if current_heading:
            level = _heading_level(current_heading)
            ancestors = [parent_headings[lvl] for lvl in sorted(parent_headings) if lvl < level]
            text_parts.extend(ancestors)
        text_parts.append(current_heading or "")
        text_parts.extend(current_content)
        text_parts = [p for p in text_parts if p]
        chunk_text = "\n\n".join(text_parts)
        if chunk_text.strip():
            chunks.append(
                {
                    "text": chunk_text,
                    "key": _slugify(current_heading) if current_heading else f"section-{len(chunks)}",
                    "meta": {"heading": current_heading},
                }
            )

    for part in parts:
        part = part.strip()
        if not part:
            continue
        level = _heading_level(part)
        if level:
            _emit_chunk()
            current_heading = part
            current_content = []
            parent_headings[level] = part
            # Clear any deeper headings
            for lvl in [k for k in parent_headings if k > level]:
                del parent_headings[lvl]
        else:
            current_content.append(part)

    _emit_chunk()

    if not chunks and content.strip():
        chunks.append({"text": content.strip(), "key": "full", "meta": {}})

    return _merge_small_chunks(chunks)


def extract(
    paths: Annotated[Optional[list[Path]], typer.Argument(help="Files or directories")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files even if already ingested")] = False,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
):
    """Extract markdown files, splitting by headings."""
    # Build list of source_uris that will be generated
    source_uri_map: dict[str, Path] = {}
    for md_file in resolve_paths(paths or [], EXTENSIONS):
        uri = f"file://{md_file.resolve()}"
        source_uri_map[uri] = md_file

    # Check which already exist (unless --force)
    if not force and source_uri_map:
        from uridx.db.engine import init_db

        init_db()
        existing = get_existing_source_uris(list(source_uri_map.keys()))
        for uri in existing:
            print(f"Skipping {source_uri_map[uri]} (already ingested)", file=sys.stderr)
            del source_uri_map[uri]

    if not source_uri_map:
        return

    for source_uri, md_file in source_uri_map.items():
        try:
            chunks = _parse(md_file)
        except Exception as e:
            print(f"Error parsing {md_file}: {e}", file=sys.stderr)
            continue

        if not chunks:
            continue

        output(
            {
                "source_uri": source_uri,
                "chunks": chunks,
                "tags": ["markdown", "document"] + (tag or []),
                "title": md_file.stem,
                "source_type": "markdown",
                "context": json.dumps({"path": str(md_file)}),
                "replace": True,
                "created_at": get_file_mtime(md_file),
            }
        )
