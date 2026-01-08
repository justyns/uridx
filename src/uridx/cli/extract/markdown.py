"""Extract markdown files, splitting by headings."""

import json
import re
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from .base import get_file_mtime, output, resolve_paths

EXTENSIONS = {".md", ".markdown"}


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

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{1,6}\s+", part):
            if current_heading or current_content:
                text_parts = [current_heading] if current_heading else []
                text_parts.extend(current_content)
                chunk_text = "\n\n".join(text_parts)
                if chunk_text.strip():
                    chunks.append(
                        {
                            "text": chunk_text,
                            "key": _slugify(current_heading) if current_heading else f"section-{len(chunks)}",
                            "meta": {"heading": current_heading},
                        }
                    )
            current_heading = part
            current_content = []
        else:
            current_content.append(part)

    if current_heading or current_content:
        text_parts = [current_heading] if current_heading else []
        text_parts.extend(current_content)
        chunk_text = "\n\n".join(text_parts)
        if chunk_text.strip():
            chunks.append(
                {
                    "text": chunk_text,
                    "key": _slugify(current_heading) if current_heading else f"section-{len(chunks)}",
                    "meta": {"heading": current_heading},
                }
            )

    if not chunks and content.strip():
        chunks.append({"text": content.strip(), "key": "full", "meta": {}})

    return chunks


def extract(
    paths: Annotated[Optional[list[Path]], typer.Argument(help="Files or directories")] = None,
):
    """Extract markdown files, splitting by headings."""
    for md_file in resolve_paths(paths or [], EXTENSIONS):
        try:
            chunks = _parse(md_file)
        except Exception as e:
            print(f"Error parsing {md_file}: {e}", file=sys.stderr)
            continue

        if not chunks:
            continue

        output(
            {
                "source_uri": f"file://{md_file.resolve()}",
                "chunks": chunks,
                "tags": ["markdown", "document"],
                "title": md_file.stem,
                "source_type": "markdown",
                "context": json.dumps({"path": str(md_file)}),
                "replace": True,
                "created_at": get_file_mtime(md_file),
            }
        )
