"""Extract markdown files, splitting by headings."""

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import typer

from .base import NoDefaultExcludes, NoGitignore, file_hash, file_uri, get_file_mtime, output, prepare_files

if TYPE_CHECKING:
    from uridx.record import Record

EXTENSIONS = {".md", ".mdc", ".markdown"}
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


def _parse(content: str) -> list[dict]:
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


def iter_records(files: list[Path], *, tag: Optional[list[str]] = None) -> Iterator["Record"]:
    """Yield ingest records for markdown files, splitting by headings."""
    from uridx.record import Record

    for md_file in files:
        try:
            chunks = _parse(md_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error parsing {md_file}: {e}", file=sys.stderr)
            continue

        if not chunks:
            continue

        yield Record(
            source_uri=file_uri(md_file),
            chunks=chunks,
            tags=["markdown", "document"] + (tag or []),
            title=md_file.stem,
            source_type="markdown",
            context=json.dumps({"path": str(md_file)}),
            created_at=get_file_mtime(md_file),
            source_hash=file_hash(md_file),
        )


def extract(
    paths: Annotated[Optional[list[Path]], typer.Argument(help="Files or directories")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files even if already ingested")] = False,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
    uri: Annotated[Optional[str], typer.Option("--uri", help="Read one document from stdin as this source URI")] = None,
    title: Annotated[Optional[str], typer.Option("--title", help="Title for stdin input")] = None,
    no_gitignore: NoGitignore = False,
    no_default_excludes: NoDefaultExcludes = False,
):
    """Extract markdown files, splitting by headings."""
    if uri is None:
        if title:
            print("Error: --title needs --uri.", file=sys.stderr)
            raise typer.Exit(2)
        for rec in iter_records(
            prepare_files(paths or [], EXTENSIONS, force, no_gitignore, no_default_excludes), tag=tag
        ):
            output(rec)
        return

    if paths:
        print("Error: --uri reads stdin; drop the paths.", file=sys.stderr)
        raise typer.Exit(2)
    if sys.stdin.isatty():
        print("Error: --uri reads markdown from stdin, but nothing was piped in.", file=sys.stderr)
        raise typer.Exit(2)
    chunks = _parse(sys.stdin.read())
    if not chunks:
        print(f"Error: stdin was empty; nothing to extract for {uri}.", file=sys.stderr)
        raise typer.Exit(2)

    from uridx.record import Record

    output(
        Record(
            source_uri=uri,
            chunks=chunks,
            tags=["markdown", "document"] + (tag or []),
            title=title,
            source_type="markdown",
        )
    )
