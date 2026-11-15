"""Extract documents using docling (PDF, DOCX, XLSX, PPTX, HTML, images)."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional, Union
from urllib.parse import urlparse

import typer

from .base import (
    MissingExtractorDependency,
    NoDefaultExcludes,
    NoGitignore,
    drop_indexed,
    file_hash,
    file_uri,
    get_file_mtime,
    output,
    prepare_files,
)

if TYPE_CHECKING:
    from uridx.record import Record

EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".html",
    ".xhtml",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
    ".webp",
    ".md",
    ".adoc",
    ".csv",
}


def iter_records(sources: list[Union[str, Path]], *, tag: Optional[list[str]] = None) -> Iterator["Record"]:
    """Yield ingest records for docling-supported sources (local files or http(s) URLs).

    `add` passes local Paths only; the `extract` wrapper may also pass URL strings.
    The DocumentConverter/HybridChunker are built lazily on first iteration. Raises
    MissingExtractorDependency if docling isn't installed.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker import HybridChunker
    except ImportError as e:
        raise MissingExtractorDependency("docling not installed. Install with: uv pip install 'uridx[docling]'") from e

    converter = DocumentConverter()
    chunker = HybridChunker()
    extra_tags = tag or []

    for source in sources:
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            source_uri, convert_arg, created_at, source_hash = source, source, None, None
        else:
            path = Path(source)
            source_uri, convert_arg, created_at = file_uri(path), str(path), get_file_mtime(path)
            source_hash = file_hash(path)
        record = _build_record(converter, chunker, convert_arg, source_uri, created_at, extra_tags)
        if record:
            record.source_hash = source_hash
            yield record


def extract(
    sources: Annotated[Optional[list[str]], typer.Argument(help="Files, directories, or URLs")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files even if already ingested")] = False,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
    no_gitignore: NoGitignore = False,
    no_default_excludes: NoDefaultExcludes = False,
):
    """Extract documents using docling (requires docling)."""
    sources = sources or []
    urls: list[Union[str, Path]] = [s for s in sources if s.startswith(("http://", "https://"))]
    local_paths = [Path(s) for s in sources if not s.startswith(("http://", "https://"))]

    # URLs are filtered by URI, files by content hash.
    survivors: list[Union[str, Path]] = list(drop_indexed({url: url for url in urls}, force).values())
    survivors += prepare_files(local_paths, EXTENSIONS, force, no_gitignore, no_default_excludes)
    if not survivors:
        return

    try:
        for rec in iter_records(survivors, tag=tag):
            output(rec)
    except MissingExtractorDependency as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(1)


def _build_record(
    converter, chunker, source: str, source_uri: str, created_at: str | None, extra_tags: list[str]
) -> Optional["Record"]:
    """Convert a single source and return an ingest record (or None on error/empty)."""
    from uridx.record import Record

    try:
        result = converter.convert(source)
        doc = result.document
        chunk_iter = chunker.chunk(dl_doc=doc)
    except Exception as e:
        print(f"Error processing {source}: {e}", file=sys.stderr)
        return None

    chunks = []
    for i, chunk in enumerate(chunk_iter):
        text = chunk.text.strip() if hasattr(chunk, "text") else str(chunk).strip()
        if text:
            chunks.append({"text": text, "key": f"chunk-{i}"})

    if not chunks:
        return None

    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        title = Path(parsed.path).stem or parsed.netloc
        ext = Path(parsed.path).suffix.lstrip(".").lower() or "html"
    else:
        title = Path(source).stem
        ext = Path(source).suffix.lstrip(".").lower()

    return Record(
        source_uri=source_uri,
        chunks=chunks,
        tags=["document", *([ext] if ext else []), *extra_tags],
        title=title,
        source_type="document",
        context=json.dumps({"source": source}),
        created_at=created_at,
    )
