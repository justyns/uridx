"""Extract documents using docling (PDF, DOCX, XLSX, PPTX, HTML, images)."""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlparse

import typer

from uridx.config import URIDX_API_URL
from uridx.db.operations import get_existing_source_uris

from .base import get_file_mtime, output, resolve_paths

SUPPORTED_EXTENSIONS = {
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


def extract(
    sources: Annotated[Optional[list[str]], typer.Argument(help="Files, directories, or URLs")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files even if already ingested")] = False,
):
    """Extract documents using docling (requires docling)."""
    try:
        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker import HybridChunker
    except ImportError:
        print("docling not installed. Install with: uv pip install 'uridx[docling]'", file=sys.stderr)
        raise typer.Exit(1)

    sources = sources or []
    urls = [s for s in sources if s.startswith(("http://", "https://"))]
    local_paths = [Path(s) for s in sources if not s.startswith(("http://", "https://"))]

    # Build list of source_uris that will be generated
    source_uri_map: dict[str, tuple[str, str | None]] = {}
    for url in urls:
        source_uri_map[url] = (url, None)
    for file_path in resolve_paths(local_paths, SUPPORTED_EXTENSIONS):
        uri = f"file://{file_path.resolve()}"
        source_uri_map[uri] = (str(file_path), get_file_mtime(file_path))

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

    converter = DocumentConverter()
    chunker = HybridChunker()

    for source_uri, (source, created_at) in source_uri_map.items():
        _convert_source(converter, chunker, source, source_uri, created_at=created_at)


def _convert_source(converter, chunker, source: str, source_uri: str, created_at: str | None = None):
    """Convert a single source and output JSONL."""
    try:
        result = converter.convert(source)
        doc = result.document
        chunk_iter = chunker.chunk(dl_doc=doc)
    except Exception as e:
        print(f"Error processing {source}: {e}", file=sys.stderr)
        return

    chunks = []
    for i, chunk in enumerate(chunk_iter):
        text = chunk.text.strip() if hasattr(chunk, "text") else str(chunk).strip()
        if text:
            chunks.append({"text": text, "key": f"chunk-{i}"})

    if not chunks:
        return

    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        title = Path(parsed.path).stem or parsed.netloc
        ext = Path(parsed.path).suffix.lstrip(".").lower() or "html"
    else:
        title = Path(source).stem
        ext = Path(source).suffix.lstrip(".").lower()

    record = {
        "source_uri": source_uri,
        "chunks": chunks,
        "tags": ["document", ext] if ext else ["document"],
        "title": title,
        "source_type": "document",
        "context": json.dumps({"source": source}),
        "replace": True,
    }
    if created_at:
        record["created_at"] = created_at
    output(record)
