"""Extract documents using docling (PDF, DOCX, XLSX, PPTX, HTML, images)."""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import urlparse

import typer

from .base import get_file_mtime, output, resolve_paths

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".html", ".xhtml", ".htm",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
    ".md", ".adoc", ".csv",
}


def extract(
    sources: Annotated[Optional[list[str]], typer.Argument(help="Files, directories, or URLs")] = None,
):
    """Extract documents using docling (requires docling)."""
    try:
        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker import HybridChunker
    except ImportError:
        print("docling not installed. Install with: uv pip install 'uridx[docling]'", file=sys.stderr)
        raise typer.Exit(1)

    converter = DocumentConverter()
    chunker = HybridChunker()

    sources = sources or []
    urls = [s for s in sources if s.startswith(("http://", "https://"))]
    local_paths = [Path(s) for s in sources if not s.startswith(("http://", "https://"))]

    for url in urls:
        _convert_source(converter, chunker, url, url, created_at=None)

    for file_path in resolve_paths(local_paths, SUPPORTED_EXTENSIONS):
        _convert_source(converter, chunker, str(file_path), f"file://{file_path.resolve()}", created_at=get_file_mtime(file_path))


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
