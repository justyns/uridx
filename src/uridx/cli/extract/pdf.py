"""Extract PDF files by page using pdfplumber."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Optional

import typer

from .base import MissingExtractorDependency, file_uri, output, prepare_files

EXTENSIONS = {".pdf"}


def iter_records(files: list[Path], *, tag: Optional[list[str]] = None) -> Iterator[dict]:
    """Yield ingest records for PDF files, one chunk per page (requires pdfplumber)."""
    try:
        import pdfplumber
    except ImportError as e:
        raise MissingExtractorDependency("pdfplumber not installed. Install with: uv pip install 'uridx[pdf]'") from e

    for pdf_file in files:
        chunks = []
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                    except Exception as e:
                        print(f"Error extracting page {i + 1} from {pdf_file}: {e}", file=sys.stderr)
                        continue
                    if text and text.strip():
                        chunks.append({"text": text.strip(), "key": f"page-{i + 1}", "meta": {"page_number": i + 1}})
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}", file=sys.stderr)
            continue

        if not chunks:
            continue

        yield {
            "source_uri": file_uri(pdf_file),
            "chunks": chunks,
            "tags": ["pdf", "document"] + (tag or []),
            "title": pdf_file.stem,
            "source_type": "pdf",
            "context": json.dumps({"path": str(pdf_file), "pages": len(chunks)}),
            "replace": True,
        }


def extract(
    paths: Annotated[Optional[list[Path]], typer.Argument(help="Files or directories")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files even if already ingested")] = False,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
):
    """Extract PDF files by page (requires pdfplumber)."""
    try:
        for rec in iter_records(prepare_files(paths or [], EXTENSIONS, force), tag=tag):
            output(rec)
    except MissingExtractorDependency as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(1)
