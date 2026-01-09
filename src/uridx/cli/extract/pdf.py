"""Extract PDF files by page using pdfplumber."""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from uridx.config import URIDX_API_URL
from uridx.db.operations import get_existing_source_uris

from .base import output


def extract(
    path: Annotated[Optional[Path], typer.Argument(help="File or directory")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files even if already ingested")] = False,
):
    """Extract PDF files by page (requires pdfplumber)."""
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber not installed. Install with: uv pip install 'uridx[pdf]'", file=sys.stderr)
        raise typer.Exit(1)

    root = path or Path.cwd()

    if root.is_file():
        files = [root]
    else:
        files = list(root.rglob("*.pdf"))

    # Build list of source_uris that will be generated
    source_uri_map: dict[str, Path] = {}
    for pdf_file in files:
        if not pdf_file.is_file():
            continue
        uri = f"file://{pdf_file.resolve()}"
        source_uri_map[uri] = pdf_file

    # Check which already exist (unless --force)
    if not force and source_uri_map:
        if not URIDX_API_URL:
            from uridx.db.engine import init_db

            init_db()
        existing = get_existing_source_uris(list(source_uri_map.keys()))
        for uri in existing:
            print(f"Skipping {source_uri_map[uri]} (already ingested)", file=sys.stderr)
            del source_uri_map[uri]

    if not source_uri_map:
        return

    for source_uri, pdf_file in source_uri_map.items():
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

        output(
            {
                "source_uri": source_uri,
                "chunks": chunks,
                "tags": ["pdf", "document"],
                "title": pdf_file.stem,
                "source_type": "pdf",
                "context": json.dumps({"path": str(pdf_file), "pages": len(chunks)}),
                "replace": True,
            }
        )
