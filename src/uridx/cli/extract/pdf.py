"""Extract PDF files by page using pdfplumber."""

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from .base import output


def extract(
    path: Annotated[Optional[Path], typer.Argument(help="File or directory")] = None,
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

    for pdf_file in files:
        if not pdf_file.is_file():
            continue

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
                "source_uri": f"file://{pdf_file.resolve()}",
                "chunks": chunks,
                "tags": ["pdf", "document"],
                "title": pdf_file.stem,
                "source_type": "pdf",
                "context": json.dumps({"path": str(pdf_file), "pages": len(chunks)}),
                "replace": True,
            }
        )
