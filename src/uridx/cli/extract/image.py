"""Extract image descriptions via Ollama vision model."""

import base64
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Optional

import httpx
import typer

from .base import MissingExtractorDependency, file_uri, get_file_mtime, output, prepare_files

EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def iter_records(
    files: list[Path],
    *,
    tag: Optional[list[str]] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Iterator[dict]:
    """Yield ingest records describing images via an Ollama vision model.

    model/base_url default from env (OLLAMA_VISION_MODEL/OLLAMA_BASE_URL) so a generic
    caller (the `add` command) needs no extra args. Raises MissingExtractorDependency
    if Ollama is unreachable, so a whole image bucket is skipped rather than crashing.
    """
    ollama_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    vision_model = model or os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")

    for img_file in files:
        try:
            with open(img_file, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": vision_model,
                        "prompt": "Describe this image in detail. Include any text visible in the image.",
                        "images": [image_data],
                        "stream": False,
                    },
                )
                response.raise_for_status()
                description = response.json()["response"]
        except httpx.ConnectError as e:
            raise MissingExtractorDependency(f"Cannot connect to Ollama at {ollama_url}") from e
        except Exception as e:
            print(f"Error describing {img_file}: {e}", file=sys.stderr)
            continue

        if not description or not description.strip():
            continue

        yield {
            "source_uri": file_uri(img_file),
            "chunks": [
                {"text": description.strip(), "key": "description", "meta": {"original_filename": img_file.name}}
            ],
            "tags": ["image"] + (tag or []),
            "title": img_file.stem,
            "source_type": "image",
            "context": json.dumps({"path": str(img_file), "vision_model": vision_model}),
            "replace": True,
            "created_at": get_file_mtime(img_file),
        }


def extract(
    paths: Annotated[Optional[list[Path]], typer.Argument(help="Files or directories")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Vision model")] = "",
    base_url: Annotated[str, typer.Option("--base-url", help="Ollama URL")] = "",
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files even if already ingested")] = False,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Additional tags")] = None,
):
    """Extract image descriptions via Ollama vision model."""
    try:
        for rec in iter_records(
            prepare_files(paths or [], EXTENSIONS, force), tag=tag, model=model or None, base_url=base_url or None
        ):
            output(rec)
    except MissingExtractorDependency as e:
        print(str(e), file=sys.stderr)
        raise typer.Exit(1)
