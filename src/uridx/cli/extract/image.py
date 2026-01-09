"""Extract image descriptions via Ollama vision model."""

import base64
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import httpx
import typer

from uridx.config import URIDX_API_URL
from uridx.db.operations import get_existing_source_uris

from .base import get_file_mtime, output, resolve_paths

EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def extract(
    paths: Annotated[Optional[list[Path]], typer.Argument(help="Files or directories")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Vision model")] = "",
    base_url: Annotated[str, typer.Option("--base-url", help="Ollama URL")] = "",
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-process all files even if already ingested")] = False,
):
    """Extract image descriptions via Ollama vision model."""
    ollama_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    vision_model = model or os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")

    # Build list of source_uris that will be generated
    source_uri_map: dict[str, Path] = {}
    for img_file in resolve_paths(paths or [], EXTENSIONS):
        uri = f"file://{img_file.resolve()}"
        source_uri_map[uri] = img_file

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

    for source_uri, img_file in source_uri_map.items():
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
        except httpx.ConnectError:
            print(f"Cannot connect to Ollama at {ollama_url}", file=sys.stderr)
            raise typer.Exit(1)
        except Exception as e:
            print(f"Error describing {img_file}: {e}", file=sys.stderr)
            continue

        if not description or not description.strip():
            continue

        output(
            {
                "source_uri": source_uri,
                "chunks": [
                    {"text": description.strip(), "key": "description", "meta": {"original_filename": img_file.name}}
                ],
                "tags": ["image"],
                "title": img_file.stem,
                "source_type": "image",
                "context": json.dumps({"path": str(img_file), "vision_model": vision_model}),
                "replace": True,
                "created_at": get_file_mtime(img_file),
            }
        )
