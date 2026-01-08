"""Extract image descriptions via Ollama vision model."""

import base64
import json
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import httpx
import typer

from .base import get_file_mtime, output, resolve_paths

EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def extract(
    paths: Annotated[Optional[list[Path]], typer.Argument(help="Files or directories")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Vision model")] = "",
    base_url: Annotated[str, typer.Option("--base-url", help="Ollama URL")] = "",
):
    """Extract image descriptions via Ollama vision model."""
    ollama_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    vision_model = model or os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")

    for img_file in resolve_paths(paths or [], EXTENSIONS):
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
                "source_uri": f"file://{img_file.resolve()}",
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
