"""Extract subcommands with plugin support.

Built-in extractors and plugin discovery via entry points.
Plugins register under 'uridx.extractors' entry point group.
"""

import base64
import json
import os
import re
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Annotated, Optional

import httpx
import typer

app = typer.Typer(help="Extract content to JSONL for ingestion")

MARKDOWN_EXTENSIONS = {".md", ".markdown"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def _output(record: dict) -> None:
    """Output a single JSONL record to stdout."""
    print(json.dumps(record))


# --- Claude Code Extractor ---


def _extract_claude_content(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    texts.append(f"[Tool: {block.get('name')}]")
        return "\n".join(texts)
    return ""


def _is_tool_result(msg: dict) -> bool:
    content = msg.get("message", {}).get("content", [])
    if isinstance(content, list) and content:
        first = content[0] if content else {}
        return isinstance(first, dict) and first.get("type") == "tool_result"
    return False


def _build_turns(messages: list[dict]) -> list[dict]:
    turns = []
    current_user = None
    current_assistant = []
    turn_index = 0

    for msg in messages:
        msg_type = msg.get("type")
        if msg_type not in ("user", "assistant"):
            continue

        content = _extract_claude_content(msg.get("message", {}))
        if not content:
            continue

        if msg_type == "user" and not _is_tool_result(msg):
            if current_user or current_assistant:
                text_parts = []
                if current_user:
                    text_parts.append(f"User: {current_user}")
                if current_assistant:
                    text_parts.append(f"Assistant: {' '.join(current_assistant)}")
                if text_parts:
                    turns.append(
                        {
                            "text": "\n\n".join(text_parts),
                            "key": f"turn-{turn_index}",
                            "meta": {"turn_index": turn_index},
                        }
                    )
                    turn_index += 1
            current_user = content
            current_assistant = []
        elif msg_type == "assistant":
            current_assistant.append(content)

    if current_user or current_assistant:
        text_parts = []
        if current_user:
            text_parts.append(f"User: {current_user}")
        if current_assistant:
            text_parts.append(f"Assistant: {' '.join(current_assistant)}")
        if text_parts:
            turns.append(
                {
                    "text": "\n\n".join(text_parts),
                    "key": f"turn-{turn_index}",
                    "meta": {"turn_index": turn_index},
                }
            )

    return turns


def _parse_claude_conversation(jsonl_path: Path) -> dict | None:
    messages = []
    metadata = {}
    first_timestamp = None
    last_timestamp = None

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                messages.append(msg)
                if first_timestamp is None:
                    first_timestamp = msg.get("timestamp")
                last_timestamp = msg.get("timestamp")
                if not metadata and msg.get("cwd"):
                    metadata = {
                        "project_path": msg.get("cwd"),
                        "agent_id": msg.get("agentId"),
                        "session_id": msg.get("sessionId"),
                        "git_branch": msg.get("gitBranch"),
                        "slug": msg.get("slug"),
                    }
            except json.JSONDecodeError:
                continue

    if not messages:
        return None

    chunks = _build_turns(messages)
    if not chunks:
        return None

    title = metadata.get("slug") or jsonl_path.stem
    metadata["started_at"] = first_timestamp
    metadata["ended_at"] = last_timestamp

    return {"chunks": chunks, "metadata": metadata, "title": title}


@app.command("claude-code")
def extract_claude_code(
    path: Annotated[Optional[Path], typer.Argument(help="Projects directory")] = None,
):
    """Extract Claude Code conversations from ~/.claude/projects/"""
    projects_dir = path or (Path.home() / ".claude" / "projects")

    if not projects_dir.exists():
        print(f"Projects directory not found: {projects_dir}", file=sys.stderr)
        raise typer.Exit(1)

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        project_hash = project_dir.name

        for jsonl_file in project_dir.glob("*.jsonl"):
            if jsonl_file.stat().st_size == 0:
                continue

            try:
                result = _parse_claude_conversation(jsonl_file)
            except Exception as e:
                print(f"Error parsing {jsonl_file}: {e}", file=sys.stderr)
                continue

            if not result or not result["chunks"]:
                continue

            _output(
                {
                    "source_uri": f"claude-code://{project_hash}/{jsonl_file.stem}",
                    "chunks": result["chunks"],
                    "tags": ["claude-code", "conversation"],
                    "title": result["title"],
                    "source_type": "claude-code",
                    "context": json.dumps(result["metadata"]),
                    "replace": True,
                }
            )


# --- Markdown Extractor ---


def _slugify(text: str) -> str:
    if not text:
        return "untitled"
    text = re.sub(r"^#+\s*", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:50] or "untitled"


def _parse_markdown(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    heading_pattern = r"^(#{1,6}\s+.+)$"
    parts = re.split(heading_pattern, content, flags=re.MULTILINE)

    chunks = []
    current_heading = None
    current_content = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{1,6}\s+", part):
            if current_heading or current_content:
                text_parts = [current_heading] if current_heading else []
                text_parts.extend(current_content)
                chunk_text = "\n\n".join(text_parts)
                if chunk_text.strip():
                    chunks.append(
                        {
                            "text": chunk_text,
                            "key": _slugify(current_heading) if current_heading else f"section-{len(chunks)}",
                            "meta": {"heading": current_heading},
                        }
                    )
            current_heading = part
            current_content = []
        else:
            current_content.append(part)

    if current_heading or current_content:
        text_parts = [current_heading] if current_heading else []
        text_parts.extend(current_content)
        chunk_text = "\n\n".join(text_parts)
        if chunk_text.strip():
            chunks.append(
                {
                    "text": chunk_text,
                    "key": _slugify(current_heading) if current_heading else f"section-{len(chunks)}",
                    "meta": {"heading": current_heading},
                }
            )

    if not chunks and content.strip():
        chunks.append({"text": content.strip(), "key": "full", "meta": {}})

    return chunks


@app.command("markdown")
def extract_markdown(
    path: Annotated[Optional[Path], typer.Argument(help="File or directory")] = None,
):
    """Extract markdown files, splitting by headings."""
    root = path or Path.cwd()

    if root.is_file():
        files = [root]
    else:
        files = [f for f in root.rglob("*") if f.suffix.lower() in MARKDOWN_EXTENSIONS and f.is_file()]

    for md_file in files:
        try:
            chunks = _parse_markdown(md_file)
        except Exception as e:
            print(f"Error parsing {md_file}: {e}", file=sys.stderr)
            continue

        if not chunks:
            continue

        _output(
            {
                "source_uri": f"file://{md_file.resolve()}",
                "chunks": chunks,
                "tags": ["markdown", "document"],
                "title": md_file.stem,
                "source_type": "markdown",
                "context": json.dumps({"path": str(md_file)}),
                "replace": True,
            }
        )


# --- PDF Extractor ---


@app.command("pdf")
def extract_pdf(
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

        _output(
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


# --- Image Extractor ---


@app.command("image")
def extract_image(
    path: Annotated[Optional[Path], typer.Argument(help="File or directory")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Vision model")] = "",
    base_url: Annotated[str, typer.Option("--base-url", help="Ollama URL")] = "",
):
    """Extract image descriptions via Ollama vision model."""
    ollama_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    vision_model = model or os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")

    root = path or Path.cwd()

    if root.is_file():
        files = [root]
    else:
        files = [f for f in root.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file()]

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
        except httpx.ConnectError:
            print(f"Cannot connect to Ollama at {ollama_url}", file=sys.stderr)
            raise typer.Exit(1)
        except Exception as e:
            print(f"Error describing {img_file}: {e}", file=sys.stderr)
            continue

        if not description or not description.strip():
            continue

        _output(
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
            }
        )


# --- Plugin Discovery ---


def load_plugins():
    """Load extractor plugins from entry points."""
    try:
        eps = entry_points(group="uridx.extractors")
    except TypeError:
        eps = entry_points().get("uridx.extractors", [])

    for ep in eps:
        try:
            extractor = ep.load()
            if isinstance(extractor, typer.Typer):
                app.add_typer(extractor, name=ep.name)
            elif callable(extractor):
                app.command(ep.name)(extractor)
        except Exception as e:
            print(f"Failed to load extractor plugin '{ep.name}': {e}", file=sys.stderr)


load_plugins()
