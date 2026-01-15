# uridx

Personal semantic search index with MCP interface. Index your notes, chats, code, and documents for unified search.

## Installation

```bash
uv sync
```

Requires [Ollama](https://ollama.ai/) running locally with an embedding model:

```bash
ollama pull qwen3-embedding:0.6b
```

## CLI Usage

### Ingest content

JSONL format (recommended for structured data):
```bash
echo '{"source_uri": "note://idea/1", "title": "Project idea", "source_type": "note", "tags": ["idea"], "chunks": [{"text": "Build a semantic search tool for personal knowledge."}]}' | uridx ingest
```

Raw text:
```bash
cat document.md | uridx ingest --text "file://docs/document.md"
```

### Extract documents with docling

[Docling](https://github.com/docling-project/docling) handles PDFs, DOCX, XLSX, PPTX, HTML, images, and more with OCR and table extraction:

```bash
uv pip install 'uridx[docling]'

# Single file
uridx extract docling myfile.pdf | uridx ingest

# Directory (processes all supported files)
uridx extract docling ./documents/ | uridx ingest
```

### Search

```bash
uridx search "semantic search"
uridx search "project ideas" --tag idea
uridx search "python tips" --type note --limit 5
uridx search "exact phrase" --json
```

### Stats

```bash
uridx stats
```

### MCP Server

```bash
# stdio mode (for MCP clients like Claude Desktop)
uridx serve

# HTTP mode (for remote API access)
uridx serve --http --port 8000
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `URIDX_DB_PATH` | `~/.local/share/uridx/uridx.db` | SQLite database path |
| `URIDX_API_URL` | (none) | Remote uridx server URL for extractors |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:0.6b` | Embedding model |

Example with remote Ollama:
```bash
OLLAMA_BASE_URL=http://my-server:11434 uridx serve
```
