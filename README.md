# uridx

Personal semantic search index CLI. Index your notes, chats, code, and documents for unified search.

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

Add custom tags during ingest:
```bash
uridx extract markdown ./notes/ --tag memory --tag 2026-03 | uridx ingest
echo "test" | uridx ingest --text "test://uri" --tag custom --tag notes
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

# Filter by minimum score to drop low-quality results
uridx search "test" --min-score 0.4

# Filter by source URI prefix
uridx search "test" --source-prefix "memory://"

# Only results created after a date
uridx search "test" --after 2026-03-01

# Adjust hybrid weights (higher = more keyword, lower = more semantic)
uridx search "test" --bm25-weight 0.7

# BM25-only (no semantic/vector search)
uridx search "test" --no-semantic

# Combine filters
uridx search "test" --min-score 0.4 --source-prefix "notes/" --after 2026-03-01
```

### Delete

```bash
# Delete a single item by URI
uridx delete --uri "file:///path/to/removed.md"

# Preview what a prefix delete would remove
uridx delete --source-prefix "file:///home/user/scratch/" --dry-run

# Bulk delete all items matching a prefix
uridx delete --source-prefix "file:///home/user/scratch/"
```

### Stats

```bash
uridx stats
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `URIDX_DB_PATH` | `~/.local/share/uridx/uridx.db` | SQLite database path |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:0.6b` | Embedding model |
| `URIDX_MIN_SCORE` | (none) | Global minimum score threshold for search results |
| `URIDX_BM25_WEIGHT` | `0.3` | Default BM25 weight in hybrid search (vector weight = 1 - this) |

Example with remote Ollama:
```bash
OLLAMA_BASE_URL=http://my-server:11434 uridx search "test"
```
