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

### Add files

The quickest way to index files or folders is to use `uridx add <path>`. The extractor is auto-detected by file extension, but you can override it with `--extractor <name>`.:

```bash
uridx add notes.md                        # single file
uridx add ./notes/                        # whole directory (recursive)
uridx add ./docs/ --tag work              # tag everything
uridx add report.pdf --extractor docling  # force an extractor
uridx add ./mixed/ --dry-run              # preview "path -> extractor", ingest nothing
```

For extractors that aren't actual files on disk (e.g. ingesting claude code/tsugite conversations, remote http urls, etc), use the `extract … | ingest` pipe below.

### Ingest content

For structured data, remote/multi-machine ingest, or piping between machines, use the composable `extract … | ingest` pipe directly.

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

Example with remote Ollama:
```bash
OLLAMA_BASE_URL=http://my-server:11434 uridx search "test"
```
