# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

uridx is a personal semantic search index CLI tool. It indexes notes, chats, code, and documents for unified hybrid search combining vector similarity and BM25 full-text search.

## Commands

```bash
uv sync                              # Install dependencies
uv run ruff check src/               # Lint
uv run ruff format src/              # Format
uv run uridx --help                  # CLI help
```

## Architecture

**Data Flow**: CLI → operations.py → SQLite + sqlite-vec + FTS5

- **CLI** (`cli/main.py`): Typer app with `add`, `ingest`, `search`, `stats`, `delete` commands, plus an `extract` subcommand group
- **Ingestion model**: `extract <type> … | ingest` is the composable plumbing (extractors emit JSONL → `ingest` consumes stdin). `add <path>` is porcelain over it: it auto-detects the extractor by file extension (`cli/extract/registry.py`) and runs the same `iter_records` generators in-process. File-drop extractors (`markdown`, `pdf`, `image`, `docling`) expose `iter_records()`; structured sources (`tsugite`, `claude_code`) are `extract`-only.
- **Search** (`search/hybrid.py`): Combines vector search (70% weight) with BM25 (30% weight) via `SearchResult` dataclass
- **Database** (`db/`):
  - `models.py`: SQLModel tables - Item, Chunk, Tag, Setting
  - `engine.py`: Manages SQLite connection, loads sqlite-vec extension, creates FTS5 virtual table with triggers
  - `operations.py`: CRUD operations, handles embedding generation and storage
- **Embeddings** (`embeddings/ollama.py`): Ollama API client with sync/async variants, serializes embeddings to binary for sqlite-vec
- **Config** (`config.py`): Environment variables for DB path, Ollama URL, and embedding model

**Key Data Structures**:
- Items have a unique `source_uri` and contain multiple Chunks
- Chunks store text and are indexed in both `chunk_embeddings` (vector) and `chunks_fts` (FTS5) tables
- Tags are linked to Items via composite primary key

**Database Schema**:
- Regular tables: `item`, `chunk`, `tag`, `setting` (via SQLModel)
- Virtual tables: `chunk_embeddings` (vec0), `chunks_fts` (fts5 with triggers)
