"""Standard Record model shared by the extractors and ingester."""

from collections.abc import Iterable, Iterator

from pydantic import BaseModel, Field, ValidationError


TITLE_CHUNK_KEY = "title"
"""Reserved chunk key: the item title, indexed as its own chunk so it is searchable."""


class RecordError(Exception):
    """Invalid record line in a JSONL stream."""


class ChunkInput(BaseModel):
    """One unit of text to embed and index, with optional key and metadata."""

    text: str
    key: str | None = None
    meta: dict | None = None
    embedding: str | None = None  # base64 packed float32 (export/import wire only)


class LocationInput(BaseModel):
    """A known location of an item's source, with the machine that holds it."""

    uri: str
    machine: str | None = None


class Record(BaseModel):
    """A source item plus its chunks. `source_uri` is the only required field.

    The trailing optional fields are the export/import wire format: an export
    line is a Record plus provenance, so `import` reuses the ingest machinery.
    """

    source_uri: str
    chunks: list[ChunkInput] = Field(default_factory=list)
    title: str | None = None
    source_type: str | None = None
    context: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str | None = None
    machine: str | None = None
    updated_at: str | None = None
    content_hash: str | None = None  # informational; import recomputes from chunk texts
    locations: list[LocationInput] | None = None
    embed_model: str | None = None
    embed_dim: int | None = None


def iter_jsonl_records(lines: Iterable[str]) -> Iterator[Record]:
    for i, line in enumerate(lines, 1):
        if not (line := line.strip()):
            continue
        try:
            yield Record.model_validate_json(line)
        except ValidationError as e:
            raise RecordError(f"invalid record on line {i}:\n{e}") from e
