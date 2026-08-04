"""Standard Record model shared by the extractors and ingester."""

from collections.abc import Iterable, Iterator

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


TITLE_CHUNK_KEY = "title"
"""Reserved chunk key: the item title, indexed as its own chunk so it is searchable."""


class RecordError(Exception):
    """Invalid record line in a JSONL stream."""


class ChunkInput(BaseModel):
    """One unit of text to embed and index, with optional key and metadata."""

    # extra="ignore" would let a record whose body used the wrong key validate clean and index nothing.
    model_config = ConfigDict(extra="forbid")

    text: str
    key: str | None = None
    meta: dict | None = None
    embedding: str | None = None  # base64 packed float32 (export/import wire only)


class LocationInput(BaseModel):
    """A known location of an item's source, with the machine that holds it."""

    model_config = ConfigDict(extra="forbid")

    uri: str
    machine: str | None = None


class Record(BaseModel):
    """A source item plus its chunks. `source_uri` is the only required field.

    The trailing optional fields are the export/import wire format: an export
    line is a Record plus provenance, so `import` reuses the ingest machinery.
    """

    model_config = ConfigDict(extra="forbid")

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

    @model_validator(mode="after")
    def _must_have_content(self):
        """Reject records with nothing to index; they all hash alike and merge into each other."""
        if not any(c.text.strip() for c in self.chunks) and not (self.title or "").strip():
            raise ValueError("record has no indexable content: needs a non-empty 'chunks' entry or a 'title'")
        return self


def iter_jsonl_records(lines: Iterable[str]) -> Iterator[Record]:
    for i, line in enumerate(lines, 1):
        if not (line := line.strip()):
            continue
        try:
            yield Record.model_validate_json(line)
        except ValidationError as e:
            raise RecordError(f"invalid record on line {i}:\n{e}") from e
