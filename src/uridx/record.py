"""Standard Record model shared by the extractors and ingester."""

from pydantic import BaseModel, Field


class ChunkInput(BaseModel):
    """One unit of text to embed and index, with optional key and metadata."""

    text: str
    key: str | None = None
    meta: dict | None = None


class Record(BaseModel):
    """A source item plus its chunks. `source_uri` is the only required field."""

    source_uri: str
    chunks: list[ChunkInput] = Field(default_factory=list)
    title: str | None = None
    source_type: str | None = None
    context: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str | None = None
    machine: str | None = None
