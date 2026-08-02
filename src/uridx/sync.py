"""Export/import of the index as JSONL for multi-machine federation.

An export line is a Record plus provenance (timestamps, content_hash, locations,
optional embeddings), so import reuses the ingest machinery: content-hash upsert
dedups across machines, and preserved updated_at values keep re-imports from
marking items dirty (no sync echo).
"""

import base64
import json
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime

from sqlalchemy.orm import selectinload
from sqlmodel import select

from uridx.db.engine import get_session, get_setting
from uridx.db.models import Item
from uridx.db.operations import IngestStatus, ingest_records, read_chunk_embeddings
from uridx.embeddings import deserialize_embedding
from uridx.record import ChunkInput, LocationInput, Record

EXPORT_PAGE_ITEMS = 100


def _embed_settings() -> tuple[str | None, int | None]:
    """The model and dimension the stored vectors were embedded with."""
    dim = get_setting("embed_dimension")
    return get_setting("embed_model"), int(dim) if dim else None


def iter_export_records(
    since: datetime | None = None,
    source_prefix: str | None = None,
    source_type: str | None = None,
    with_embeddings: bool = True,
) -> Iterator[Record]:
    """Yield every matching item as a Record. `since` is inclusive (>=).

    Embeddings are read back from the vector table and stamped with the model
    recorded in settings (not the env), so export stays correct mid-migration.
    Items are paged by id so streaming a large index stays flat in memory.
    """
    embed_model, embed_dim = _embed_settings() if with_embeddings else (None, None)
    filters = []
    if since is not None:
        filters.append(Item.updated_at >= since)
    if source_prefix:
        filters.append(Item.source_uri.startswith(source_prefix))
    if source_type:
        filters.append(Item.source_type == source_type)

    last_id = 0
    while True:
        with get_session() as session:
            items = session.exec(
                select(Item)
                .where(Item.id > last_id, *filters)
                .order_by(Item.id)
                .limit(EXPORT_PAGE_ITEMS)
                .options(selectinload(Item.chunks), selectinload(Item.tags), selectinload(Item.locations))
            ).all()
            if not items:
                return
            last_id = items[-1].id
            blobs = (
                read_chunk_embeddings(session, [c.id for item in items for c in item.chunks]) if with_embeddings else {}
            )
            page = [_export_record(item, blobs, embed_model, embed_dim) for item in items]
        yield from page


def _export_record(item: Item, blobs: dict[int, bytes], embed_model: str | None, embed_dim: int | None) -> Record:
    """One item as an export line: its chunks in order, plus provenance."""
    chunks = [
        ChunkInput(
            text=c.text,
            key=c.chunk_key,
            meta=json.loads(c.meta) if c.meta else None,
            embedding=base64.b64encode(blobs[c.id]).decode() if c.id in blobs else None,
        )
        for c in sorted(item.chunks, key=lambda c: c.chunk_index or 0)
    ]
    has_embeddings = any(c.embedding for c in chunks)
    return Record(
        source_uri=item.source_uri,
        title=item.title,
        source_type=item.source_type,
        context=item.context,
        tags=[t.tag for t in item.tags],
        created_at=item.created_at.isoformat(),
        updated_at=item.updated_at.isoformat(),
        content_hash=item.content_hash,
        locations=[LocationInput(uri=loc.uri, machine=loc.machine) for loc in item.locations] or None,
        chunks=chunks,
        embed_model=embed_model if has_embeddings else None,
        embed_dim=embed_dim if has_embeddings else None,
    )


def _decode_embeddings(record: Record, model_id: str | None, dim: int | None) -> list[list[float]] | None:
    """Decode a record's embeddings iff they match the local model; None means recompute."""
    if not model_id or not dim or record.embed_model != model_id or record.embed_dim != dim:
        return None
    if not record.chunks or any(c.embedding is None for c in record.chunks):
        return None
    try:
        return [deserialize_embedding(base64.b64decode(c.embedding), dim) for c in record.chunks]
    except Exception:
        return None


def import_records(
    records: Iterable[Record],
    on_result: Callable[[Record, IngestStatus], None] | None = None,
) -> dict[IngestStatus, int]:
    """Ingest an export stream, passing embeddings through when the model matches."""
    model_id, dim = _embed_settings()
    return ingest_records(records, on_result=on_result, precomputed=lambda r: _decode_embeddings(r, model_id, dim))
