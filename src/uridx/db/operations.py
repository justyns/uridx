"""Reading and writing the index.

Two merge policies run through the write path, selected by whether an incoming record
carries an `updated_at`:

- **local** (no `updated_at`) — this machine just read the source, so it is
  authoritative: metadata is overwritten, the tag set is replaced wholesale, and
  `updated_at` bumps to now whenever something actually changed.
- **sync** (`updated_at` present) — the record came from another machine's index, so
  metadata competes on timestamp (last writer wins) and tags union. A peer that never
  saw a tag must not delete it, and re-importing unchanged data must not mark the item
  dirty, or every sync would echo back out on the next export.
"""

import hashlib
import json
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Literal

from sqlmodel import func, select

from uridx.config import get_machine_id
from uridx.db.engine import get_session
from uridx.db.models import Chunk, Item, Location, Setting, Tag
from uridx.db.schema import create_embeddings_tables
from uridx.embeddings import get_dimension, get_embeddings_sync, get_model_id, serialize_embedding
from uridx.record import TITLE_CHUNK_KEY, ChunkInput, Record
from uridx.timeutil import to_naive_utc, utcnow

IngestStatus = Literal["added", "unchanged", "merged"]

EMBED_BATCH_CHUNKS = 64


def compute_content_hash(chunks: list[ChunkInput]) -> str:
    """Compute SHA256 hash of chunk texts for change detection."""
    content = "\n".join(c.text for c in chunks)
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _delete_chunk_embeddings(session, chunk_ids: list[int]) -> None:
    """Drop both vector rows for these chunks, in the caller's transaction."""
    if not chunk_ids:
        return
    conn = session.connection()
    placeholders = ",".join("?" * len(chunk_ids))
    params = tuple(chunk_ids)
    conn.exec_driver_sql(f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})", params)
    conn.exec_driver_sql(f"DELETE FROM chunk_embeddings_bq WHERE chunk_id IN ({placeholders})", params)


def _insert_chunk_embeddings(session, pairs: Iterable[tuple[int, list[float]]]) -> None:
    """Write the float vectors and their derived bit vectors, in the caller's transaction."""
    rows = [(chunk_id, serialize_embedding(embedding)) for chunk_id, embedding in pairs]
    if not rows:
        return
    conn = session.connection()
    conn.exec_driver_sql("INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)", rows)
    conn.exec_driver_sql(
        "INSERT INTO chunk_embeddings_bq (chunk_id, embedding) VALUES (?, vec_quantize_binary(?))", rows
    )


def read_chunk_embeddings(session, chunk_ids: list[int]) -> dict[int, bytes]:
    """Read stored embedding blobs by chunk id (export pass-through)."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = session.connection().exec_driver_sql(
        f"SELECT chunk_id, embedding FROM chunk_embeddings WHERE chunk_id IN ({placeholders})", tuple(chunk_ids)
    )
    return {chunk_id: blob for chunk_id, blob in rows}


def _apply_locations(session, item: Item, record: Record) -> bool:
    """Add missing locations and backfill null machines; never overwrite a known origin.

    A record that declares its own locations already carries its provenance, so the
    local hostname is not stamped over it. Returns True if anything changed.
    """
    machine = record.machine or (None if record.locations else get_machine_id())
    wanted: dict[str, str | None] = {record.source_uri: machine}
    for loc in record.locations or []:
        wanted[loc.uri] = wanted.get(loc.uri) or loc.machine
    existing = {loc.uri: loc for loc in session.exec(select(Location).where(Location.item_id == item.id))}
    changed = False
    for uri, mach in wanted.items():
        row = existing.get(uri)
        if row is None:
            session.add(Location(item_id=item.id, uri=uri, machine=mach))
            changed = True
        elif mach and row.machine is None:
            row.machine = mach
            changed = True
    return changed


def _set_item_meta(item: Item, record: Record) -> bool:
    """Copy the record's metadata onto the item; returns True if any field changed."""
    incoming = (record.title, record.source_type, record.context)
    changed = (item.title, item.source_type, item.context) != incoming
    item.title, item.source_type, item.context = incoming
    return changed


def _reconcile_tags(session, item: Item, tags: list[str], *, replace: bool) -> bool:
    """Bring the item's tags in line with `tags`; returns True if anything changed.

    Local writes own the tag set outright (replace); sync writes union, so importing
    from a peer that never saw a tag can't delete it. See the module docstring.
    """
    existing = {t.tag for t in item.tags}
    if not replace:
        missing = [t for t in tags if t not in existing]
        for tag_name in missing:
            session.add(Tag(item_id=item.id, tag=tag_name))
        return bool(missing)

    if existing == set(tags):
        return False
    for tag in list(item.tags):
        session.delete(tag)
    for tag_name in tags:
        session.add(Tag(item_id=item.id, tag=tag_name))
    return True


def _advance_updated_at(item: Item, incoming: datetime | None, changed: bool) -> None:
    """Move the item's clock forward under whichever policy applies.

    Local writes bump to now only when something actually changed; sync writes take
    the incoming stamp but never move backwards, so a repeated import of an unchanged
    record leaves the item clean and doesn't echo back out on the next export.
    """
    if incoming is not None:
        item.updated_at = max(item.updated_at, incoming)
    elif changed:
        item.updated_at = utcnow()


def _detach_item(session, item: Item) -> Item:
    """Load relationships and detach item from session for safe return."""
    _ = item.chunks
    _ = item.tags
    _ = item.locations
    session.expunge(item)
    return item


def _touch_existing(session, item: Item, record: Record, tags: list[str], incoming: datetime | None) -> Item:
    """Same URI, same content: reconcile metadata, tags and locations without re-chunking."""
    changed = _apply_locations(session, item, record)
    if incoming is None:
        changed = _set_item_meta(item, record) or changed
        changed = _reconcile_tags(session, item, tags, replace=True) or changed
    else:
        if incoming > item.updated_at:
            _set_item_meta(item, record)
        _reconcile_tags(session, item, tags, replace=False)
    _advance_updated_at(item, incoming, changed)
    return item


def _merge_into(session, item: Item, record: Record, tags: list[str], incoming: datetime | None) -> Item:
    """Same content under a new URI: record the extra location, union tags, leave metadata alone."""
    changed = _apply_locations(session, item, record)
    changed = _reconcile_tags(session, item, tags, replace=False) or changed
    _advance_updated_at(item, incoming, changed)
    return item


def _create_item(
    session,
    record: Record,
    tags: list[str],
    content_hash: str,
    incoming: datetime | None,
    embeddings: list[list[float]] | None,
) -> Item:
    """Insert the item, its chunks, tags, locations and vectors as one transaction."""
    created_at = to_naive_utc(record.created_at)
    item = Item(
        source_uri=record.source_uri,
        title=record.title,
        source_type=record.source_type,
        context=record.context,
        content_hash=content_hash,
        created_at=created_at or utcnow(),
        updated_at=incoming or created_at or utcnow(),
    )
    session.add(item)
    session.flush()

    chunks = [
        Chunk(
            item_id=item.id,
            chunk_key=chunk.key,
            chunk_index=idx,
            text=chunk.text,
            meta=json.dumps(chunk.meta) if chunk.meta else None,
        )
        for idx, chunk in enumerate(record.chunks)
    ]
    session.add_all(chunks)
    session.flush()

    for tag_name in tags:
        session.add(Tag(item_id=item.id, tag=tag_name))
    _apply_locations(session, item, record)

    if chunks:
        if embeddings is None:
            embeddings = get_embeddings_sync([c.text for c in chunks])
        _insert_chunk_embeddings(session, zip((c.id for c in chunks), embeddings))
    return item


def prepare_record(record: Record) -> str:
    """Normalize a record for indexing and return its content hash.

    Prepends the title as its own chunk so it's searchable (FTS + vector), unless one
    is already there (import round-trips). Runs before hashing, so retitling an item
    counts as a content change.
    """
    if record.title and not any(c.key == TITLE_CHUNK_KEY for c in record.chunks):
        record.chunks.insert(0, ChunkInput(text=record.title, key=TITLE_CHUNK_KEY))
    return compute_content_hash(record.chunks)


def ingest_record(
    record: Record,
    *,
    extra_tags: list[str] | None = None,
    embeddings: list[list[float]] | None = None,
    content_hash: str | None = None,
) -> tuple[Item, IngestStatus]:
    """Upsert one Record — the single write path into the index. Returns (item, status).

    Three outcomes, keyed on content hash: the URI is already indexed with this exact
    content ("unchanged"), the content is already indexed under a different URI
    ("merged" — the new URI becomes another location of the one item), or it is new
    ("added"; a URI whose content changed is deleted and re-added).

    `record.updated_at` selects the merge policy — see the module docstring. Pass `embeddings`
    (one per chunk) and/or `content_hash` to reuse work the caller already did.

    Does NOT init the DB — the caller must call init_db() once before looping.
    """
    content_hash = content_hash or prepare_record(record)
    tags = list(dict.fromkeys(record.tags + (extra_tags or [])))
    incoming = to_naive_utc(record.updated_at)

    with get_session() as session:
        existing = session.exec(select(Item).where(Item.source_uri == record.source_uri)).first()
        if existing and existing.content_hash == content_hash:
            item, status = _touch_existing(session, existing, record, tags, incoming), "unchanged"
        else:
            if existing:
                _delete_item_in_session(session, existing)
                session.flush()
            twin = session.exec(select(Item).where(Item.content_hash == content_hash)).first()
            if twin:
                item, status = _merge_into(session, twin, record, tags, incoming), "merged"
            else:
                item, status = _create_item(session, record, tags, content_hash, incoming, embeddings), "added"

        session.commit()
        session.refresh(item)
        return _detach_item(session, item), status


def ingest_records(
    records: Iterable[Record],
    *,
    extra_tags: list[str] | None = None,
    on_result: Callable[[Record, IngestStatus], None] | None = None,
    batch_chunks: int = EMBED_BATCH_CHUNKS,
    precomputed: Callable[[Record], list[list[float]] | None] | None = None,
) -> dict[IngestStatus, int]:
    """Ingest a record stream, batching embedding calls across records.

    Buffers records until ~batch_chunks chunks accumulate, then embeds all new
    content in one call: one query partitions the batch into known content (hash
    already indexed — no embedding) and new content (embedded once per unique
    hash, so intra-batch duplicates aren't re-embedded). Shared by `ingest` and
    `add`. Returns counts by status; on_result fires per record as it lands.

    precomputed, if given, may supply ready-made vectors for a record (import
    pass-through); records it returns None for are batch-embedded as usual.

    Fast path: records whose (content_hash, source_uri) already exist are counted
    "unchanged" without touching the item at all — a re-run over an already-indexed
    source skips per-record DB work entirely. Disabled per record when extra_tags
    are given or the record carries sync provenance (updated_at/locations), since
    those must still merge into the existing item.
    """
    counts: dict[IngestStatus, int] = {"added": 0, "unchanged": 0, "merged": 0}
    batch: list[tuple[Record, str]] = []

    def flush():
        if not batch:
            return
        hashes = {h for _, h in batch}
        with get_session() as session:
            known_pairs = set(
                session.exec(select(Item.content_hash, Item.source_uri).where(Item.content_hash.in_(hashes))).all()
            )
        known = {h for h, _ in known_pairs}
        to_embed = {h: rec for rec, h in batch if h not in known}  # one representative per unique new hash
        embeddings_by_hash = {}
        if precomputed:
            embeddings_by_hash = {h: v for h, rec in to_embed.items() if (v := precomputed(rec)) is not None}
            to_embed = {h: rec for h, rec in to_embed.items() if h not in embeddings_by_hash}
        texts = [c.text for rec in to_embed.values() for c in rec.chunks]
        embedded = iter(get_embeddings_sync(texts)) if texts else iter(())
        embeddings_by_hash |= {h: [next(embedded) for _ in rec.chunks] for h, rec in to_embed.items()}
        for rec, h in batch:
            skippable = not extra_tags and rec.updated_at is None and rec.locations is None
            if skippable and (h, rec.source_uri) in known_pairs:
                status = "unchanged"
            else:
                _, status = ingest_record(
                    rec, extra_tags=extra_tags, embeddings=embeddings_by_hash.get(h), content_hash=h
                )
            counts[status] += 1
            if on_result:
                on_result(rec, status)
        batch.clear()

    try:
        for record in records:
            batch.append((record, prepare_record(record)))
            if sum(len(rec.chunks) for rec, _ in batch) >= batch_chunks:
                flush()
    finally:
        flush()  # also persists buffered work when the record stream raises
    return counts


def _delete_item_in_session(session, item: Item) -> None:
    _delete_chunk_embeddings(session, [c.id for c in item.chunks])
    session.delete(item)


def delete_item(source_uri: str) -> bool:
    with get_session() as session:
        item = session.exec(select(Item).where(Item.source_uri == source_uri)).first()
        if not item:
            return False
        _delete_item_in_session(session, item)
        session.commit()
        return True


def delete_items_by_prefix(prefix: str) -> int:
    if not prefix:
        raise ValueError("prefix must not be empty")
    with get_session() as session:
        items = session.exec(select(Item).where(Item.source_uri.startswith(prefix))).all()
        _delete_chunk_embeddings(session, [c.id for item in items for c in item.chunks])
        for item in items:
            session.delete(item)
        session.commit()
        return len(items)


def list_items_by_prefix(prefix: str) -> list[Item]:
    if not prefix:
        raise ValueError("prefix must not be empty")
    with get_session() as session:
        items = session.exec(select(Item).where(Item.source_uri.startswith(prefix))).all()
        for item in items:
            _detach_item(session, item)
        return items


def get_item(source_uri: str) -> Item | None:
    with get_session() as session:
        item = session.exec(select(Item).where(Item.source_uri == source_uri)).first()
        if not item:
            loc = session.exec(select(Location).where(Location.uri == source_uri)).first()
            if loc:
                item = session.get(Item, loc.item_id)
        if not item:
            return None

        return _detach_item(session, item)


def reindex_embeddings(batch_size: int = EMBED_BATCH_CHUNKS, on_progress=None) -> int:
    """Re-embed every chunk from its stored text with the current embedding backend/model.

    Drops and recreates the vector table at the backend's dimension (updating the
    embed_dimension and embed_model settings), so this is how you switch models.
    Items, chunks, tags, and FTS are untouched. Returns the number of chunks re-embedded.
    """
    dim = get_dimension()
    with get_session() as session:
        conn = session.connection()
        conn.exec_driver_sql("DROP TABLE IF EXISTS chunk_embeddings")
        conn.exec_driver_sql("DROP TABLE IF EXISTS chunk_embeddings_bq")
        create_embeddings_tables(conn)
        session.merge(Setting(key="embed_dimension", value=str(dim)))
        session.merge(Setting(key="embed_model", value=get_model_id()))
        session.commit()

        total = session.exec(select(func.count(Chunk.id))).one()
        done = 0
        last_id = 0
        while True:
            rows = session.exec(
                select(Chunk.id, Chunk.text).where(Chunk.id > last_id).order_by(Chunk.id).limit(batch_size)
            ).all()
            if not rows:
                break
            embeddings = get_embeddings_sync([text for _, text in rows])
            _insert_chunk_embeddings(session, zip((chunk_id for chunk_id, _ in rows), embeddings))
            session.commit()
            last_id = rows[-1][0]
            done += len(rows)
            if on_progress:
                on_progress(done, total)
        return done


def get_stats() -> dict:
    with get_session() as session:
        items_count = session.exec(select(func.count(Item.id))).one()
        chunks_count = session.exec(select(func.count(Chunk.id))).one()
        tags_count = session.exec(select(func.count()).select_from(Tag)).one()

        source_types_result = session.exec(
            select(Item.source_type, func.count(Item.id)).group_by(Item.source_type)
        ).all()

        source_types = {st or "unknown": count for st, count in source_types_result}

        return {
            "items": items_count,
            "chunks": chunks_count,
            "tags": tags_count,
            "source_types": source_types,
        }


def get_existing_source_uris(uris: list[str]) -> set[str]:
    """Return which source_uris already exist in the local DB."""
    if not uris:
        return set()
    with get_session() as session:
        existing = session.exec(select(Item.source_uri).where(Item.source_uri.in_(uris))).all()
        return set(existing)
