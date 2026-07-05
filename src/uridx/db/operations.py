import hashlib
import json
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Literal

from sqlmodel import func, select

from uridx.config import get_machine_id
from uridx.db.engine import create_embeddings_table, get_raw_connection, get_session
from uridx.db.models import Chunk, Item, Location, Setting, Tag
from uridx.embeddings import get_dimension, get_embeddings_sync, get_model_id, serialize_embedding
from uridx.record import ChunkInput, Record

IngestStatus = Literal["added", "unchanged", "merged"]

EMBED_BATCH_CHUNKS = 64


def compute_content_hash(chunks: list[ChunkInput]) -> str:
    """Compute SHA256 hash of chunk texts for change detection."""
    content = "\n".join(c.text for c in chunks)
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _delete_chunk_embeddings(chunk_ids: list[int]) -> None:
    if not chunk_ids:
        return
    conn = get_raw_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" * len(chunk_ids))
    cursor.execute(f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)
    conn.commit()
    conn.close()


def _insert_chunk_embeddings(cursor, pairs: Iterable[tuple[int, list[float]]]) -> None:
    cursor.executemany(
        "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
        [(chunk_id, serialize_embedding(embedding)) for chunk_id, embedding in pairs],
    )


def _ensure_location(session, item_id: int, uri: str, machine: str | None) -> None:
    existing = session.exec(select(Location).where(Location.item_id == item_id, Location.uri == uri)).first()
    if not existing:
        session.add(Location(item_id=item_id, uri=uri, machine=machine))
    elif machine and existing.machine != machine:
        existing.machine = machine


def _detach_item(session, item: Item) -> Item:
    """Load relationships and detach item from session for safe return."""
    _ = item.chunks
    _ = item.tags
    _ = item.locations
    session.expunge(item)
    return item


def add_item(
    source_uri: str,
    title: str | None = None,
    source_type: str | None = None,
    context: str | None = None,
    chunks: list[ChunkInput] | None = None,
    tags: list[str] | None = None,
    expires_at: datetime | None = None,
    created_at: datetime | str | None = None,
    machine: str | None = None,
    embeddings: list[list[float]] | None = None,
    content_hash: str | None = None,
) -> tuple[Item, IngestStatus]:
    """Upsert an item; returns (item, status).

    Pass precomputed embeddings (one per chunk) and/or content_hash to skip
    recomputing them per item.
    """
    chunks = chunks or []
    tags = list(dict.fromkeys(tags or []))
    new_hash = content_hash or compute_content_hash(chunks)

    # Parse created_at if string
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

    with get_session() as session:
        existing = session.exec(select(Item).where(Item.source_uri == source_uri)).first()

        if existing and existing.content_hash == new_hash:
            _ensure_location(session, existing.id, source_uri, machine)
            existing.title = title
            existing.source_type = source_type
            existing.context = context
            existing.expires_at = expires_at
            existing.updated_at = datetime.utcnow()

            existing_tags = {t.tag for t in existing.tags}
            if existing_tags != set(tags):
                for tag in list(existing.tags):
                    session.delete(tag)
                for tag_name in tags:
                    session.add(Tag(item_id=existing.id, tag=tag_name))

            session.commit()
            session.refresh(existing)
            return _detach_item(session, existing), "unchanged"

        if existing:
            _delete_item_in_session(session, existing)
            session.commit()

        # Check if content_hash matches another item (merge case)
        hash_match = session.exec(select(Item).where(Item.content_hash == new_hash)).first()
        if hash_match:
            _ensure_location(session, hash_match.id, source_uri, machine)
            existing_tags = {t.tag for t in hash_match.tags}
            for tag_name in tags:
                if tag_name not in existing_tags:
                    session.add(Tag(item_id=hash_match.id, tag=tag_name))
            hash_match.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(hash_match)
            return _detach_item(session, hash_match), "merged"

        item = Item(
            source_uri=source_uri,
            title=title,
            source_type=source_type,
            context=context,
            content_hash=new_hash,
            expires_at=expires_at,
            created_at=created_at or datetime.utcnow(),
            updated_at=created_at or datetime.utcnow(),
        )
        session.add(item)
        session.flush()

        chunk_records = []
        texts_to_embed = []

        for idx, chunk_data in enumerate(chunks):
            chunk_record = Chunk(
                item_id=item.id,
                chunk_key=chunk_data.key,
                chunk_index=idx,
                text=chunk_data.text,
                meta=json.dumps(chunk_data.meta) if chunk_data.meta else None,
            )
            session.add(chunk_record)
            chunk_records.append(chunk_record)
            texts_to_embed.append(chunk_data.text)

        session.flush()

        for tag_name in tags:
            session.add(Tag(item_id=item.id, tag=tag_name))

        session.add(Location(item_id=item.id, uri=source_uri, machine=machine))

        session.commit()

        if texts_to_embed:
            if embeddings is None:
                embeddings = get_embeddings_sync(texts_to_embed)
            conn = get_raw_connection()
            _insert_chunk_embeddings(conn.cursor(), zip((c.id for c in chunk_records), embeddings))
            conn.commit()
            conn.close()

        session.refresh(item)
        return _detach_item(session, item), "added"


def ingest_record(
    record: Record,
    *,
    extra_tags: list[str] | None = None,
    default_machine: str | None = None,
    embeddings: list[list[float]] | None = None,
    content_hash: str | None = None,
) -> tuple[Item, IngestStatus]:
    """Ingest one Record (the contract extractors emit) via add_item.

    Does NOT init the DB — the caller must call init_db() once before looping.
    """
    tags = (record.tags + (extra_tags or [])) or None  # add_item dedups
    machine = record.machine or default_machine or get_machine_id()
    return add_item(
        source_uri=record.source_uri,
        title=record.title,
        context=record.context,
        source_type=record.source_type,
        tags=tags,
        chunks=record.chunks,
        created_at=record.created_at,
        machine=machine,
        embeddings=embeddings,
        content_hash=content_hash,
    )


def ingest_records(
    records: Iterable[Record],
    *,
    extra_tags: list[str] | None = None,
    on_result: Callable[[Record, IngestStatus], None] | None = None,
    batch_chunks: int = EMBED_BATCH_CHUNKS,
) -> dict[IngestStatus, int]:
    """Ingest a record stream, batching embedding calls across records.

    Buffers records until ~batch_chunks chunks accumulate, then embeds all new
    content in one call: one query partitions the batch into known content (hash
    already indexed — no embedding) and new content (embedded once per unique
    hash, so intra-batch duplicates aren't re-embedded). Shared by `ingest` and
    `add`. Returns counts by status; on_result fires per record as it lands.
    """
    counts: dict[IngestStatus, int] = {"added": 0, "unchanged": 0, "merged": 0}
    batch: list[tuple[Record, str]] = []

    def flush():
        if not batch:
            return
        hashes = {h for _, h in batch}
        with get_session() as session:
            known = set(session.exec(select(Item.content_hash).where(Item.content_hash.in_(hashes))).all())
        to_embed = {h: rec for rec, h in batch if h not in known}  # one representative per unique new hash
        texts = [c.text for rec in to_embed.values() for c in rec.chunks]
        embedded = iter(get_embeddings_sync(texts)) if texts else iter(())
        embeddings_by_hash = {h: [next(embedded) for _ in rec.chunks] for h, rec in to_embed.items()}
        for rec, h in batch:
            _, status = ingest_record(rec, extra_tags=extra_tags, embeddings=embeddings_by_hash.get(h), content_hash=h)
            counts[status] += 1
            if on_result:
                on_result(rec, status)
        batch.clear()

    try:
        for record in records:
            batch.append((record, compute_content_hash(record.chunks)))
            if sum(len(rec.chunks) for rec, _ in batch) >= batch_chunks:
                flush()
    finally:
        flush()  # also persists buffered work when the record stream raises
    return counts


def _delete_item_in_session(session, item: Item) -> None:
    _delete_chunk_embeddings([c.id for c in item.chunks])
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
        all_chunk_ids = [c.id for item in items for c in item.chunks]
        _delete_chunk_embeddings(all_chunk_ids)
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
    conn = get_raw_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS chunk_embeddings")
    create_embeddings_table(cursor, dim)
    conn.commit()

    with get_session() as session:
        session.merge(Setting(key="embed_dimension", value=str(dim)))
        session.merge(Setting(key="embed_model", value=get_model_id()))
        session.commit()

    total = cursor.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    done = 0
    last_id = 0
    while True:
        rows = cursor.execute(
            "SELECT id, text FROM chunk WHERE id > ? ORDER BY id LIMIT ?", (last_id, batch_size)
        ).fetchall()
        if not rows:
            break
        embeddings = get_embeddings_sync([text for _, text in rows])
        _insert_chunk_embeddings(cursor, zip((chunk_id for chunk_id, _ in rows), embeddings))
        conn.commit()
        last_id = rows[-1][0]
        done += len(rows)
        if on_progress:
            on_progress(done, total)
    conn.close()
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
