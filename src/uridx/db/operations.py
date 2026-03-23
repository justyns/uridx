import hashlib
import json
import sys
from datetime import datetime

from sqlmodel import func, select

from uridx.db.engine import get_raw_connection, get_session
from uridx.db.models import Chunk, Item, Location, Tag
from uridx.embeddings import get_embeddings_sync, serialize_embedding


def compute_content_hash(chunks: list[dict]) -> str:
    """Compute SHA256 hash of chunk texts for change detection."""
    content = "\n".join(c.get("text", "") for c in chunks)
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
    chunks: list[dict] | None = None,
    tags: list[str] | None = None,
    expires_at: datetime | None = None,
    replace: bool = False,  # kept for API compatibility, ignored
    created_at: datetime | str | None = None,
    machine: str | None = None,
) -> Item:
    chunks = chunks or []
    tags = tags or []
    new_hash = compute_content_hash(chunks)

    # Parse created_at if string
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

    with get_session() as session:
        existing = session.exec(select(Item).where(Item.source_uri == source_uri)).first()

        if existing and existing.content_hash == new_hash:
            print(f"  Skipping {source_uri} (unchanged)", file=sys.stderr)
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
            return _detach_item(session, existing)

        if existing:
            _delete_item_in_session(session, existing)
            session.commit()

        # Check if content_hash matches another item (merge case)
        hash_match = session.exec(select(Item).where(Item.content_hash == new_hash)).first()
        if hash_match:
            print(f"  Merging {source_uri} into {hash_match.source_uri}", file=sys.stderr)
            _ensure_location(session, hash_match.id, source_uri, machine)
            existing_tags = {t.tag for t in hash_match.tags}
            for tag_name in tags:
                if tag_name not in existing_tags:
                    session.add(Tag(item_id=hash_match.id, tag=tag_name))
            hash_match.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(hash_match)
            return _detach_item(session, hash_match)

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
            text = chunk_data["text"]
            chunk_record = Chunk(
                item_id=item.id,
                chunk_key=chunk_data.get("key"),
                chunk_index=idx,
                text=text,
                meta=json.dumps(chunk_data.get("meta")) if chunk_data.get("meta") else None,
            )
            session.add(chunk_record)
            chunk_records.append(chunk_record)
            texts_to_embed.append(text)

        session.flush()

        for tag_name in tags:
            session.add(Tag(item_id=item.id, tag=tag_name))

        session.add(Location(item_id=item.id, uri=source_uri, machine=machine))

        session.commit()

        if texts_to_embed:
            embeddings = get_embeddings_sync(texts_to_embed)
            conn = get_raw_connection()
            cursor = conn.cursor()
            for chunk_record, embedding in zip(chunk_records, embeddings):
                cursor.execute(
                    "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_record.id, serialize_embedding(embedding)),
                )
            conn.commit()
            conn.close()

        session.refresh(item)
        return _detach_item(session, item)


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


def list_recent(
    limit: int = 10,
    source_type: str | None = None,
    tags: list[str] | None = None,
) -> list[Item]:
    """List items sorted by updated_at (most recent first)."""
    with get_session() as session:
        query = select(Item).order_by(Item.updated_at.desc())

        if source_type:
            query = query.where(Item.source_type == source_type)

        items = session.exec(query).all()

        if tags:
            tag_set = set(tags)
            items = [i for i in items if tag_set <= {t.tag for t in i.tags}]

        items = items[:limit]

        for item in items:
            _detach_item(session, item)

        return items
