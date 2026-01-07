import hashlib
import json
import sys
from datetime import datetime

from sqlmodel import func, select

from uridx.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL
from uridx.db.engine import get_raw_connection, get_session
from uridx.db.models import Chunk, Item, Tag
from uridx.embeddings.ollama import get_embeddings_sync, serialize_embedding


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


def add_item(
    source_uri: str,
    title: str | None = None,
    source_type: str | None = None,
    context: str | None = None,
    chunks: list[dict] | None = None,
    tags: list[str] | None = None,
    expires_at: datetime | None = None,
    replace: bool = False,  # kept for API compatibility, ignored
) -> Item:
    chunks = chunks or []
    tags = tags or []
    new_hash = compute_content_hash(chunks)

    with get_session() as session:
        existing = session.exec(select(Item).where(Item.source_uri == source_uri)).first()

        if existing and existing.content_hash == new_hash:
            print(f"  Skipping {source_uri} (unchanged)", file=sys.stderr)
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
            _ = existing.chunks
            _ = existing.tags
            session.expunge(existing)
            return existing

        if existing:
            _delete_chunk_embeddings([c.id for c in existing.chunks])
            for chunk in list(existing.chunks):
                session.delete(chunk)
            for tag in list(existing.tags):
                session.delete(tag)
            session.delete(existing)
            session.commit()

        item = Item(
            source_uri=source_uri,
            title=title,
            source_type=source_type,
            context=context,
            content_hash=new_hash,
            expires_at=expires_at,
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

        session.commit()

        if texts_to_embed:
            embeddings = get_embeddings_sync(texts_to_embed, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
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
        _ = item.chunks
        _ = item.tags
        session.expunge(item)
        return item


def delete_item(source_uri: str) -> bool:
    with get_session() as session:
        item = session.exec(select(Item).where(Item.source_uri == source_uri)).first()
        if not item:
            return False

        _delete_chunk_embeddings([c.id for c in item.chunks])
        session.delete(item)
        session.commit()
        return True


def get_item(source_uri: str) -> Item | None:
    with get_session() as session:
        item = session.exec(select(Item).where(Item.source_uri == source_uri)).first()
        if not item:
            return None

        _ = item.chunks
        _ = item.tags
        session.expunge(item)
        return item


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
