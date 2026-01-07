import json
from datetime import datetime

from sqlmodel import func, select

from uridx.config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL
from uridx.db.engine import get_raw_connection, get_session
from uridx.db.models import Chunk, Item, Tag
from uridx.embeddings.ollama import get_embeddings_sync, serialize_embedding


def add_item(
    source_uri: str,
    title: str | None = None,
    source_type: str | None = None,
    context: str | None = None,
    chunks: list[dict] | None = None,
    tags: list[str] | None = None,
    expires_at: datetime | None = None,
    replace: bool = False,
) -> Item:
    chunks = chunks or []
    tags = tags or []

    with get_session() as session:
        existing = session.exec(select(Item).where(Item.source_uri == source_uri)).first()

        if existing and replace:
            delete_item(source_uri)
            existing = None

        if existing:
            existing.title = title
            existing.source_type = source_type
            existing.context = context
            existing.expires_at = expires_at
            existing.updated_at = datetime.utcnow()

            existing_keys = {c.chunk_key: c for c in existing.chunks if c.chunk_key}
            new_chunk_keys = {c.get("key") for c in chunks if c.get("key")}

            chunks_to_delete = [c for c in existing.chunks if c.chunk_key and c.chunk_key not in new_chunk_keys]
            chunk_ids_to_delete = [c.id for c in chunks_to_delete]

            if chunk_ids_to_delete:
                conn = get_raw_connection()
                cursor = conn.cursor()
                placeholders = ",".join("?" * len(chunk_ids_to_delete))
                cursor.execute(
                    f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
                    chunk_ids_to_delete,
                )
                conn.commit()
                conn.close()

                for c in chunks_to_delete:
                    session.delete(c)

            chunks_to_embed = []
            chunk_records = []

            for idx, chunk_data in enumerate(chunks):
                text = chunk_data["text"]
                key = chunk_data.get("key")
                meta = chunk_data.get("meta")

                if key and key in existing_keys:
                    chunk_record = existing_keys[key]
                    chunk_record.text = text
                    chunk_record.meta = json.dumps(meta) if meta else None
                    chunk_record.chunk_index = idx
                else:
                    chunk_record = Chunk(
                        item_id=existing.id,
                        chunk_key=key,
                        chunk_index=idx,
                        text=text,
                        meta=json.dumps(meta) if meta else None,
                    )
                    session.add(chunk_record)

                chunks_to_embed.append(text)
                chunk_records.append(chunk_record)

            session.flush()

            for tag in existing.tags:
                session.delete(tag)

            for tag_name in tags:
                session.add(Tag(item_id=existing.id, tag=tag_name))

            session.commit()

            if chunks_to_embed:
                embeddings = get_embeddings_sync(chunks_to_embed, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
                conn = get_raw_connection()
                cursor = conn.cursor()
                for chunk_record, embedding in zip(chunk_records, embeddings):
                    cursor.execute(
                        "DELETE FROM chunk_embeddings WHERE chunk_id = ?",
                        (chunk_record.id,),
                    )
                    cursor.execute(
                        "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                        (chunk_record.id, serialize_embedding(embedding)),
                    )
                conn.commit()
                conn.close()

            session.refresh(existing)
            _ = existing.chunks
            _ = existing.tags
            session.expunge(existing)
            return existing

        item = Item(
            source_uri=source_uri,
            title=title,
            source_type=source_type,
            context=context,
            expires_at=expires_at,
        )
        session.add(item)
        session.flush()

        chunk_records = []
        texts_to_embed = []

        for idx, chunk_data in enumerate(chunks):
            text = chunk_data["text"]
            key = chunk_data.get("key")
            meta = chunk_data.get("meta")

            chunk_record = Chunk(
                item_id=item.id,
                chunk_key=key,
                chunk_index=idx,
                text=text,
                meta=json.dumps(meta) if meta else None,
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

        chunk_ids = [c.id for c in item.chunks]

        if chunk_ids:
            conn = get_raw_connection()
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(chunk_ids))
            cursor.execute(
                f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            conn.commit()
            conn.close()

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
