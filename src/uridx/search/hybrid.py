from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlmodel import select

from uridx.config import URIDX_BM25_WEIGHT, URIDX_MIN_SCORE
from uridx.db.engine import get_raw_connection, get_session
from uridx.db.models import Chunk, Item, Tag
from uridx.embeddings import get_embeddings_sync, serialize_embedding


@dataclass
class SearchResult:
    source_uri: str
    title: str | None
    source_type: str | None
    chunk_text: str
    score: float
    tags: list[str] = field(default_factory=list)
    created_at: datetime | None = None


def escape_fts_query(query: str) -> str:
    """Escape a query string for FTS5 by treating it as a phrase search."""
    return f'"{query.replace('"', '""')}"'


def _fts_search(cursor, query: str, limit: int) -> list[tuple[int, float]]:
    cursor.execute(
        "SELECT rowid, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
        (escape_fts_query(query), limit * 3),
    )
    results = cursor.fetchall()
    if not results:
        return []
    fts_min = min(r[1] for r in results) or -1.0
    return [(rowid, score / fts_min if fts_min != 0 else 0.0) for rowid, score in results]


def hybrid_search(
    query: str,
    limit: int = 10,
    source_type: str | None = None,
    tags: list[str] | None = None,
    semantic: bool = True,
    recency_boost: float = 0.3,
    min_score: float | None = None,
    source_prefix: str | None = None,
    after: datetime | None = None,
    bm25_weight: float | None = None,
) -> list[SearchResult]:
    if min_score is None:
        min_score = URIDX_MIN_SCORE
    if bm25_weight is None:
        bm25_weight = URIDX_BM25_WEIGHT
    bm25_weight = max(0.0, min(1.0, bm25_weight))
    vector_weight = 1.0 - bm25_weight

    conn = get_raw_connection()
    cursor = conn.cursor()

    if semantic:
        query_embedding = get_embeddings_sync([query])[0]
        embedding_blob = serialize_embedding(query_embedding)

        cursor.execute(
            "SELECT chunk_id, distance FROM chunk_embeddings WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (embedding_blob, limit * 3),
        )
        vec_results = cursor.fetchall()

        fts_results = _fts_search(cursor, query, limit)

        vec_scores = {}
        if vec_results:
            max_dist = max(r[1] for r in vec_results) or 1.0
            for chunk_id, distance in vec_results:
                vec_scores[chunk_id] = 1.0 - (distance / max_dist)

        fts_scores = dict(fts_results)

        all_chunk_ids = set(vec_scores.keys()) | set(fts_scores.keys())
        combined_scores = {}
        for chunk_id in all_chunk_ids:
            v_score = vec_scores.get(chunk_id, 0.0)
            f_score = fts_scores.get(chunk_id, 0.0)
            combined_scores[chunk_id] = vector_weight * v_score + bm25_weight * f_score

        ranked_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    else:
        ranked_chunks = _fts_search(cursor, query, limit)

    conn.close()

    if not ranked_chunks:
        return []

    chunk_ids = [c[0] for c in ranked_chunks]
    score_map = {c[0]: c[1] for c in ranked_chunks}

    with get_session() as session:
        stmt = select(Chunk, Item).join(Item, Chunk.item_id == Item.id).where(Chunk.id.in_(chunk_ids))

        if source_type:
            stmt = stmt.where(Item.source_type == source_type)
        if source_prefix:
            stmt = stmt.where(Item.source_uri.startswith(source_prefix))
        if after:
            stmt = stmt.where(Item.created_at >= after)

        results_data = session.exec(stmt).all()

        results_with_tags = []
        for chunk, item in results_data:
            item_tags = session.exec(select(Tag).where(Tag.item_id == item.id)).all()
            if tags:
                item_tag_names = {t.tag for t in item_tags}
                if not all(t in item_tag_names for t in tags):
                    continue
            results_with_tags.append((chunk, item, item_tags))

        results = []
        for chunk, item, item_tags in results_with_tags:
            results.append(
                SearchResult(
                    source_uri=item.source_uri,
                    title=item.title,
                    source_type=item.source_type,
                    chunk_text=chunk.text,
                    score=score_map.get(chunk.id, 0.0),
                    tags=[t.tag for t in item_tags],
                    created_at=item.created_at,
                )
            )

        if recency_boost > 0 and results:
            now = datetime.now(timezone.utc)
            max_age_days = 365.0
            for r in results:
                if r.created_at:
                    created = r.created_at.replace(tzinfo=timezone.utc) if r.created_at.tzinfo is None else r.created_at
                    age_days = (now - created).total_seconds() / 86400
                    age_score = max(0.0, 1.0 - (age_days / max_age_days))
                else:
                    age_score = 0.0
                r.score = r.score * (1 - recency_boost) + age_score * recency_boost

        results.sort(key=lambda x: x.score, reverse=True)
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        return results[:limit]
