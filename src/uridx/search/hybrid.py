from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlmodel import select

from uridx.config import URIDX_MIN_SCORE
from uridx.db.engine import get_raw_connection, get_session
from uridx.db.models import Chunk, Item, Tag
from uridx.embeddings import get_embeddings_sync, serialize_embedding
from uridx.search.query import process_query


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
    """Search FTS5 index using extracted keywords from the query."""
    terms = process_query(query)
    if not terms.fts_query:
        return []
    cursor.execute(
        "SELECT rowid, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
        (terms.fts_query, limit * 3),
    )
    return cursor.fetchall()


def _rrf(result_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Each input list is [(doc_id, score)] where lower score = better for vector
    distance and more-negative = better for BM25. Results are sorted by their
    original score before ranking.
    """
    fused: dict[int, float] = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    if not fused:
        return []
    # Normalize to 0-1 range
    max_score = max(fused.values())
    if max_score > 0:
        fused = {doc_id: score / max_score for doc_id, score in fused.items()}
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


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
    bm25_weight: float | None = None,  # kept for API compat, not used with RRF
) -> list[SearchResult]:
    if min_score is None:
        min_score = URIDX_MIN_SCORE

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

        ranked_chunks = _rrf([vec_results, fts_results])
    else:
        fts_results = _fts_search(cursor, query, limit)
        ranked_chunks = _rrf([fts_results])

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
