from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlmodel import select

from uridx.config import URIDX_MIN_SCORE, URIDX_RESCORE_K
from uridx.db.engine import get_session
from uridx.db.models import Chunk, Item, Tag
from uridx.embeddings import get_embeddings_sync, serialize_embedding
from uridx.record import TITLE_CHUNK_KEY
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


def _rrf(result_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Combine ranked result lists using Reciprocal Rank Fusion.

    Each input list is [(doc_id, score)] ordered by the source ranker; only the
    ordering is used, never the raw score. Returns [(doc_id, normalized_score)] in
    (0, 1] sorted descending. Also used with a single list, to put one ranker's
    output on that same scale — the rest of the pipeline (recency mixing, min_score,
    the final sort) assumes higher is better and O(1) magnitudes.
    """
    fused: dict[int, float] = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    if not fused:
        return []
    max_score = max(fused.values())
    fused = {doc_id: score / max_score for doc_id, score in fused.items()}
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _load_tags(session, item_ids: set[int]) -> dict[int, list[str]]:
    """Map each item_id to its list of tags."""
    tags_by_item: dict[int, list[str]] = {}
    for t in session.exec(select(Tag).where(Tag.item_id.in_(item_ids))).all():
        tags_by_item.setdefault(t.item_id, []).append(t.tag)
    return tags_by_item


def _apply_item_filters(
    stmt,
    source_type: str | None,
    source_prefix: str | None,
    after: datetime | None,
    tags: list[str] | None = None,
):
    """Apply the shared source_type / source_prefix / after / tags WHERE clauses to a Chunk+Item select."""
    if source_type:
        stmt = stmt.where(Item.source_type == source_type)
    if source_prefix:
        stmt = stmt.where(Item.source_uri.startswith(source_prefix))
    if after:
        stmt = stmt.where(Item.created_at >= after)
    for t in tags or []:
        stmt = stmt.where(Item.id.in_(select(Tag.item_id).where(Tag.tag == t)))
    return stmt


def _preview_chunk_id():
    """Correlated subquery for the chunk that best previews an item.

    Body before title (the title is already displayed on its own line, so previewing
    it would just repeat it), then lowest chunk_index. Always resolves as long as the
    item has any chunk at all.
    """
    return (
        select(Chunk.id)
        .where(Chunk.item_id == Item.id)
        .order_by((Chunk.chunk_key == TITLE_CHUNK_KEY).asc(), Chunk.chunk_index)
        .limit(1)
        .correlate(Item)
        .scalar_subquery()
    )


def _list_results(
    *,
    limit: int,
    source_type: str | None,
    tags: list[str] | None,
    source_prefix: str | None,
    after: datetime | None,
) -> list[SearchResult]:
    """Empty-query listing: items matching the filters, newest first, unranked.

    Lets `search` browse a slice of the index (e.g. a whole thread via
    `--tag thread:<id>`) without a ranking query.
    """
    with get_session() as session:
        stmt = select(Chunk, Item).join(Item, Chunk.item_id == Item.id).where(Chunk.id == _preview_chunk_id())
        stmt = _apply_item_filters(stmt, source_type, source_prefix, after, tags)
        stmt = stmt.order_by(Item.created_at.desc()).limit(limit)
        rows = session.exec(stmt).all()
        tags_by_item = _load_tags(session, {item.id for _, item in rows})

        return [
            SearchResult(
                source_uri=item.source_uri,
                title=item.title,
                source_type=item.source_type,
                chunk_text=chunk.text,
                score=0.0,
                tags=tags_by_item.get(item.id, []),
                created_at=item.created_at,
            )
            for chunk, item in rows
        ]


def _vector_search(conn, embedding_blob: bytes, limit: int) -> list[tuple[int, float]]:
    """Two-stage KNN: coarse hamming scan over sign-bit vectors, exact rescore of the survivors.

    The bit vectors are 32x smaller than the floats, so the full-table pass is cheap;
    only the URIDX_RESCORE_K coarse candidates pay full float distance math.
    """
    query_bits = conn.exec_driver_sql("SELECT vec_quantize_binary(?)", (embedding_blob,)).scalar()
    candidates = [
        row[0]
        for row in conn.exec_driver_sql(
            "SELECT chunk_id FROM chunk_embeddings_bq "
            "ORDER BY vec_distance_hamming(vec_bit(embedding), vec_bit(?)) LIMIT ?",
            (query_bits, max(URIDX_RESCORE_K, limit)),
        )
    ]
    if not candidates:
        return []
    placeholders = ",".join("?" * len(candidates))
    return conn.exec_driver_sql(
        f"SELECT chunk_id, vec_distance_l2(embedding, ?) AS distance FROM chunk_embeddings "
        f"WHERE chunk_id IN ({placeholders}) ORDER BY distance LIMIT ?",
        (embedding_blob, *candidates, limit),
    ).fetchall()


def _fts_search(conn, fts_query: str, limit: int) -> list[tuple[int, float]]:
    """Best-first BM25 matches. Note bm25() is negative and *descends* with relevance,
    so the rows are ordered ascending and the raw scores must be re-ranked before use."""
    if not fts_query:
        return []
    return conn.exec_driver_sql(
        "SELECT rowid, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
        (fts_query, limit * 3),
    ).fetchall()


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
) -> list[SearchResult]:
    if min_score is None:
        min_score = URIDX_MIN_SCORE

    if not query.strip():
        return _list_results(limit=limit, source_type=source_type, tags=tags, source_prefix=source_prefix, after=after)

    fts_query = process_query(query)
    query_blob = serialize_embedding(get_embeddings_sync([query])[0]) if semantic else None

    with get_session() as session:
        conn = session.connection()
        rankings = [_fts_search(conn, fts_query, limit)]
        if semantic:
            rankings.append(_vector_search(conn, query_blob, limit * 3))
        ranked_chunks = _rrf(rankings)
        if not ranked_chunks:
            return []

        chunk_ids = [c[0] for c in ranked_chunks]
        score_map = {c[0]: c[1] for c in ranked_chunks}

        stmt = select(Chunk, Item).join(Item, Chunk.item_id == Item.id).where(Chunk.id.in_(chunk_ids))
        stmt = _apply_item_filters(stmt, source_type, source_prefix, after, tags)

        results_data = session.exec(stmt).all()
        tags_by_item = _load_tags(session, {item.id for _, item in results_data})

        results = [
            SearchResult(
                source_uri=item.source_uri,
                title=item.title,
                source_type=item.source_type,
                chunk_text=chunk.text,
                score=score_map.get(chunk.id, 0.0),
                tags=tags_by_item.get(item.id, []),
                created_at=item.created_at,
            )
            for chunk, item in results_data
        ]

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
